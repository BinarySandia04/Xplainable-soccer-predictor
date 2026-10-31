"""Goalkeeper and referee team rescue for pixel player evidence.

The ATD feed swaps or freezes its keeper slots while one goal is under
sustained pressure (measured on BAR-ATH 15 627-15 748: Barcelona's keeper slot
teleports into Athletic's goal area and *follows the Athletic keeper*, 0.6-10
px, while Athletic's own slot freezes 90-170 px away).  Team assignment by
nearest projected slot therefore hands the defending keeper to the attacking
club's colour.  The dedicated player model's ``goalkeeper``/``referee`` classes
settle this outright, but caches produced before the classes were persisted
threw them away; these helpers rebuild the truth from roster geometry instead:

* a keeper roster slot tracks *some* keeper even when its team id is wrong, and
  the keeper a box belongs to is decided by the goal he is standing in;
* referees are the only tracked people who never come near *any* ATD slot for
  their whole track lifetime -- players always match a slot in some frame.

``rescue_keeper_teams``/``rescue_officials`` fix existing npz caches; the
perception runner uses :func:`derive_keeper_slots` with the model classes for
new runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .perception import pitch_to_pixel

GOAL_X = (0.0, None)  # goal 0 = x=0 side, goal 1 = x=length side; filled from pitch


@dataclass(frozen=True)
class KeeperSlots:
    """Per graded half and goal: which ATD roster slot is the keeper, whose team."""

    slot: np.ndarray  # (halves, 2 goals) int slot index, -1 when not identifiable
    team: np.ndarray  # (halves, 2 goals) int defending team, -1 when unknown


def goal_side(x: float, length: float) -> int:
    """Which goal a pitch x-coordinate belongs to (left half of the pitch -> 0)."""
    return 0 if x < length / 2.0 else 1


def derive_keeper_slots(
    atd_players: np.ndarray,
    atd_teams: np.ndarray,
    pitch: tuple[float, float],
    halves: tuple[tuple[int, int], ...],
    min_valid_fraction: float = 0.2,
    max_keeper_distance: float = 22.0,
) -> KeeperSlots:
    """Identify each half's keeper slot per goal from the ATD roster geometry.

    The keeper is the slot whose median position in the half sits closest to a
    goal centre; a keeper must stay near the goal he defends, so slots that
    roam (substitutions are brief) never win, and sides are derived per half
    because the ends swap at half-time.
    """
    length, width = float(pitch[0]), float(pitch[1])
    centers = np.array([[0.0, width / 2.0], [length, width / 2.0]])
    teams = np.asarray(atd_teams, dtype=np.int8)
    slots = np.full((len(halves), 2), -1, dtype=np.int16)
    out_teams = np.full((len(halves), 2), -1, dtype=np.int8)
    n_frames = int(atd_players.shape[0])
    for h, (start, stop) in enumerate(halves):
        start, stop = max(0, int(start)), min(n_frames, int(stop))
        if stop - start < 250:
            continue
        window = np.asarray(atd_players[start:stop], dtype=np.float64)
        valid = np.isfinite(window).all(axis=2)  # (frames, slots)
        with np.errstate(invalid="ignore"):
            median = np.full((window.shape[1], 2), np.nan)
            for p in range(window.shape[1]):
                rows = window[:, p][valid[:, p]]
                if rows.size and valid[:, p].mean() >= min_valid_fraction:
                    median[p] = np.median(rows, axis=0)
        dist = np.linalg.norm(median[:, None, :] - centers[None, :, :], axis=2)  # (slots, goals)
        dist[~np.isfinite(dist).all(axis=1)] = np.inf
        for g in (0, 1):
            p = int(np.argmin(dist[:, g]))
            if np.isfinite(dist[p, g]) and dist[p, g] <= max_keeper_distance and teams[p] >= 0:
                slots[h, g] = p
                out_teams[h, g] = int(teams[p])
    return KeeperSlots(slot=slots, team=out_teams)


def half_index(frame: int, halves: tuple[tuple[int, int], ...]) -> int:
    for h, (start, stop) in enumerate(halves):
        if start <= frame < stop:
            return h
    return -1


def rescue_keeper_teams(
    players_pixel: np.ndarray,
    players_pitch: np.ndarray,
    players_present: np.ndarray,
    players_team: np.ndarray,
    atd_players: np.ndarray,
    matrices: np.ndarray,
    keepers: KeeperSlots,
    halves: tuple[tuple[int, int], ...],
    pitch: tuple[float, float],
    claim_radius_px: float = 12.0,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Fix keeper teams in a cached npz (independent copy returned).

    A stored box claiming a keeper slot (within ``claim_radius_px`` of its
    projection) is the keeper visible at that moment; his team is the one that
    defends the goal he is standing in.  This survives both ATD failure modes:
    the slot that teleports and follows the wrong keeper, and the slot frozen
    in an empty area (its stale claimer, if any, is resolved by goal side too).
    Returns the fixed team array and ``[(frame, row, old, new), ...]`` flips.
    """
    teams = np.array(players_team, copy=True)
    present = np.asarray(players_present, dtype=bool)
    pixels = np.asarray(players_pixel, dtype=np.float64)
    pitch_xy = np.asarray(players_pitch, dtype=np.float64)
    length = float(pitch[0])
    flips: list[tuple[int, int, int, int]] = []
    n_frames = int(teams.shape[0])
    n_slots = int(keepers.slot.shape[1])
    for f in range(n_frames):
        h = half_index(f, halves)
        if h < 0:
            continue
        matrix = np.asarray(matrices[f], dtype=np.float64)
        rows = np.flatnonzero(present[f] & np.isfinite(pixels[f]).all(axis=1))
        if rows.size == 0:
            continue
        claimed: dict[int, tuple[float, int]] = {}  # goal -> (distance, row)
        for g in range(n_slots):
            slot = int(keepers.slot[h, g])
            if slot < 0:
                continue
            pos = np.asarray(atd_players[f, slot], dtype=np.float64)
            if not np.isfinite(pos).all():
                continue
            proj = pitch_to_pixel(matrix, pos, pitch)
            d = np.linalg.norm(pixels[f, rows] - proj[None, :], axis=1)
            near = np.flatnonzero(d <= claim_radius_px)
            if near.size == 0:
                continue
            row = int(rows[near[int(np.argmin(d[near]))]])
            foot = pitch_xy[f, row]
            if not np.isfinite(foot).all():
                continue
            side = goal_side(float(foot[0]), length)
            # measure against the goal the box stands nearest, not the goal the
            # claiming slot nominally guards: under the ATD confusion both keeper
            # slots can sit on one side, and each slot's own goal is then a lie
            # (measured 19 863: the Athletic keeper slot glued to the holding
            # Barcelona keeper lost its claim 94-22 to Barça's frozen slot).
            dist_goal = float(np.hypot(foot[0] - side * length, 0.0))
            key = side
            if key not in claimed or dist_goal < claimed[key][0]:
                claimed[key] = (dist_goal, row)
        for side, (_, row) in claimed.items():
            right = int(keepers.team[h, side])
            if right >= 0 and int(teams[f, row]) != right:
                flips.append((f, row, int(teams[f, row]), right))
                teams[f, row] = right
    return teams, flips


def rescue_officials(
    players_pixel: np.ndarray,
    players_present: np.ndarray,
    players_track: np.ndarray,
    atd_players: np.ndarray,
    matrices: np.ndarray,
    pitch: tuple[float, float],
    min_appearances: int = 25,
    slot_free_px: float = 50.0,
    min_free_fraction: float = 0.85,
) -> np.ndarray:
    """Flag referee boxes: tracks that never come near any ATD slot, ever.

    Players (even mislabelled ones) match some roster slot in many frames;
    officials are in neither roster, so across a track's whole lifetime the
    fraction of frames whose foot point stays ``slot_free_px`` away from every
    projected slot stays near 1.  ``players_official`` bool (n, slots).
    """
    n, p = players_present.shape
    official = np.zeros((n, p), dtype=bool)
    present = np.asarray(players_present, dtype=bool)
    pixels = np.asarray(players_pixel, dtype=np.float64)
    tracks = np.asarray(players_track)
    atd = np.asarray(atd_players, dtype=np.float64)
    # accumulate per track: appearances and frames never near a slot
    track_seen: dict[int, int] = {}
    track_free: dict[int, int] = {}
    for f in range(n):
        matrix = np.asarray(matrices[f], dtype=np.float64)
        atd_valid = np.isfinite(atd[f]).all(axis=1)
        if not atd_valid.any():
            continue
        projected = pitch_to_pixel(matrix, atd[f][atd_valid], pitch)
        rows = np.flatnonzero(present[f] & np.isfinite(pixels[f]).all(axis=1))
        if rows.size == 0:
            continue
        dists = np.linalg.norm(pixels[f, rows][:, None, :] - projected[None, :, :], axis=2)
        nearest = dists.min(axis=1)
        for row, d in zip(rows, nearest):
            tid = int(tracks[f, row])
            if tid < 0:
                continue
            track_seen[tid] = track_seen.get(tid, 0) + 1
            if d > slot_free_px:
                track_free[tid] = track_free.get(tid, 0) + 1
    for tid, seen in track_seen.items():
        free = track_free.get(tid, 0)
        if seen >= min_appearances and free / seen >= min_free_fraction:
            official |= present & (tracks == tid)
    return official
