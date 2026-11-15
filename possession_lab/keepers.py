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


def _duplicates_inherit_keeper_team(
    teams: np.ndarray,
    present: np.ndarray,
    pitch_xy: np.ndarray,
    tracks: np.ndarray,
    resolved: dict[tuple[int, int], tuple[int, int]],
    track_rows: dict[int, tuple[np.ndarray, np.ndarray]],
    halves: tuple[tuple[int, int], ...],
    length: float,
    zone_m: float,
    radius_m: float,
    flips: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Hand the keeper's team to duplicated detections of the keeper.

    The players model occasionally emits the same body twice (measured 115 384:
    two tracks, IoU 0.98, identical feet at 0.0 m).  Two boxes on one spot are one
    person, so a track that spends most of its life on a resolved keeper's spot
    is that keeper's duplicate and wears his team; the duplicate that missed the
    slot claim would otherwise keep the roster's nominal (wrong) shirt, and its
    box at the ball would count as a touch by the wrong side.
    """
    n = teams.shape[0]
    keeper_xy = np.full((n, 2, 2), np.nan)
    for (track_id, h), (side, _right) in resolved.items():
        start_h, stop_h = halves[h]
        goal_x = 0.0 if side == 0 else length
        fr, rw = track_rows.get(track_id, (np.empty(0, dtype=int), np.empty(0, dtype=int)))
        sel = (fr >= start_h) & (fr < stop_h) & (np.abs(pitch_xy[fr, rw, 0] - goal_x) <= zone_m)
        keeper_xy[fr[sel], side] = pitch_xy[fr[sel], rw[sel]]
    valid = present & np.isfinite(pitch_xy).all(axis=2) & (tracks >= 0)
    if not valid.any():
        return teams
    size = int(tracks[valid].max()) + 1
    tr = np.where(valid, tracks, -1)
    count = np.bincount(tr[valid], minlength=size)
    hit = np.zeros((2, size), dtype=np.int64)
    for s in range(0, n, 20_000):
        e = min(n, s + 20_000)
        with np.errstate(invalid="ignore"):
            d = np.linalg.norm(pitch_xy[s:e, :, None, :] - keeper_xy[s:e, None, :, :], axis=3)
        near = np.isfinite(d) & (d <= radius_m)
        v = valid[s:e]
        for side in (0, 1):
            hit[side] += np.bincount(tr[s:e][v & near[:, :, side]], minlength=size)
    duplicate = np.zeros(size, dtype=bool)
    for side in (0, 1):
        duplicate |= (count >= 3) & (hit[side] >= 0.8 * count)
    for track_id, _h in resolved:
        if 0 <= track_id < size:
            duplicate[track_id] = False
    if not duplicate.any():
        return teams
    # which keeper each duplicate stands on, and the team that keeper wears in the
    # half: the duplicate is the same body, whether or not the main track was seen
    # on that particular frame (measured 115 396: the keeper's only box that frame
    # was the duplicate, so a same-frame match would have missed it)
    side_of = np.where(hit[1] > hit[0], 1, 0)
    right_by_half_side: dict[tuple[int, int], int] = {}
    for (_track_id, h), (side, right) in resolved.items():
        right_by_half_side.setdefault((h, side), right)
    safe = np.where(valid, tracks, 0)
    ii, jj = np.nonzero(valid & duplicate[safe])
    for i, j in zip(ii.tolist(), jj.tolist()):
        h = half_index(i, halves)
        if h < 0:
            continue
        side = int(side_of[int(tracks[i, j])])
        right = right_by_half_side.get((h, side), -1)
        if right < 0:
            continue
        goal_x = 0.0 if side == 0 else length
        if abs(float(pitch_xy[i, j, 0]) - goal_x) > zone_m:
            continue
        if int(teams[i, j]) != right:
            flips.append((i, j, int(teams[i, j]), right))
            teams[i, j] = right
    return teams


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
    players_track: np.ndarray | None = None,
    min_track_claims: int = 8,
    track_zone_m: float = 20.0,
    duplicate_radius_m: float = 0.3,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Fix keeper teams in a cached npz (independent copy returned).

    A stored box claiming a keeper slot (within ``claim_radius_px`` of its
    projection) is the keeper visible at that moment; his team is the one that
    defends the goal he is standing in.  This survives both ATD failure modes:
    the slot that teleports and follows the wrong keeper, and the slot frozen
    in an empty area (its stale claimer, if any, is resolved by goal side too).

    With ``players_track`` the rescue is decided **per track and half**, not per
    frame: a keeper defends the same goal for a whole half, so a track claimed
    often enough inside that goal's zone (``min_track_claims`` claims within
    ``track_zone_m`` of the goal line) is handed the defending team for every
    frame it appears there.  The per-frame radius alone made the same keeper
    flicker 0/1 as his distance to the projection crossed it (measured 115 421:
    18.1 px, just outside 12 px, after 70 frames claimed as team 1).

    Keeper detections are sometimes duplicated (two tracks on the same box).  A
    track that stands on the resolved keeper's spot for most of its life is the
    same body and inherits his team (measured 115 384: track 15 096 duplicated
    the keeper's box at 0.0 m wearing Barcelona's team, and the box at the ball
    then counted as a touch by the wrong side).

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
    tracks = None if players_track is None else np.asarray(players_track)
    claims: list[tuple[int, int, int, float, int, int]] = []
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
        for side, (dist_goal, row) in claimed.items():
            right = int(keepers.team[h, side])
            if right < 0:
                continue
            track_id = int(tracks[f, row]) if tracks is not None else -1
            claims.append((f, row, h, dist_goal, right, track_id))
    # Track stabilisation: one person keeps one team for a whole half.  A track
    # claimed on the keeper projection often enough inside a goal's zone is the
    # keeper of that goal, whatever the per-frame radius says on any one frame.
    resolved: dict[tuple[int, int], tuple[int, int]] = {}
    track_rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if tracks is not None and claims:
        by_track: dict[tuple[int, int], dict[int, list[int]]] = {}
        for f, row, h, dist_goal, _right, track_id in claims:
            if track_id < 0 or dist_goal > track_zone_m:
                continue
            side = goal_side(float(pitch_xy[f, row, 0]), length)
            by_track.setdefault((track_id, h), {}).setdefault(side, []).append(f)
        if by_track:
            mask = present & np.isfinite(pitch_xy).all(axis=2) & (tracks >= 0)
            nn, kk = np.nonzero(mask)
            trv = tracks[nn, kk]
            order = np.argsort(trv, kind="stable")
            nn, kk, trv = nn[order], kk[order], trv[order]
            uniq, starts = np.unique(trv, return_index=True)
            ends = np.r_[starts[1:], len(trv)]
            track_rows = {int(u): (nn[s:e], kk[s:e]) for u, s, e in zip(uniq, starts, ends)}
        for (track_id, h), sides in by_track.items():
            side, frames = max(sides.items(), key=lambda kv: len(kv[1]))
            if len(frames) < min_track_claims:
                continue
            right = int(keepers.team[h, side])
            if right < 0:
                continue
            resolved[(track_id, h)] = (side, right)
            start_h, stop_h = halves[h]
            goal_x = 0.0 if side == 0 else length
            fr, rw = track_rows.get(track_id, (np.empty(0, dtype=int), np.empty(0, dtype=int)))
            if not fr.size:
                continue
            sel = (fr >= start_h) & (fr < stop_h) & (np.abs(pitch_xy[fr, rw, 0] - goal_x) <= track_zone_m)
            for f, row in zip(fr[sel], rw[sel]):
                if int(teams[f, row]) != right:
                    flips.append((int(f), int(row), int(teams[f, row]), right))
                    teams[f, row] = right
    for f, row, h, _dist_goal, right, track_id in claims:
        if track_id >= 0 and (track_id, h) in resolved:
            continue
        if int(teams[f, row]) != right:
            flips.append((f, row, int(teams[f, row]), right))
            teams[f, row] = right
    if tracks is not None and resolved and duplicate_radius_m > 0:
        teams = _duplicates_inherit_keeper_team(
            teams, present, pitch_xy, tracks, resolved, track_rows, halves,
            length, track_zone_m, duplicate_radius_m, flips,
        )
    flips.sort(key=lambda item: (item[0], item[1]))
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
