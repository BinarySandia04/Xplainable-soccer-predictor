"""Referee-signal detection from the tactical video's audio track.

The tracking data carries no referee object and the smart tags no whistle code, but the tactical
MP4 has a full audio stream, and the referee's whistle is the one signal that *defines* a stoppage.
It is recorded quietly on this feed (about -43 dBFS), so detection works on relative spectral
shape, not level: a whistle is a sustained tonal peak in the 2.4-3.8 kHz range that stands out
against the low-frequency crowd bed.

Measured on the public ground truth: inside the four stoppage windows of the first five minutes the
band score peaks at 5.7-10.9, against 1.9-3.2 in live-play controls, and lands within +/-2 s of the
``D``-run boundary.  That makes the whistle an event trigger with roughly +/-1 s accuracy; the
tracking remains the frame-accurate authority.

The extractor streams the MP4 through ffmpeg (``imageio-ffmpeg``'s static binary, or a system
``ffmpeg``), computes the per-video-frame score, and caches it under ``outputs/whistle.npz``.  When
the video or ffmpeg is unavailable everything degrades to an empty event set, which with default
weights is a no-op for the model.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (
    AUDIO_SR,
    BAND_HI,
    BAND_LO,
    ENVELOPE_TAU_S,
    FFT_HOP,
    FFT_SIZE,
    FPS,
    LOW_HI,
    LOW_LO,
    MERGE_GAP_S,
    MIN_EVENT_S,
    N_FRAMES,
    OUTPUT_DIR,
    W_LOCAL_RATIO,
    W_MIN,
    WHISTLE_CACHE,
)

# Number of STFT frames transformed at once.  The full match is ~1.2 million frames; materialising
# the whole spectrum would need several gigabytes, so the score is computed in bounded blocks.
_STFT_BLOCK = 20_000


def ffmpeg_binary() -> str | None:
    """Path to a usable ffmpeg: the static one shipped with imageio-ffmpeg, else the system one."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def decode_audio(
    video_path: str | Path,
    ff: str | None = None,
    chunk: int = 4 * 1024 * 1024,
) -> np.ndarray:
    """Decode the video's audio to mono float32 at ``AUDIO_SR``.  Streams through ffmpeg's pipe."""
    ff = ff or ffmpeg_binary()
    if not ff:
        raise RuntimeError("no ffmpeg available: install imageio-ffmpeg or ffmpeg on PATH")
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    proc = subprocess.Popen(
        [
            ff,
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_SR),
            "-f",
            "f32le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    chunks: list[np.ndarray] = []
    while True:
        piece = proc.stdout.read(chunk)
        if not piece:
            break
        chunks.append(np.frombuffer(piece, dtype="<f4"))
    proc.wait()
    if proc.returncode != 0 or not chunks:
        raise RuntimeError(
            f"ffmpeg failed to decode audio from {video_path} (code {proc.returncode})"
        )
    return np.concatenate(chunks)


def band_score(audio: np.ndarray, sr: int = AUDIO_SR) -> np.ndarray:
    """Per-STFT-frame whistle-band score: band peakiness x band-to-low energy ratio."""
    if audio.size < FFT_SIZE:
        return np.zeros(0, dtype=np.float32)
    window = np.hanning(FFT_SIZE).astype(np.float32)
    n = 1 + (audio.size - FFT_SIZE) // FFT_HOP
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sr)
    band = (freqs >= BAND_LO) & (freqs <= BAND_HI)
    low = (freqs >= LOW_LO) & (freqs <= LOW_HI)
    values = np.empty(n, dtype=np.float32)
    view = np.lib.stride_tricks.sliding_window_view(audio, FFT_SIZE)
    for start in range(0, n, _STFT_BLOCK):
        stop = min(n, start + _STFT_BLOCK)
        count = stop - start
        frames = view[
            start * FFT_HOP : start * FFT_HOP + (count - 1) * FFT_HOP + 1 : FFT_HOP
        ]
        spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
        band_energy = spectrum[:, band].sum(axis=1)
        low_energy = spectrum[:, low].sum(axis=1)
        peakiness = spectrum[:, band].max(axis=1) / (
            spectrum[:, band].mean(axis=1) + 1e-12
        )
        ratio = band_energy / (low_energy + 1e-12)
        values[start:stop] = (peakiness * np.clip(ratio, 0.0, None)).astype(np.float32)
    return values


def score_per_video_frame(
    audio: np.ndarray, n_frames: int, sr: int = AUDIO_SR
) -> np.ndarray:
    """Collapse STFT-frame scores onto video frames (max over the STFT frames covering each)."""
    score = np.zeros(n_frames, dtype=np.float32)
    values = band_score(audio, sr)
    if values.size == 0:
        return score
    times = (
        np.arange(values.size) * FFT_HOP + FFT_SIZE / 2.0
    ) / sr
    frames = np.floor(times * FPS).astype(np.int64)
    inside = frames < n_frames
    np.maximum.at(score, frames[inside], values[inside])
    return score


def _local_median(score: np.ndarray, span: int) -> np.ndarray:
    """Running median over +-``span`` frames: median on a coarse grid, then interpolated."""
    n = score.size
    step = max(1, span)
    nodes = np.arange(0, n + step, step, dtype=np.float64)
    med = np.empty(nodes.shape, dtype=np.float64)
    for i, center in enumerate(nodes):
        lo, hi = max(0, int(center) - span), min(n, int(center) + span)
        med[i] = np.median(score[lo:hi]) if hi > lo else 0.0
    return np.interp(np.arange(n, dtype=np.float64), nodes, med).astype(np.float32)


def detect_events(score: np.ndarray, n_frames: int) -> list[dict]:
    """Whistle events from a per-video-frame score: thresholded above the local crowd baseline.

    Blasts are runs above the floor, joined only across dropouts of ``MERGE_GAP_S`` or less (a
    tracker-style hole in the score, not a second toot: separate toots stay separate events, which
    is what the annotation wants).  Returns dicts with ``frame`` (the peak frame), ``start``/
    ``stop`` (half-open), ``peak`` and ``width_s``.
    """
    score = np.asarray(score, dtype=np.float32)[:n_frames]
    if score.size == 0:
        return []
    floor = np.maximum(W_MIN, W_LOCAL_RATIO * _local_median(score, 120))
    above = score >= floor
    join = int(round(MERGE_GAP_S * FPS))
    edges = np.flatnonzero(np.diff(np.concatenate([[False], above, [False]])))
    merged: list[list[int]] = []
    for start, stop in zip(edges[::2], edges[1::2]):
        if merged and start - merged[-1][1] <= join:
            merged[-1][1] = stop
        else:
            merged.append([int(start), int(stop)])

    events: list[dict] = []
    for start, stop in merged:
        run = score[start:stop]
        if run.size < MIN_EVENT_S * FPS:
            continue
        events.append(
            {
                "frame": int(start + int(np.argmax(run))),
                "start": int(start),
                "stop": int(stop),
                "peak": float(run.max()),
                "width_s": float(run.size / FPS),
            }
        )
    return events


def envelope(
    events: list[dict],
    n_frames: int,
    tau_pre_s: float | None = None,
    tau_post_s: float | None = None,
) -> np.ndarray:
    """0..1 per-frame support for the dead-ball state around whistle events.

    Asymmetric exponential tails: ``tau_pre`` opens support *before* the blast (the ref signals a
    stoppage as the ball is still travelling out), ``tau_post`` closes it after.  A tight post-tail
    keeps the restart whistle from claiming live frames just after the throw.
    """
    out = np.zeros(n_frames, dtype=np.float32)
    tau_pre = (tau_pre_s if tau_pre_s is not None else ENVELOPE_TAU_S) * FPS
    tau_post = (tau_post_s if tau_post_s is not None else ENVELOPE_TAU_S) * FPS
    for event in events:
        lo = max(0, event["start"] - int(4 * tau_pre))
        hi = min(n_frames, event["stop"] + int(4 * tau_post))
        idx = np.arange(lo, hi)
        dt_before = np.maximum(0, event["start"] - idx) / tau_pre
        dt_after = np.maximum(0, idx - event["stop"] + 1) / tau_post
        value = np.exp(-(dt_before + dt_after))
        np.maximum(out[lo:hi], value, out=out[lo:hi])
    return out


@dataclass(frozen=True)
class WhistleData:
    """Score envelope plus detected events, aligned to tactical video frames."""

    score: np.ndarray
    env: np.ndarray
    events: tuple[dict, ...]
    meta: dict

    def __len__(self) -> int:
        return int(self.score.size)


def _events_from_arrays(
    frames: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    peaks: np.ndarray,
    widths: np.ndarray,
) -> tuple[dict, ...]:
    return tuple(
        {
            "frame": int(frame),
            "start": int(start),
            "stop": int(stop),
            "peak": float(peak),
            "width_s": float(width),
        }
        for frame, start, stop, peak, width in zip(frames, starts, stops, peaks, widths)
    )


def load_or_extract(
    video_path: str | Path | None,
    n_frames: int = N_FRAMES,
    cache_path: str | Path = WHISTLE_CACHE,
    force: bool = False,
) -> WhistleData:
    """Load the cached whistle table; extract (and cache) from the video when missing.

    Never raises when audio is simply unavailable: it returns an empty ``WhistleData`` so the
    pipeline runs unchanged without the video or ffmpeg.
    """
    cache_path = Path(cache_path)
    if cache_path.is_file() and not force:
        try:
            blob = np.load(cache_path, allow_pickle=False)
            meta = json.loads(str(blob["meta_json"]))
            if int(meta.get("n_frames", -1)) == int(n_frames):
                events = _events_from_arrays(
                    blob["frames"],
                    blob["starts"],
                    blob["stops"],
                    blob["peaks"],
                    blob["widths"],
                )
                return WhistleData(
                    score=blob["score"], env=blob["env"], events=events, meta=meta
                )
        except Exception:
            pass

    empty_meta = {"n_frames": int(n_frames), "source": None, "available": False}
    try:
        audio = decode_audio(video_path) if video_path is not None else None
    except Exception:
        audio = None
    if audio is None:
        return WhistleData(
            score=np.zeros(n_frames, dtype=np.float32),
            env=np.zeros(n_frames, dtype=np.float32),
            events=(),
            meta=empty_meta,
        )

    score = score_per_video_frame(audio, n_frames)
    events = detect_events(score, n_frames)
    env = envelope(events, n_frames)
    meta = {
        "n_frames": int(n_frames),
        "source": str(video_path),
        "available": True,
        "audio_seconds": float(audio.size / AUDIO_SR),
        "n_events": len(events),
        "band_hz": [BAND_LO, BAND_HI],
        "min_event_s": MIN_EVENT_S,
        "w_min": W_MIN,
        "w_local_ratio": W_LOCAL_RATIO,
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            score=score,
            env=env,
            frames=np.array([e["frame"] for e in events], dtype=np.int64),
            starts=np.array([e["start"] for e in events], dtype=np.int64),
            stops=np.array([e["stop"] for e in events], dtype=np.int64),
            peaks=np.array([e["peak"] for e in events], dtype=np.float32),
            widths=np.array([e["width_s"] for e in events], dtype=np.float32),
            meta_json=json.dumps(meta),
        )
    except OSError:
        pass
    return WhistleData(score=score, env=env, events=tuple(events), meta=meta)
