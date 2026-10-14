"""Candidate frame sampling with correct timestamps.

Fixes relative to the official `load_video` (copied from LLaVA-NeXT):
- timestamps are always frame_index / native_fps (the original divides by the *step*);
- indices are unique (np.linspace on a short video repeats frames, giving duplicate rows);
- the sampling rule is explicit: either a target fps or a fixed count, capped by `max_frames`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SampledVideo:
    frames: np.ndarray  # [n, H, W, 3] uint8 RGB
    indices: np.ndarray  # frame indices in the source video
    timestamps: np.ndarray  # seconds
    native_fps: float
    duration: float

    def subset(self, idx) -> "SampledVideo":
        idx = np.asarray(idx, dtype=int)
        return SampledVideo(self.frames[idx], self.indices[idx], self.timestamps[idx], self.native_fps, self.duration)

    def pil(self):
        from PIL import Image

        return [Image.fromarray(f) for f in self.frames]


def candidate_indices(
    total_frames: int,
    native_fps: float,
    *,
    num_frames: int | None = None,
    fps: float | None = None,
    max_frames: int | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> np.ndarray:
    """Pick candidate frame indices: `num_frames` evenly spaced, or one every 1/fps seconds.
    Exactly one of num_frames / fps must be given. Frames are taken at bin centres."""
    if (num_frames is None) == (fps is None):
        raise ValueError("give exactly one of num_frames or fps")
    if total_frames <= 0 or native_fps <= 0:
        raise ValueError(f"invalid video: total_frames={total_frames}, fps={native_fps}")
    lo = 0 if start_sec is None else max(0, int(round(start_sec * native_fps)))
    hi = total_frames if end_sec is None else min(total_frames, int(round(end_sec * native_fps)))
    if hi <= lo:
        raise ValueError(f"empty segment [{start_sec}, {end_sec}]")
    span = hi - lo
    if fps is not None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        num_frames = max(1, int(round(span / native_fps * fps)))
    if max_frames is not None:
        num_frames = min(int(num_frames), int(max_frames))
    num_frames = int(max(1, min(num_frames, span)))
    edges = np.linspace(lo, hi, num_frames + 1)
    idx = np.floor((edges[:-1] + edges[1:]) / 2).astype(int)
    return np.unique(np.clip(idx, lo, hi - 1))


def sample_video(path: str, **kwargs) -> SampledVideo:
    """Decode candidate frames with decord. kwargs as in `candidate_indices`."""
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:  # On Windows, loading decord before torch makes torch's DLLs fail to initialise (WinError 1114).
        import torch  # noqa: F401
    except ImportError:
        pass
    from decord import VideoReader, cpu

    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    native_fps = float(vr.get_avg_fps())
    total = len(vr)
    idx = candidate_indices(total, native_fps, **kwargs)
    frames = vr.get_batch(idx.tolist()).asnumpy()
    return SampledVideo(frames, idx, idx / native_fps, native_fps, total / native_fps)
