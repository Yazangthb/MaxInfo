from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class VideoSegment:
    start_sec: float
    end_sec: float


def load_video_segment(
    video_path: str,
    fps_sample: float = 1.0,
    max_candidates: int = 256,
    start_sec: float | None = None,
    end_sec: float | None = None,
    force_sample: bool = True,
    num_threads: int = 1,
):
    """Decode a video segment and sample frames at approximately `fps_sample`.

    Returns:
        frames_np: np.ndarray (N, H, W, 3) RGB uint8
        timestamps: list[float] (seconds)
        orig_frame_idx: list[int] (indices in original video)
        segment: (start_sec, end_sec) actually used
        duration: float seconds
        native_fps: float
    """

    import os

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if fps_sample <= 0:
        raise ValueError("fps_sample must be > 0")

    if max_candidates <= 0:
        raise ValueError("max_candidates must be > 0")

    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=int(num_threads))
    native_fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    duration = total_frames / native_fps if native_fps > 0 else 0.0

    s = 0.0 if start_sec is None else max(0.0, float(start_sec))
    e = duration if end_sec is None else min(duration, float(end_sec))
    if e <= s:
        raise ValueError(f"Invalid segment: start_sec={s}, end_sec={e}, duration={duration:.2f}")

    start_f = int(round(s * native_fps))
    end_f = int(round(e * native_fps))
    end_f = min(end_f, total_frames)

    step = max(1, int(round(native_fps / float(fps_sample))))
    frame_idx = list(range(start_f, end_f, step))

    if force_sample and len(frame_idx) > int(max_candidates):
        import numpy as np

        frame_idx = np.linspace(start_f, end_f - 1, int(max_candidates), dtype=int).tolist()

    if len(frame_idx) == 0:
        raise ValueError("No frames selected (segment too small or fps_sample too low).")

    frames_np = vr.get_batch(frame_idx).asnumpy()  # RGB uint8
    timestamps = [i / native_fps for i in frame_idx]
    return frames_np, timestamps, frame_idx, (s, e), duration, native_fps


def _temporal_ids_from_timestamps(
    timestamps: Iterable[float],
    time_scale: float,
    *,
    clip_min: int = 0,
):
    import numpy as np

    ts = np.asarray(list(timestamps), dtype=np.float32)
    if ts.size == 0:
        return np.zeros((0,), dtype=np.int32)

    if time_scale <= 0:
        raise ValueError("time_scale must be > 0")

    # Nearest tick id; equivalent to snapping to np.arange(..., step=time_scale)
    tid = np.rint(ts / float(time_scale)).astype(np.int32)
    if clip_min is not None:
        tid = np.maximum(tid, int(clip_min))
    return tid


def encode_video_candidates(
    video_path: str,
    choose_fps: float = 3.0,
    max_candidates: int = 256,
    start_sec: float | None = None,
    end_sec: float | None = None,
    time_scale: float = 0.1,
):
    """Experiment pipeline 2 candidate encoding.

    This mirrors the notebook behavior:
      - decode candidates (optionally a segment)
      - convert to PIL
      - build per-frame temporal id groups (singletons)

    Returns:
        frames: list[PIL.Image.Image]
        ts_group: list[np.ndarray] (each is shape (1,), int32)
        orig_frame_idx: list[int]
        timestamps: list[float]
    """

    import numpy as np
    from PIL import Image

    frames_np, timestamps, orig_frame_idx, (_s, _e), _duration, _native_fps = load_video_segment(
        video_path=video_path,
        fps_sample=choose_fps,
        max_candidates=max_candidates,
        start_sec=start_sec,
        end_sec=end_sec,
        force_sample=True,
    )

    frames = [Image.fromarray(x.astype("uint8")).convert("RGB") for x in frames_np]

    tids = _temporal_ids_from_timestamps(timestamps, time_scale=time_scale)
    ts_group = [np.array([t], dtype=np.int32) for t in tids]
    return frames, ts_group, orig_frame_idx, timestamps

