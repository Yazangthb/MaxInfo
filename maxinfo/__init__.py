"""MaxInfo: training-free key-frame selection utilities.

Primary entrypoint for the notebook replication is:
  - [`maxinfo.pipeline.compare_sampling_on_video()`](maxinfo/pipeline.py:1)

This package is intentionally lightweight and notebook-friendly.
"""

from .pipeline import compare_sampling_on_video
from .video import encode_video_candidates, load_video_segment
from .selection import (
    uniform_select,
    maxinfo_select_rect_maxvol,
    rect_maxvol_greedy,
    svd_reduce_dim,
)

__all__ = [
    "compare_sampling_on_video",
    "load_video_segment",
    "encode_video_candidates",
    "uniform_select",
    "maxinfo_select_rect_maxvol",
    "rect_maxvol_greedy",
    "svd_reduce_dim",
]

