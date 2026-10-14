"""MaxInfo: training-free key-frame selection by rectangular maximum volume (arXiv:2502.03183),
re-implemented with the issues of the official release fixed (see ISSUES.md)."""

from .pipeline import MaxInfo, MaxInfoConfig, MaxInfoOutput, uniform_baseline
from .selection import SelectionResult, chunked_select, rect_maxvol, select_frames, svd_basis, uniform_select
from .video import SampledVideo, candidate_indices, sample_video


def FrameEncoder(*args, **kwargs):  # lazy: torch/transformers are optional dependencies
    from .encoders import FrameEncoder as _FE

    return _FE(*args, **kwargs)


__all__ = [
    "MaxInfo", "MaxInfoConfig", "MaxInfoOutput", "uniform_baseline", "FrameEncoder",
    "SelectionResult", "chunked_select", "rect_maxvol", "select_frames", "svd_basis", "uniform_select",
    "SampledVideo", "candidate_indices", "sample_video",
]
