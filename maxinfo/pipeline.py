"""End-to-end MaxInfo block: sample candidates -> embed -> select -> frames + timestamps.

Fast version: num_candidates == budget (MaxInfo only drops redundant frames).
Slow version: num_candidates > budget (oversample, then select up to `budget`).
Selected frames always carry their timestamps so the caller can tell the VLLM where they are
(MaxInfo frames are not evenly spaced; see `maxinfo.prompts`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .selection import SelectionResult, chunked_select, select_frames, uniform_select
from .video import SampledVideo, sample_video


@dataclass
class MaxInfoConfig:
    rank: int = 8
    tol: float = 0.3
    center: bool = False
    budget_mode: str = "greedy"  # "uniform" = paper's even subsampling of pivots
    n_chunks: int | None = None  # set (e.g. 32) for chunk-based MaxInfo
    group_size: int = 1  # 2 for Qwen2/2.5-VL (frame pairs share one temporal patch)
    min_frames: int = 1

    def select(self, features: np.ndarray, budget: int | None) -> SelectionResult:
        if self.n_chunks:
            return chunked_select(features, n_chunks=self.n_chunks, budget=budget, rank=self.rank, tol=self.tol,
                                  center=self.center, group_size=self.group_size)
        return select_frames(features, budget=budget, rank=self.rank, tol=self.tol, center=self.center,
                             budget_mode=self.budget_mode, group_size=self.group_size, min_frames=self.min_frames)


@dataclass
class MaxInfoOutput:
    video: SampledVideo  # selected frames (temporal order) with indices/timestamps
    candidates: SampledVideo
    selection: SelectionResult
    features: np.ndarray = field(repr=False)


class MaxInfo:
    """Example:
        mi = MaxInfo(FrameEncoder(), MaxInfoConfig(rank=8, tol=0.3))
        out = mi.from_video("v.mp4", num_candidates=128, budget=64)   # slow version
        frames, times = out.video.pil(), out.video.timestamps
    """

    def __init__(self, encoder, config: MaxInfoConfig | None = None):
        self.encoder = encoder
        self.config = config or MaxInfoConfig()

    def select(self, candidates: SampledVideo, budget: int | None) -> MaxInfoOutput:
        feats = self.encoder(candidates.pil())
        sel = self.config.select(feats, budget)
        return MaxInfoOutput(candidates.subset(sel.indices), candidates, sel, feats)

    def from_video(self, path: str, *, budget: int, num_candidates: int | None = None,
                   fps: float | None = None, max_candidates: int | None = None) -> MaxInfoOutput:
        if num_candidates is None and fps is None:
            num_candidates = budget  # fast version
        cand = sample_video(path, num_frames=num_candidates, fps=fps, max_frames=max_candidates)
        return self.select(cand, budget)


def uniform_baseline(path: str, budget: int) -> SampledVideo:
    """Uniform sampling with the same decoder, for like-for-like comparisons."""
    return sample_video(path, num_frames=budget)


__all__ = ["MaxInfo", "MaxInfoConfig", "MaxInfoOutput", "uniform_baseline", "uniform_select"]
