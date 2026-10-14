# MaxInfo (clean re-implementation)

Training-free key-frame selection for video LLMs by **rectangular maximum volume** on frame embeddings
([Li et al., arXiv:2502.03183](https://arxiv.org/abs/2502.03183)). It is rewritten from scratch because the official
release does not apply the method as published. See [ISSUES.md](ISSUES.md): 5 critical, 6 evaluation and 11 design
issues, each verified.

The headline issue: the official LLaVA-Video/Qwen2-VL notebooks hardcode `tol=0.1`, which makes MaxVol keep every candidate.
Their "MaxInfo" output is therefore **identical to uniform sampling** (measured on real CLIP features), even after fixing the syntax
error that stops them from running at all.

## Method

1. Sample n candidate frames; embed each with CLIP (or SigLIP/DINOv2) → Q ∈ ℝⁿˣᵈ.
2. Thin SVD, keep Q_s = U[:, :r] (r = 8).
3. Rectangular MaxVol on Q_s: square MaxVol for r rows, then greedily add the row with the largest coefficient norm
   ‖cᵢ‖ (C = Q_s·pinv(Q_s[piv])) until every remaining ‖cᵢ‖ ≤ tol (0.3). Each step multiplies the volume by √(1+‖cᵢ‖²).
4. If the selection exceeds the model's frame budget, keep the first `budget` rows in greedy order.
5. Give the VLLM the frames **and their timestamps**.

Fast version = candidates == budget. Slow version = candidates > budget (e.g. 128 → 64). Chunk-based = MaxVol per segment.

## Install

```bash
uv venv && uv pip install -e ".[encoders,video,eval,dev]"
```

The core (`maxinfo.selection`) needs only numpy. Windows note: import `torch` before `decord` (the package does this for you).

## Use

```python
from maxinfo import FrameEncoder, MaxInfo, MaxInfoConfig

mi = MaxInfo(FrameEncoder(), MaxInfoConfig(rank=8, tol=0.3))         # CLIP-L/336 by default
out = mi.from_video("video.mp4", num_candidates=128, budget=64)       # slow version
frames, times = out.video.pil(), out.video.timestamps                 # temporal order + seconds
```

Selection only, on your own features:

```python
from maxinfo import select_frames, chunked_select
sel = select_frames(features, budget=64, rank=8, tol=0.3)             # sel.indices (temporal), sel.order (greedy)
sel = chunked_select(features, n_chunks=32, budget=64)                # chunk-based MaxInfo
sel = select_frames(features, budget=180, group_size=2)               # Qwen2-VL: select frame pairs
```

| `MaxInfoConfig` | default | meaning |
|---|---|---|
| `rank` | 8 | SVD rank r |
| `tol` | 0.3 | MaxVol tolerance: smaller → more frames. Only matters when fewer than `budget` pivots are found. |
| `budget_mode` | `"greedy"` | `"uniform"` = the paper's even subsampling of pivots |
| `n_chunks` | None | e.g. 32 for chunk-based MaxInfo |
| `group_size` | 1 | 2 for Qwen2/2.5-VL (temporal patch size 2) |
| `center` | False | mean-centre features before SVD (the official code does not) |

## Evaluate

```python
from maxinfo import FrameEncoder, MaxInfo, MaxInfoConfig
from maxinfo.eval import load_videomme, run_mcq, score
from maxinfo.models import make_llava_video

items = load_videomme("videomme/test-00000-of-00001.parquet", "videomme/videos")
mi = MaxInfo(FrameEncoder(device="cuda"), MaxInfoConfig())
run_mcq(items, make_llava_video(maxinfo=mi, budget=64, num_candidates=128), "runs/llava7b_maxinfo.jsonl")
run_mcq(items, make_llava_video(maxinfo=None, budget=64), "runs/llava7b_uniform.jsonl")   # same pipeline, uniform
print(score("runs/llava7b_maxinfo.jsonl", by="duration"))
```

Runs are resumable, failures are logged with tracebacks (never scored as an answer), and the run aborts when
failures are systematic. Adapters: `make_llava_video`, `make_qwen2_vl`, `make_minicpm_v`. **They mirror each model's
official usage but have not been run end-to-end here (no suitable GPU).**

## Verify

```bash
pytest                                                    # 39 tests; set MAXVOLPY_SRC to compare against maxvolpy
python scripts/extract_features.py VIDEOS/ feats/ --num-frames 128
python scripts/compare_official.py feats/ --budget 64     # official-as-released vs intended vs this package
```

## Layout

```
maxinfo/selection.py   svd_basis, rect_maxvol (numpy), select_frames, chunked_select, uniform_select
maxinfo/encoders.py    FrameEncoder (CLIP/SigLIP/DINOv2, batched, transformers 4.x & 5.x)
maxinfo/video.py       candidate sampling with correct timestamps (decord)
maxinfo/pipeline.py    MaxInfo block (fast / slow / chunked)
maxinfo/prompts.py     time instruction + MCQ prompt
maxinfo/models.py      LLaVA-Video, Qwen2-VL, MiniCPM-V adapters
maxinfo/eval/          dataset loaders, answer parsing, resumable runner, scoring
```

## Citation

```bibtex
@article{li2025maxinfo,
  title={MaxInfo: A Training-Free Key-Frame Selection Method Using Maximum Volume for Enhanced Video Understanding},
  author={Li, Pengyi and Abdullaeva, Irina and Gambashidze, Alexander and Kuznetsov, Andrey and Oseledets, Ivan},
  journal={arXiv preprint arXiv:2502.03183},
  year={2025}
}
```
