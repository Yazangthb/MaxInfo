# Issues in the official MaxInfo release, verified

Audited: the 5 notebooks in [FusionBrainLab/MaxInfo](https://github.com/FusionBrainLab/MaxInfo) (`upstream/main`)
and the paper (arXiv:2502.03183v3, 29 Dec 2025). Every item says **how it was verified**. "Measured" numbers
come from real CLIP-L/336 features of 9 local videos (128 uniform candidates per video; medians), using the reference
`maxvolpy` 0.3.8 `rect_maxvol` that the notebooks import. Reproduce with `scripts/extract_features.py` +
`scripts/compare_official.py`.

Status: ✅ fixed in this package · 📝 documented / configurable · ⛔ not fixable in code (paper)

## A. The method is not actually applied (critical)

| # | Issue | Where | Verified by | Status |
|---|---|---|---|---|
| A1 | `max_vol_extract_index(initail_M, tol, …)` ignores its `tol` argument and calls `rect_maxvol(initail_M, tol = 0.1)`. The caller's `tol = 0.30` (the paper's best value) never takes effect. | LLaVA fast/slow, Qwen fast/slow | Code reading | ✅ `tol` is a real parameter (`MaxInfoConfig.tol`) |
| A2 | Consequence of A1: with rank 8 and tol 0.1, rect-MaxVol keeps **every** candidate (it needs about r/tol² ≈ 800 rows to satisfy the tolerance). The fast version therefore passes all 64 frames unchanged. The slow version keeps all 128 and then subsamples them evenly to 64, which is **identical to uniform sampling**. | LLaVA, Qwen | Measured: 128/128 kept on every video; selected indices 100% identical to uniform-64, same log-volume (−2.78) and same max gap | ✅ test `test_small_tol_selects_everything` documents it; defaults are tol 0.3, rank 8 |
| A3 | Syntax error: `svd_reduce_dim(image_features, r = 8` (missing `)`), so the cell defining MaxInfo does not run. | LLaVA fast/slow, Qwen fast/slow | `pyflakes`: "'(' was never closed" in all 4 | ✅ |
| A4 | `torch` and `CLIPModel` are never imported (NameError). | Qwen fast/slow | `pyflakes` (after fixing A3) | ✅ |
| A5 | Slow Qwen calls `reduce_video_frames1(reduced_video_inputs, 180)` on a *list* of tensors, which raises `AttributeError: 'list' object has no attribute 'shape'` for every question. | Qwen slow | Runtime | ✅ budget is applied inside selection |
| A6 | transformers ≥ 5: `get_image_features()` returns a `BaseModelOutputWithPooling`, so the notebooks' `[0]` takes the **patch-token hidden states** ([1, tokens, hidden]) instead of the embedding. The SVD then broadcasts over a 4-D array, and MaxVol fails. No dependency versions are pinned anywhere. | all CLIP calls | Runtime with a tiny CLIP on transformers 5.1 | ✅ `encoders._image_embedding` handles 4.x and 5.x; test `test_frame_encoder_returns_projected_embedding` |
| A7 | Every exception is caught and written as answer `-1`. Combined with A4–A6, a run "finishes" with every answer wrong and nothing flags it. | all LLaVA/Qwen | Code reading + A5 runtime | ✅ `eval.runner` records the traceback, never produces an answer from a failure, and aborts on consecutive or systematic errors |

## B. Evaluation correctness

| # | Issue | Verified by | Status |
|---|---|---|---|
| B1 | The EgoSchema parser takes the first capital A–E in **any word**: "Based on the video, the answer is C" → **B**; "Answer: D" → **A**; "Every step shows (A)" → **E**. | Runtime | ✅ `eval.parsing.parse_choice` accepts only explicit answer forms and returns `None` (counted separately) otherwise |
| B2 | The Video-MME-style parser (`videomme_dump`) maps unparsable output to **"C"**, and only knows A–D although LongVideoBench has up to 5 options ("E" → C). | Runtime | ✅ same parser; unused in MiniCPM, but a trap if reused |
| B3 | LLaVA argparse: `--answer-file` is required but not passed, and `--model-name` is passed but never declared, yet `args.model_name` is used. → `SystemExit: 2`. | Runtime | ✅ adapters take plain arguments |
| B4 | No scoring code. MiniCPM saves raw text only, and the LLaVA/Qwen frame-count files have fixed names (`frames`, `frames-informations`) that each run overwrites. | Code reading | ✅ `eval.runner.score` (overall + per-group, errors and unparsable outputs reported) |
| B5 | A missing video leaves `video_path` unbound inside the Dataset, so the DataLoader worker crashes outside the try block. MiniCPM's error path itself references undefined names (`traceback`, `re`, `video_ytid`, `selected_frames`). | `pyflakes` + code reading | ✅ loaders validate fields and files up front |
| B6 | Debug output: `print(np.allclose(M, C @ M[piv]))` is printed for every video, and with tol < 1 it is always `False`. | Code reading | ✅ removed |

## C. Method / design

| # | Issue | Verified by | Status |
|---|---|---|---|
| C1 | **Temporal information is lost.** LLaVA/Qwen receive irregularly spaced frames without timestamps (the LLaVA-Video `time_instruction` is commented out), so the model assumes even spacing. MiniCPM is the only pipeline that passes `temporal_ids`. | Code reading | ✅ every selection carries timestamps; `prompts.time_instruction`; MiniCPM adapter passes `temporal_ids` |
| C2 | **Qwen2-VL merges consecutive frame pairs** into one temporal patch (`temporal_patch_size = 2` in the transformers config and processor). After MaxInfo, a "pair" can be two frames minutes apart, fused into one token. | transformers source | ✅ `MaxInfoConfig(group_size=2)` selects whole pairs; Qwen adapter enforces it |
| C3 | Odd frame counts drop the **temporally last** selected frame. That is needed by no model here: the Qwen processor handles odd counts, and LLaVA and MiniCPM don't need even counts. | Runtime | ✅ Qwen adapter pads by repeating the last frame; others don't trim |
| C4 | The budget is enforced by evenly subsampling the time-sorted pivots, which throws away MaxVol's greedy order (most volume first). Keeping the first `budget` pivots in greedy order gives a strictly larger volume. | Measured: median log-vol −1.76 (greedy) vs −1.95 (official) at 128→64 | ✅ `budget_mode="greedy"` (default); `"uniform"` reproduces the paper |
| C5 | Chunk-based MaxInfo (paper §3.5, Table 2) is not in the release, and the paper doesn't say how the budget is enforced for it. | Repo contents | ✅ `chunked_select` (proportional budget, every chunk represented) |
| C6 | The SVD is uncentered: the top singular direction is the mean CLIP embedding (\|cos\| = 1.00 on all videos, σ₁/σ₂ ≈ 3.5), so 1 of the 8 rank dimensions describes what all frames share. Centering changes about 28% of the selection (Jaccard 0.72). The effect on accuracy is untested. | Measured | 📝 `center=False` default (paper behaviour), `center=True` available |
| C7 | MaxVol maximises volume, which favours extreme frames: average coverage (mean best cosine similarity of each candidate to a selected frame) is **lower** than uniform's (0.936 vs 0.956), and the largest temporal gap is about 3× larger (7.8% vs 2.3% of the video). Chunked MaxInfo restores coverage (0.975, gap 3.9%). | Measured | 📝 reported by `compare_official.py`; chunked variant available |
| C8 | CLIP is run **one frame at a time**, and on a different device than `device_map="auto"` chose (the notebook's `auto_select_device` returns `cuda:0`). | Code reading | ✅ batched encoder on one explicit device |
| C9 | `load_video` (copied from LLaVA-NeXT) computes `frame_time = i / step` instead of `i / fps`, and `np.linspace(..., dtype=int)` repeats frames on short videos (duplicate rows). | Code reading | ✅ `video.candidate_indices` (unique, bin-centred), timestamps = idx / fps |
| C10 | `maxvolpy` is imported as `maxvolpy.maxvolpy.maxvol`, a path that only exists in a cloned repo. The PyPI sdist needs a C compiler (the build failed here), and its fallback is slow pure Python. | Install attempt | ✅ numpy-only `rect_maxvol` with O(n·r) Sherman–Morrison updates; matches maxvolpy (test `test_matches_maxvolpy`) |
| C11 | MiniCPM: `CUDA_VISIBLE_DEVICES="4,5,6,7"` is hardcoded, and selection indexes temporal-id *groups* by *frame* index, which is only correct while `MAX_NUM_PACKING = 1`. | Code reading | ✅ adapter builds `temporal_ids` per selected frame |

## D. Paper-level (not fixable in code)

| # | Issue |
|---|---|
| D1 | The frame ranges reported for LLaVA (e.g. `128→(12,64)`) cannot come from the released code, which always yields exactly 64 (A2). The released notebooks do not reproduce the tables. |
| D2 | Table 4 compares MaxInfo numbers reproduced in the authors' pipeline against baselines **copied from other papers** (marked \*), contrary to §4's "compared … against itself". The baseline code is not released. |
| D3 | Notation: `r` means both the SVD rank and the number of selected frames; Eq. 6 says S is "r×n" (should be r×d); Fig. 3's caption ("effect of initial sampling") describes Fig. 4. |
| D4 | Appendix C: Vol(conv. hull) = κ·√det(SᵀS) does not hold in general, and the "upper bound on differential entropy" argument is informal. |
| D5 | §A claims tolerance and rank are listed in the result tables for reproducibility. They are not. |

## Checked and refuted

- **"MiniCPM `svd_reduce_dim(...).cpu()` crashes"**: no. `np.linalg.svd` on a torch tensor returns a tensor, so `.cpu()` works.
- **"maxvolpy's 10-iteration square-MaxVol start doesn't converge"**: it converges on all videos (max \|C\| ≤ 1.04 against a 1.05 threshold).
- **"this fork's own `maxinfo.rect_maxvol_greedy` is not real MaxVol"**: it matches the reference closely (same K, same volume, Jaccard 0.92–1.0 on synthetic data).

## Not verified here

- Downstream VLLM accuracy (needs a GPU with ≥ 16 GB). The model adapters in `maxinfo/models.py` follow each model's official usage but have **not** been run end-to-end.
- LongVideoBench field names: the HF repo is gated for this account, so `load_longvideobench` checks its expected fields at load time.
