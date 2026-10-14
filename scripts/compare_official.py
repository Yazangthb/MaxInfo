"""Compare frame selections on cached CLIP features: official notebook behaviour vs this package.

Input: a folder of .npz files with `feats` [n, d] (one per video, candidates in temporal order),
e.g. produced by scripts/extract_features.py. No VLLM involved: this checks *what gets selected*.

Metrics (per video, then median):
  n        number of frames passed to the VLLM
  same_as_uniform  fraction of selected indices identical to (official) uniform sampling with the same count
  logvol   log-volume of the selection in the rank-8 SVD basis of all candidates (higher = more diverse)
  cover    mean over all candidates of max cosine similarity to a selected frame (higher = better coverage)
  maxgap   largest temporal gap between consecutive selected frames, as a fraction of the video
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from maxinfo.selection import chunked_select, rect_maxvol, select_frames, svd_basis


def official_notebook(F, budget, tol):
    """LLaVA/Qwen notebook logic: rect_maxvol(U[:, :8], tol) -> sort -> drop last if odd -> linspace to budget."""
    piv = sorted(rect_maxvol(svd_basis(F, 8), tol))
    if len(piv) > 1 and len(piv) % 2:
        piv = piv[:-1]
    if len(piv) > budget:
        piv = [piv[i] for i in np.linspace(0, len(piv) - 1, budget).astype(int)]
    return np.array(piv)


def official_uniform(n, k):
    """Uniform sampling as the official code does it (linspace then truncate)."""
    return np.unique(np.linspace(0, n - 1, min(k, n)).astype(int))


def metrics(F, idx):
    Q = svd_basis(F, 8)
    B = Q[idx]
    logvol = 0.5 * np.linalg.slogdet(B.T @ B)[1] if len(idx) >= Q.shape[1] else float("nan")
    Fn = F / np.linalg.norm(F, axis=1, keepdims=True)
    cover = float((Fn @ Fn[idx].T).max(axis=1).mean())
    pts = np.r_[0, np.sort(idx), len(F) - 1]
    uni = set(official_uniform(len(F), len(idx)).tolist())
    return dict(n=len(idx), same_as_uniform=len(uni & set(idx.tolist())) / len(idx), logvol=logvol, cover=cover,
                maxgap=float(np.diff(pts).max() / len(F)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features_dir")
    ap.add_argument("--budget", type=int, default=64)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.features_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz files in {args.features_dir}")

    b = args.budget
    methods = {
        "uniform": lambda F: official_uniform(len(F), b),
        "official as released (tol=0.1)": lambda F: official_notebook(F, b, 0.1),
        "official intended (tol=0.3)": lambda F: official_notebook(F, b, 0.3),
        "new: greedy budget": lambda F: select_frames(F, budget=b, tol=0.3).indices,
        "new: adaptive (no budget, tol=0.45)": lambda F: select_frames(F, budget=None, tol=0.45).indices,
        "new: centered": lambda F: select_frames(F, budget=b, tol=0.3, center=True).indices,
        "new: chunked (8)": lambda F: chunked_select(F, n_chunks=8, budget=b, tol=0.3).indices,
        "new: pairs (Qwen2-VL)": lambda F: select_frames(F, budget=b, tol=0.3, group_size=2).indices,
    }
    table = {m: [] for m in methods}
    for f in files:
        F = np.load(f)["feats"].astype(np.float64)
        for m, fn in methods.items():
            table[m].append(metrics(F, np.asarray(fn(F), dtype=int)))

    print(f"{len(files)} videos, {len(np.load(files[0])['feats'])} candidates each, budget {b} (medians)\n")
    cols = ["n", "same_as_uniform", "logvol", "cover", "maxgap"]
    print(f"{'method':34}" + "".join(f"{c:>17}" for c in cols))
    for m, rows in table.items():
        print(f"{m:34}" + "".join(f"{np.nanmedian([r[c] for r in rows]):>17.3f}" for c in cols))


if __name__ == "__main__":
    main()
