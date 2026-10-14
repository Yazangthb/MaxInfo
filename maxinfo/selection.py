"""Frame selection by rectangular maximum volume (MaxVol) on frame embeddings.

Pure numpy, no compiled dependencies. The algorithm follows Mikhalev & Oseledets,
"Rectangular maximum-volume submatrices and their applications" (2018), which is
what the MaxInfo paper uses through `maxvolpy.rect_maxvol`:

1. square MaxVol: pick r rows of A (n x r) whose r x r submatrix has (locally) maximal volume;
2. rectangular growth: repeatedly add the row with the largest coefficient norm
   ||c_i|| where C = A @ pinv(A[piv]), until every remaining row has ||c_i|| <= tol.

Adding row i multiplies the volume sqrt(det(B^T B)) by sqrt(1 + ||c_i||^2), so the greedy
order is the order of decreasing volume gain. That order is kept (`SelectionResult.order`)
and used for budget truncation instead of the official code's even subsampling over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SelectionResult:
    """Indices into the candidate frames.

    indices: selected candidates in temporal (ascending) order — what you feed the VLLM.
    order: the same candidates in greedy MaxVol order (most volume-increasing first).
    num_before_budget: how many rows MaxVol picked before the budget was applied.
    """

    indices: np.ndarray
    order: np.ndarray
    num_before_budget: int
    params: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.indices)


def svd_basis(features: np.ndarray, rank: int, *, center: bool = False) -> np.ndarray:
    """Return U[:, :rank] of the thin SVD of the (optionally mean-centered) feature matrix.

    Without centering, the leading singular vector of CLIP features is almost exactly
    the mean embedding direction, so one of the `rank` dimensions describes what all frames
    share. `center=True` removes it. The official code does not center; that is the default here.
    """
    F = np.asarray(features, dtype=np.float64)
    if F.ndim != 2:
        raise ValueError(f"features must be 2-D [n_frames, dim], got shape {F.shape}")
    if not np.all(np.isfinite(F)):
        raise ValueError("features contain NaN or inf")
    if center:
        F = F - F.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(F, full_matrices=False)
    # Drop numerically-null directions (e.g. after centering, or duplicated frames).
    keep = int(np.sum(S > S[0] * 1e-10)) if S.size and S[0] > 0 else 0
    return U[:, : max(1, min(int(rank), keep))]


def _pivoted_gs(A: np.ndarray, k: int) -> list[int]:
    """k rows chosen greedily by largest residual norm after projecting out the rows already chosen
    (= QR with column pivoting on A^T). Rows with zero residual are taken last, in index order."""
    R = np.array(A, dtype=np.float64, copy=True)
    piv: list[int] = []
    for _ in range(min(k, R.shape[0])):
        norms = np.einsum("ij,ij->i", R, R)
        norms[piv] = -1.0
        i = int(np.argmax(norms))
        piv.append(i)
        nrm = np.sqrt(norms[i])
        if nrm > 1e-12:
            v = R[i] / nrm
            R -= np.outer(R @ v, v)
    return piv


def _square_maxvol(A: np.ndarray, tol: float = 1.05, max_iters: int = 100) -> list[int]:
    """Square MaxVol on A (n x r, full column rank): r rows with locally maximal |det|.

    Initialised by greedy pivoted Gram-Schmidt (= QR with column pivoting on A^T), then
    improved by row swaps while some |C_ij| > tol, with C = A @ inv(A[piv]).
    """
    piv = _pivoted_gs(A, A.shape[1])
    for _ in range(max_iters):
        C = np.linalg.solve(A[piv].T, A.T).T  # A @ inv(A[piv])
        i, j = np.unravel_index(int(np.argmax(np.abs(C))), C.shape)
        if abs(C[i, j]) <= tol:
            break
        piv[j] = int(i)
    return piv


def rect_maxvol(
    A: np.ndarray,
    tol: float = 1.0,
    *,
    max_k: int | None = None,
    min_k: int | None = None,
) -> list[int]:
    """Rectangular MaxVol row selection. Returns row indices in greedy (selection) order.

    Same stopping rule as `maxvolpy.maxvol.rect_maxvol`: stop when every unselected row has
    coefficient norm <= tol (and at least `min_k` rows are chosen), or when `max_k` rows are chosen.

    Coefficient norms are updated in O(n r) per step via Sherman-Morrison on M = (B^T B)^-1:
    ||c_j||^2 = a_j M a_j^T and, after adding row a, M' = M - M a^T a M / (1 + a M a^T).
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}")
    n, r = A.shape
    if n == 0:
        return []
    if tol <= 0:
        raise ValueError("tol must be > 0")
    if n <= r:
        # Every row is kept; order them by pivoted Gram-Schmidt so a truncation keeps the most independent.
        return _pivoted_gs(A, n)

    max_k = n if max_k is None else int(max(1, min(max_k, n)))
    min_k = 0 if min_k is None else int(min(max(min_k, 0), max_k))

    if max_k < r:
        # Fewer rows than columns: volume is defined on the leading max_k directions.
        return _square_maxvol(A[:, :max_k])

    piv = _square_maxvol(A)
    B = A[piv]
    M = np.linalg.inv(B.T @ B)
    AM = A @ M
    s = np.einsum("ij,ij->i", AM, A)  # ||c_j||^2
    selected = np.zeros(n, dtype=bool)
    selected[piv] = True
    tol2 = float(tol) ** 2

    while len(piv) < max_k:
        s_masked = np.where(selected, -np.inf, s)
        i = int(np.argmax(s_masked))
        if s_masked[i] <= tol2 and len(piv) >= min_k:
            break
        a = A[i]
        Ma = M @ a
        denom = 1.0 + a @ Ma
        proj = A @ Ma  # a_j M a^T for all j
        s = s - proj * proj / denom
        M = M - np.outer(Ma, Ma) / denom
        piv.append(i)
        selected[i] = True
    return piv


def _uniform_pick(seq, k: int) -> list:
    if len(seq) <= k:
        return list(seq)
    return [seq[i] for i in np.linspace(0, len(seq) - 1, k).round().astype(int)]


def select_frames(
    features: np.ndarray,
    *,
    budget: int | None = None,
    rank: int = 8,
    tol: float = 0.3,
    center: bool = False,
    budget_mode: str = "greedy",
    group_size: int = 1,
    min_frames: int = 1,
) -> SelectionResult:
    """MaxInfo selection over candidate frames.

    Args:
        features: [n, d] frame embeddings (one row per candidate, temporal order).
        budget: maximum number of frames to return (the VLLM's frame limit). None = no cap.
        rank: SVD rank r (paper's best: 8).
        tol: MaxVol tolerance (paper's best: 0.3). Smaller -> more frames.
        center: mean-center features before SVD (see `svd_basis`).
        budget_mode: "greedy" keeps the first `budget` rows in MaxVol order (largest volume);
            "uniform" reproduces the paper: take all pivots, then subsample evenly in time.
        group_size: select units of `group_size` consecutive candidates (e.g. 2 for Qwen2-VL,
            whose patch embedding merges consecutive frame pairs). Each unit is represented by
            the mean of its members' embeddings; budget/min_frames count frames, not units.
        min_frames: always return at least this many frames (if available).
    """
    F = np.asarray(features)
    if F.ndim != 2:
        raise ValueError(f"features must be 2-D [n_frames, dim], got shape {F.shape}")
    if budget_mode not in ("greedy", "uniform"):
        raise ValueError("budget_mode must be 'greedy' or 'uniform'")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    n = F.shape[0]
    params = dict(budget=budget, rank=rank, tol=tol, center=center, budget_mode=budget_mode,
                  group_size=group_size, min_frames=min_frames)
    if n == 0:
        empty = np.zeros(0, dtype=int)
        return SelectionResult(empty, empty, 0, params)

    # Units = groups of consecutive candidates (the last group may be shorter).
    starts = np.arange(0, n, group_size)
    units = [np.arange(s, min(s + group_size, n)) for s in starts]
    U_feats = np.stack([F[u].mean(axis=0) for u in units]) if group_size > 1 else F
    m = len(units)

    unit_budget = None if budget is None else max(1, int(budget) // group_size)
    unit_min = max(1, -(-int(min_frames) // group_size))  # ceil
    if unit_budget is not None:
        unit_min = min(unit_min, unit_budget)

    eff_rank = int(min(rank, m, unit_budget if unit_budget is not None else rank))
    Q = svd_basis(U_feats, eff_rank, center=center)

    # Greedy order is prefix-stable, so the uncapped run also gives the budgeted selection.
    all_piv = rect_maxvol(Q, tol, min_k=unit_min)
    if unit_budget is None or len(all_piv) <= unit_budget:
        order_u = all_piv
    elif budget_mode == "greedy":
        order_u = all_piv[:unit_budget]
    else:
        kept = set(_uniform_pick(sorted(all_piv), unit_budget))
        order_u = [p for p in all_piv if p in kept]

    order = np.concatenate([units[u] for u in order_u]).astype(int)
    num_before = int(sum(len(units[u]) for u in all_piv))
    return SelectionResult(np.sort(order), order, num_before, params)


def chunked_select(
    features: np.ndarray,
    *,
    n_chunks: int = 32,
    budget: int | None = None,
    rank: int = 8,
    tol: float = 0.3,
    center: bool = False,
    group_size: int = 1,
) -> SelectionResult:
    """Chunk-based MaxInfo (paper §3.5): split candidates into `n_chunks` contiguous chunks and
    run MaxVol inside each, so every part of the video is represented.

    The paper does not say how the global budget is enforced for this variant. Here the budget is
    split evenly across the (equal-sized) chunks before selection, so each chunk runs MaxVol with
    its own cap and rank <= that cap. Splitting afterwards does not work: a small chunk with
    rank >= its size has an orthogonal SVD basis, every subset has the same volume, and the
    "selection" degenerates to temporal order. A chunk may return fewer frames than its share
    when MaxVol's tolerance is met early (it is redundant).
    """
    F = np.asarray(features)
    n = F.shape[0]
    params = dict(n_chunks=n_chunks, budget=budget, rank=rank, tol=tol, center=center, group_size=group_size)
    if n == 0:
        empty = np.zeros(0, dtype=int)
        return SelectionResult(empty, empty, 0, params)
    # Chunk boundaries aligned to group_size so groups are never split across chunks.
    n_units = -(-n // group_size)
    k = int(max(1, min(n_chunks, n_units)))
    bounds = np.linspace(0, n_units, k + 1).round().astype(int) * group_size
    chunks = [(a, min(b, n)) for a, b in zip(bounds[:-1], bounds[1:]) if min(b, n) > a]

    if budget is None:
        shares = [None] * len(chunks)
    else:
        budget_units = max(1, int(budget) // group_size)
        sizes = np.array([-(-(b - a) // group_size) for a, b in chunks])
        alloc = np.zeros(len(chunks), dtype=int)
        for _ in range(budget_units):  # round-robin: even split, never more than a chunk holds
            room = sizes - alloc
            if room.max() <= 0:
                break
            alloc[int(np.argmax((room > 0) * (sizes.max() + 1) - alloc))] += 1
        shares = [int(u) * group_size for u in alloc]

    per_chunk: list[np.ndarray] = []
    total = 0
    for (a, b), share in zip(chunks, shares):
        if share == 0:
            continue
        res = select_frames(F[a:b], budget=share, rank=rank, tol=tol, center=center, group_size=group_size)
        per_chunk.append(res.order + a)
        total += res.num_before_budget

    order = np.concatenate(per_chunk).astype(int) if per_chunk else np.zeros(0, dtype=int)
    return SelectionResult(np.sort(order), order, total, params)


def uniform_select(n: int, budget: int) -> np.ndarray:
    """Baseline: `budget` evenly spaced, unique indices out of n candidates."""
    if n <= 0 or budget <= 0:
        return np.zeros(0, dtype=int)
    return np.unique(np.linspace(0, n - 1, min(budget, n)).round().astype(int))
