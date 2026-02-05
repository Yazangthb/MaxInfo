from __future__ import annotations


def svd_reduce_dim(features, r: int = 8):
    """Return U[:, :r] from SVD(features)."""

    import numpy as np

    feats = np.asarray(features)
    if feats.ndim != 2:
        raise ValueError(f"features must be 2D [T,D]; got shape {feats.shape}")

    T = feats.shape[0]
    if T == 0:
        return feats.reshape((0, 0))

    r = int(max(1, min(int(r), T)))
    U, S, Vt = np.linalg.svd(feats, full_matrices=False)
    return U[:, :r]


def _row_norms_sq(X):
    import numpy as np

    X = np.asarray(X)
    return np.sum(X * X, axis=1)


def rect_maxvol_greedy(Qs, tol: float = 0.3, maxK: int | None = None, seed: int | None = None, progress: bool = True):
    """Greedy rectangular MaxVol-like row selection.

    Matches the notebook's Experiment pipeline 2 implementation:
      - initialize with r rows
      - grow pivots until max row coefficient norm <= tol (or maxK)

    Args:
        Qs: np.ndarray [n, r]
        tol: stopping tolerance
        maxK: cap on number of rows selected
        seed: rng seed for tie-breaks
        progress: whether to show progress updates

    Returns:
        list[int]: selected row indices (sorted, unique)
    """

    import numpy as np
    from tqdm.auto import tqdm

    Qs = np.asarray(Qs, dtype=np.float32)
    n, r = Qs.shape if Qs.ndim == 2 else (0, 0)
    if n == 0:
        return []

    if maxK is None:
        maxK = n
    maxK = int(max(1, min(int(maxK), n)))

    rng = np.random.default_rng(seed)

    # 1) pick r initial rows (simple stable init)
    piv = [int(np.argmax(_row_norms_sq(Qs)))]
    while len(piv) < min(r, n):
        chosen = Qs[piv]
        centroid = chosen.mean(axis=0, keepdims=True)
        d2 = _row_norms_sq(Qs - centroid)
        cand = int(np.argmax(d2))
        if cand in piv:
            cand = int(rng.integers(0, n))
        piv.append(cand)

    piv = list(dict.fromkeys(piv))
    if len(piv) >= n:
        return sorted(piv)

    def compute_C(Qs_local, piv_local):
        B = Qs_local[piv_local, :]  # [k, r]
        Binv = np.linalg.pinv(B)  # [r, k]
        C = Qs_local @ Binv  # [n, k]
        return C

    # 2) grow until tolerance satisfied
    if progress:
        print(f"📊 Running MaxVol greedy selection (tol={tol}, n={n}, r={r})...")
    
    # Use tqdm for the main loop
    while True:
        C = compute_C(Qs, piv)
        cnorm = np.linalg.norm(C, axis=1)
        cnorm_sel = cnorm.copy()
        cnorm_sel[piv] = -np.inf
        i_star = int(np.argmax(cnorm_sel))
        best = float(cnorm_sel[i_star])
        if best <= float(tol):
            break
        piv.append(i_star)
        if len(piv) >= maxK:
            break

    if progress:
        print(f"✓ Selected {len(piv)} pivots from {n} candidates")

    return sorted(set(map(int, piv)))


def _cosine_sim(a, b, eps: float = 1e-12) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def uniform_select(frames, ts_group, budget: int = 32):
    """Uniformly sample `budget` items from candidate frames."""

    import numpy as np

    T = len(frames)
    k = min(int(budget), int(T))
    if k <= 0:
        return [], [], []

    idx = np.linspace(0, T - 1, k).astype(int).tolist()
    return [frames[i] for i in idx], [ts_group[i] for i in idx], idx


def maxinfo_select_rect_maxvol(
    *,
    vision_model,
    vision_processor,
    frames,
    ts_group,
    budget: int = 32,
    svd_rank: int = 8,
    tol: float = 0.3,
    batch_size: int = 16,
    remove_similar: bool = False,
    similarity_thr: float = 0.9,
    progress: bool = True,
):
    """Paper-like MaxInfo selection used in Experiment pipeline 2.

    Steps:
      - CLIP image embedding extraction
      - SVD -> Qs = U[:, :svd_rank]
      - RectMaxVol greedy growth until tolerance `tol`
      - budget cap via even downsample if necessary
      - optional remove_similar in CLIP space
    
    Args:
        progress: Whether to show progress updates.
    """

    import numpy as np

    from .clip import extract_clip_features

    if len(frames) == 0:
        return [], [], []

    if progress:
        print(f"🎯 MaxInfo selection: {len(frames)} frames → budget={budget}, rank={svd_rank}, tol={tol}")

    feats = extract_clip_features(
        vision_model=vision_model,
        vision_processor=vision_processor,
        frames=frames,
        batch_size=batch_size,
        normalize=False,
        progress=progress,
    ).numpy()

    if progress:
        print(f"📐 Running SVD (rank={svd_rank})...")
    Qs = svd_reduce_dim(feats, r=int(svd_rank))
    if progress:
        print(f"✓ SVD complete: Qs shape = {Qs.shape}")

    piv = rect_maxvol_greedy(Qs, tol=float(tol), maxK=len(frames), progress=progress)

    # enforce budget by evenly subsampling the pivot list
    if len(piv) > int(budget):
        if progress:
            print(f"📉 Subsampling from {len(piv)} to {budget} pivots...")
        keep = np.linspace(0, len(piv) - 1, int(budget)).astype(int).tolist()
        piv = [piv[i] for i in keep]

    if remove_similar and len(piv) > 1:
        if progress:
            print(f"🔗 Removing similar frames (thr={similarity_thr})...")
        kept = [piv[0]]
        for idx in piv[1:]:
            sim = _cosine_sim(feats[kept[-1]], feats[idx])
            if sim < float(similarity_thr):
                kept.append(idx)
        piv = kept

    if progress:
        print(f"✓ MaxInfo selection complete: {len(piv)} frames selected")

    sel_frames = [frames[i] for i in piv]
    sel_ts = [ts_group[i] for i in piv]
    return sel_frames, sel_ts, piv

