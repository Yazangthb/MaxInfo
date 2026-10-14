import os
import sys

import numpy as np
import pytest

from maxinfo.selection import chunked_select, rect_maxvol, select_frames, svd_basis, uniform_select


def clustered(n=128, d=64, k=10, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) + 3.0  # shared offset ~ CLIP's common component
    labels = np.sort(rng.integers(0, k, size=n))
    return centers[labels] + noise * rng.normal(size=(n, d))


def coeff_norms(A, piv):
    return np.linalg.norm(A @ np.linalg.pinv(A[piv]), axis=1)


def logvol(A, piv):
    B = A[piv]
    return 0.5 * np.linalg.slogdet(B.T @ B)[1]


@pytest.mark.parametrize("tol", [0.15, 0.3, 0.6])
def test_stopping_rule(tol):
    A = svd_basis(clustered(), 8)
    piv = rect_maxvol(A, tol)
    rest = np.setdiff1d(np.arange(len(A)), piv)
    assert len(set(piv)) == len(piv)
    assert coeff_norms(A, piv)[rest].max() <= tol + 1e-9


def test_prefix_stability_and_cap():
    A = svd_basis(clustered(seed=1), 8)
    full = rect_maxvol(A, 0.3)
    capped = rect_maxvol(A, 0.3, max_k=20)
    assert capped == full[:20]


def test_small_tol_selects_everything():
    """The official LLaVA/Qwen notebooks hardcode tol=0.1: with r=8 that keeps every candidate."""
    A = svd_basis(clustered(n=128), 8)
    assert len(rect_maxvol(A, 0.1)) == 128


def test_greedy_budget_has_more_volume_than_uniform_subsampling():
    F = clustered(n=256, seed=2)
    g = select_frames(F, budget=32, tol=0.15, budget_mode="greedy")
    u = select_frames(F, budget=32, tol=0.15, budget_mode="uniform")
    Q = svd_basis(F, 8)
    assert len(g) == len(u) == 32
    assert logvol(Q, g.indices) > logvol(Q, u.indices)


def test_edge_cases():
    assert len(select_frames(np.zeros((0, 16)))) == 0
    F = clustered(n=5)
    assert list(select_frames(F, rank=8).indices) == list(range(5))  # n <= rank: keep all
    r = select_frames(clustered(n=100), budget=3, rank=8)  # budget < rank
    assert len(r) == 3
    dup = np.repeat(clustered(n=20), 3, axis=0)  # duplicated frames must not crash
    r = select_frames(dup, budget=16)
    assert len(r) <= 16 and len(set(r.indices)) == len(r)
    with pytest.raises(ValueError):
        select_frames(np.full((10, 4), np.nan))


def test_min_frames():
    F = np.tile(clustered(n=1), (50, 1)) + 1e-3 * np.random.default_rng(0).normal(size=(50, 64))
    assert len(select_frames(F, min_frames=10, tol=5.0)) >= 10


def test_group_size_selects_whole_pairs():
    r = select_frames(clustered(n=129), budget=40, group_size=2, tol=0.2)
    idx = r.indices
    assert len(idx) <= 40 and len(idx) % 2 == 0
    starts = idx[idx % 2 == 0]
    assert set(idx) == set(starts) | set(starts + 1)


def test_chunked_budget_and_coverage():
    F = clustered(n=256, seed=3)
    r = chunked_select(F, n_chunks=8, budget=40)
    assert len(r) <= 40
    chunk_of = np.minimum(r.indices // 32, 7)
    assert set(chunk_of) == set(range(8))  # every chunk represented
    assert np.bincount(chunk_of).max() <= 5  # even split of the budget


def test_chunked_small_chunks_still_select_distinct_frames():
    """Regression: chunks with <= rank units used to keep their first frames in temporal order."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 64)) + 3.0
    F = np.repeat(base, 64, axis=0) + 1e-3 * rng.normal(size=(64, 64))
    distinct = [6, 7, 14, 15, 22, 23, 30, 31]  # last frames of each 8-frame chunk are different
    F[distinct] += 2.0 * rng.normal(size=(len(distinct), 64))
    r = chunked_select(F, n_chunks=4, budget=8, rank=8)  # 16 frames per chunk, 2 per chunk
    assert set(r.indices) <= set(distinct) | set(range(32, 64))
    first_half = [i for i in r.indices if i < 32]
    assert len(first_half) == 4 and set(first_half) <= set(distinct)


def test_n_le_rank_order_is_by_volume():
    A = np.eye(4)[[0, 0, 1, 2]] * np.array([[1.0], [1.0], [0.1], [5.0]])  # rows 0,1 duplicates; row 3 largest
    order = rect_maxvol(A, 0.3)
    assert sorted(order) == [0, 1, 2, 3] and order[0] == 3 and set(order[:3]) == {3, 0, 2}


def test_center_removes_mean_direction():
    F = clustered()
    _, _, Vt = np.linalg.svd(F, full_matrices=False)
    mu = F.mean(0)
    assert abs(Vt[0] @ mu) / np.linalg.norm(mu) > 0.95  # uncentered: top direction ~ mean
    Qc = svd_basis(F, 8, center=True)
    assert np.allclose(Qc.sum(0), 0, atol=1e-8)  # centered basis is orthogonal to the ones vector


def test_uniform_select_unique():
    assert list(uniform_select(5, 8)) == [0, 1, 2, 3, 4]
    assert len(uniform_select(128, 64)) == 64


MAXVOLPY = os.environ.get("MAXVOLPY_SRC")


@pytest.mark.skipif(not MAXVOLPY, reason="set MAXVOLPY_SRC to a maxvolpy source dir to compare")
@pytest.mark.parametrize("seed,tol", [(0, 0.3), (1, 0.3), (2, 0.15), (3, 0.45)])
def test_matches_maxvolpy(seed, tol):
    import contextlib
    import io

    sys.path.insert(0, MAXVOLPY)
    with contextlib.redirect_stdout(io.StringIO()):
        from maxvolpy.maxvol import rect_maxvol as ref

    A = svd_basis(clustered(n=192, seed=seed), 8)
    ours = rect_maxvol(A, tol)
    theirs, _ = ref(A, tol=tol)
    assert abs(len(ours) - len(theirs)) <= max(2, len(theirs) // 20)
    assert abs(logvol(A, ours) - logvol(A, list(theirs))) < 0.1
