"""Exhaustive exactness, numerical stability, and drop-in parity tests for FlashMP."""

from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import flashmp


def _brute_self_join(x: np.ndarray, m: int, ez: int) -> np.ndarray:
    n = len(x) - m + 1
    out = np.full(n, np.inf, dtype=np.float64)
    wins = np.lib.stride_tricks.sliding_window_view(x, m)
    valid = np.all(np.isfinite(wins), axis=1)
    means = np.where(valid, np.mean(np.where(np.isfinite(wins), wins, 0.0), axis=1), 0.0)
    centered = wins - means[:, None]
    norms = np.sqrt(np.sum(centered * centered, axis=1))
    valid &= norms > 1e-12
    for i in range(n):
        if not valid[i]:
            continue
        best_r = -2.0
        for j in range(n):
            if abs(i - j) <= ez or not valid[j]:
                continue
            r = float(np.dot(centered[i], centered[j]) / (norms[i] * norms[j]))
            if r > best_r:
                best_r = r
        if best_r > -1.5:
            out[i] = math.sqrt(max(0.0, 2.0 * m * (1.0 - min(1.0, max(-1.0, best_r)))))
    return out


def test_cold_import_fast_and_zero_numpy():
    import os

    code = (
        "import sys, time; t0=time.perf_counter(); import flashmp; "
        "dt=(time.perf_counter()-t0)*1e3; "
        "assert 'numpy' not in sys.modules, 'numpy eagerly imported!'; "
        "print(f'{dt:.3f}')"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    out = subprocess.check_output(
        [sys.executable, "-c", code],
        env=env,
        text=True,
    ).strip()
    assert float(out) < 60.0, f"Cold import too slow: {out} ms"


@pytest.mark.parametrize("N,m", [(45, 7), (128, 16), (313, 25), (600, 32)])
def test_self_join_matches_brute_force(N: int, m: int):
    rng = np.random.default_rng(100 + N + m)
    x = np.cumsum(rng.standard_normal(N)) + 0.5 * np.sin(np.linspace(0, 20, N))
    ez = (m + 3) // 4
    ref = _brute_self_join(x, m, ez)

    res = flashmp.profile(x, m, threads=2)
    assert np.max(np.abs(res.mp - ref)) < 1e-11

    # Verify STUMPY drop-in shape and column values
    st = flashmp.stump(x, m, threads=2)
    assert st.shape == (N - m + 1, 4)
    assert np.max(np.abs(st[:, 0].astype(np.float64) - ref)) < 1e-11
    # Verify exclusion zone strictly respected
    valid_mask = st[:, 1] >= 0
    assert np.all(np.abs(np.arange(len(st))[valid_mask] - st[valid_mask, 1]) > ez)

    # Verify Left-MP and Right-MP invariants
    for i in range(res.n):
        if res.left_idx[i] >= 0:
            assert res.left_idx[i] < i - ez
        if res.right_idx[i] >= 0:
            assert res.right_idx[i] > i + ez
        assert abs(res.mp[i] - min(res.left_mp[i], res.right_mp[i])) < 1e-12


def test_high_dc_and_tiny_scale_stability():
    rng = np.random.default_rng(77)
    sig = np.sin(np.linspace(0, 30 * np.pi, 1024)) + 0.08 * rng.standard_normal(1024)
    ref = flashmp.profile(sig, 32, threads=2).mp

    # Huge DC offset (1e10): immune to subtractive cancellation
    dc = flashmp.profile(sig + 1e10, 32, threads=2).mp
    assert np.max(np.abs(dc - ref)) < 1e-5

    # Tiny scale (1e-9): scale-free flatness guard preserves signal
    tiny = flashmp.profile(sig * 1e-9, 32, threads=2).mp
    assert np.max(np.abs(tiny - ref)) < 1e-11

    # Constant high-DC flat segment cleanly masked to +inf / idx=-1
    flat_sig = sig.copy() + 1e9
    flat_sig[200:260] = 1e9
    res_flat = flashmp.profile(flat_sig, 32, threads=2)
    assert res_flat.idx[210] == -1 and math.isinf(res_flat.mp[210])
    assert res_flat.idx[0] >= 0 and math.isfinite(res_flat.mp[0])


def test_nan_and_inf_isolation():
    rng = np.random.default_rng(88)
    x = np.sin(np.linspace(0, 20 * np.pi, 400)) + 0.1 * rng.standard_normal(400)
    x[100] = np.nan
    x[250] = np.inf
    res = flashmp.profile(x, 20, threads=2)
    # Any window covering index 100 or 250 must be invalid
    for bad_pos in [100, 250]:
        for i in range(max(0, bad_pos - 19), min(res.n, bad_pos + 1)):
            assert res.idx[i] == -1
            assert math.isinf(res.mp[i])
    # Windows far away must be finite and valid
    assert res.idx[0] >= 0 and math.isfinite(res.mp[0])
    assert res.idx[350] >= 0 and math.isfinite(res.mp[350])


def test_ab_join_and_leak_scan_affine():
    rng = np.random.default_rng(123)
    ref = np.cumsum(rng.standard_normal(2000))
    qry = np.cumsum(rng.standard_normal(500))
    # Inject verbatim copy, positive affine copy, and sign-flipped affine copy
    qry[50:114] = ref[400:464]
    qry[200:264] = 4.25 * ref[900:964] - 1234.5
    qry[350:414] = -2.5 * ref[1500:1564] + 88.0

    mp_a, idx_a, mp_b, idx_b = flashmp.abjoin(qry, ref, 64, threads=2)
    assert idx_a[50] == 400 and mp_a[50] < 1e-10
    assert idx_a[200] == 900 and mp_a[200] < 1e-10

    rep = flashmp.scan_leakage(ref, qry, m=64, corr_thresh=0.9999, check_neg=True, threads=2)
    assert rep.contaminated_windows >= 3
    found_pairs = {(e.qry_idx, e.ref_idx, e.match_type) for e in rep.examples}
    assert any(q == 50 and r == 400 and mt == "verbatim" for q, r, mt in found_pairs)
    assert any(q == 200 and r == 900 and mt == "affine" for q, r, mt in found_pairs)
    assert any(q == 350 and r == 1500 and mt == "neg_affine" for q, r, mt in found_pairs)


def test_multivariate_mstump_monotonicity_and_motif():
    rng = np.random.default_rng(999)
    D, N, m = 6, 600, 32
    M = rng.standard_normal((D, N))
    # Plant a 3-channel motif at positions 80 and 380
    motif = np.sin(np.linspace(0, 4 * np.pi, m))
    for c in range(3):
        M[c, 80 : 80 + m] = motif * (c + 1)
        M[c, 380 : 380 + m] = motif * (c + 1)

    P, I = flashmp.mstump(M, m, threads=2)
    assert P.shape == (D, N - m + 1)
    assert I.shape == (D, N - m + 1)
    # k-dim subspace mean distance must be monotonically non-decreasing in k
    for k in range(1, D):
        assert np.all(P[k] + 1e-12 >= P[k - 1])
    # The 3-channel motif must have ~0 distance at k=0, 1, 2
    for k in range(3):
        assert P[k, 80] < 1e-10
        assert I[k, 80] == 380


def test_floss_regime_and_auto_window():
    rng = np.random.default_rng(55)
    t = np.arange(1600, dtype=np.float64)
    period = 40
    r1 = np.sin(2 * np.pi * t[:800] / period) + 0.04 * rng.standard_normal(800)
    r2 = np.sign(np.sin(2 * np.pi * t[800:] / 25.0)) + 0.04 * rng.standard_normal(800)
    x = np.concatenate([r1, r2])

    aw = flashmp.auto_window(r1, min_m=8, max_m=120)
    assert abs(aw - period) <= 3

    res = flashmp.profile(x, 40, threads=2)
    regimes = res.regimes(max_regimes=2)
    assert len(regimes) == 1
    assert abs(regimes[0] - 800) <= 60
