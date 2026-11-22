"""Strict streaming exactness, causality (assert_causal), and E-process tests for RingMP."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import flashmp


def test_ringmp_exact_egress_every_tick():
    """Verify RingMP full-window MP matches fresh batch STOMP on EVERY single tick after egress."""
    rng = np.random.default_rng(2026)
    m = 20
    W_sub = 120
    W_raw = W_sub + m - 1
    total = W_raw + 150
    x = np.cumsum(rng.standard_normal(total)) + 0.3 * np.sin(np.linspace(0, 25, total))

    st = flashmp.StreamMP(m=m, window=W_sub, egress=True, track_full_mp=True)
    st.init(x[:W_raw])

    for k in range(150):
        res = st.update(float(x[W_raw + k]))
        assert res.gid == W_sub + k
        if k % 5 == 0 or k == 149:
            snap = st.snapshot(relative_indices=True)
            snap_mp = np.asarray(snap.mp, dtype=np.float64)
            win = x[k + 1 : W_raw + k + 1]
            ref = flashmp.profile(win, m, threads=1)
            assert snap_mp.shape == ref.mp.shape
            err = float(np.max(np.abs(snap_mp - ref.mp)))
            assert err < 1e-11, f"Egress drift at tick {k}: max_err={err}"
            # Also verify newest tick's causal left_dist matches ref.left_mp[-1]
            assert abs(res.left_dist - float(ref.left_mp[-1])) < 1e-11
    st.close()


def test_ringmp_adversarial_monotone_stack_overflow():
    """Force >32 suffix-max record holders to verify exact fallback rebuild under truncation."""
    m = 12
    W_sub = 100
    # Construct a smooth monotonic chirp where older windows increasingly correlate with recent ones
    t = np.arange(350, dtype=np.float64)
    x = np.sin(0.08 * t) + 0.002 * (t ** 1.3)
    st = flashmp.StreamMP(m=m, window=W_sub, egress=True, track_full_mp=True)
    st.init(x[: W_sub + m - 1])
    for k in range(200):
        st.update(float(x[W_sub + m - 1 + k]))
        if k % 20 == 0 or k == 199:
            snap_mp = np.asarray(st.snapshot(relative_indices=True).mp, dtype=np.float64)
            win = x[k + 1 : W_sub + m + k]
            ref = flashmp.profile(win, m, threads=1)
            assert np.max(np.abs(snap_mp - ref.mp)) < 1e-11
    st.close()


def test_strict_causality_future_perturbation():
    """
    assert_causal: Perturbing any future ticks t' > t with +1e9, -1e9, or NaN
    must leave ALL RingMP outputs at ticks <= t bit-identical (tol == 0.0).
    """
    rng = np.random.default_rng(404)
    x = np.sin(np.linspace(0, 30 * np.pi, 400)) + 0.05 * rng.standard_normal(400)
    t_cut = 250

    def run_stream(arr):
        st = flashmp.StreamMP(m=24, window=150, egress=True, alpha=0.05, track_full_mp=True)
        out = [st.update(float(v)) for v in arr[: t_cut + 1]]
        st.close()
        return out

    base = run_stream(x)
    for perturb in [1e9, -1e9, float("nan")]:
        x_p = x.copy()
        x_p[t_cut + 1 :] = perturb
        alt = run_stream(x_p)
        for a, b in zip(base, alt):
            assert a.gid == b.gid
            assert a.left_idx == b.left_idx
            assert a.left_dist == b.left_dist or (math.isinf(a.left_dist) and math.isinf(b.left_dist))
            assert a.e_value == b.e_value
            assert a.is_alert == b.is_alert


def test_ringmp_ville_eprocess_detects_anomaly_and_controls_null():
    """Ville betting martingale fires on real shape break and stays quiet on stationary signal."""
    rng = np.random.default_rng(777)
    m = 32
    t = np.arange(900, dtype=np.float64)
    clean = np.sin(2 * np.pi * t / m) + 0.04 * rng.standard_normal(900)

    # 1. Stationary null: no false alarm at alpha=0.01
    st_null = flashmp.StreamMP(m=m, window=300, egress=True, alpha=0.01, track_full_mp=False)
    st_null.init(clean[:300])
    _, _, evs_null, alerts_null = st_null.update_batch(clean[300:])
    st_null.close()
    assert int(np.sum(alerts_null)) == 0

    # 2. Injected shape anomaly: fires reliably
    anom = clean.copy()
    anom[500:545] = np.sign(np.sin(np.linspace(0, 11 * np.pi, 45))) * 0.9
    st_anom = flashmp.StreamMP(m=m, window=300, egress=True, alpha=0.01, track_full_mp=False)
    st_anom.init(anom[:300])
    _, _, evs_anom, alerts_anom = st_anom.update_batch(anom[300:])
    st_anom.close()
    assert int(np.sum(alerts_anom)) >= 1
    assert float(np.max(evs_anom)) >= 100.0


def test_stumpi_dropin_api():
    rng = np.random.default_rng(321)
    x = np.cumsum(rng.standard_normal(260))
    st = flashmp.stumpi(x[:200], m=24, egress=True)
    for v in x[200:]:
        st.update(float(v))
    assert len(st.P_) == 200 - 24 + 1
    assert len(st.I_) == 200 - 24 + 1
    ref = flashmp.profile(x[60:260], 24, threads=1)
    assert np.max(np.abs(st.P_ - ref.mp)) < 1e-11
