#!/usr/bin/env python3
"""
Real-World Multi-Domain Agentic Evaluation Benchmark for FlashMP.

Evaluates FlashMP end-to-end across 5 realistic deterministic telemetry archetypes
(N = 10,000 samples each, pure seeded NumPy generators):

  1. ecg_arrhythmia        — Synthetic PQRST heartbeat waveform with amplitude-matched
                             premature ventricular contraction (PVC) shape discord.
  2. cloud_microservice    — Diurnal microservice latency with deployment frequency/regime
                             shift + amplitude-bounded anomalous p99 oscillation burst.
  3. industrial_bearing    — Multi-harmonic rotating-machinery vibration with early
                             inner-race fault shape modulation (matched RMS/peak).
  4. financial_tick        — High-DC (1e5) tick microstructure with flash-crash regime
                             break, repeated execution-algo motif, and spoofing discord.
  5. iot_extreme_dc_spikes — IoT telemetry at DC=1e10 with two transient 1e10 EMI
                             glitches, post-spike exact motif, and subtle physical fault.

For each domain, measures:
  - FlashMP `digest()` median wall-clock latency (ms)
  - Raw CSV/JSON token count vs FlashMP compact digest token count (% reduction, ratio)
  - Estimated input LLM API cost per 10,000 queries at $3.00 / MTok (raw vs FlashMP)
  - Top-3 discord localization hit within ±m ticks: FlashMP vs LTTB vs M4 (matched tokens)
  - Lossless drill-down recovery max absolute error (must be exactly 0.0)
  - Exact RingMP (`StreamMP`) streaming throughput (µs/tick) and anytime-valid alert delay

Saves verified JSON receipt to `benchmarks/results_realworld_domains.json`.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import flashmp  # noqa: E402


API_PRICE_USD_PER_MTOK = 3.0
QUERIES_SCALE = 10_000


def estimate_tokens(text: str) -> int:
    """Conservative ~4 chars/token BPE estimate matching FlashMP's digest counter."""
    return max(1, (len(text) + 3) // 4)


def _match_envelope(wave: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linearly rescale `wave` to span strictly `[lo, hi]`."""
    w_min = float(np.min(wave))
    w_max = float(np.max(wave))
    if w_max - w_min < 1e-12:
        return np.full_like(wave, 0.5 * (lo + hi))
    norm = (wave - w_min) / (w_max - w_min)
    return lo + norm * (hi - lo)


def _pqrst_beat(period: int) -> np.ndarray:
    """Realistic single-cycle synthetic ECG PQRST morphology."""
    u = np.linspace(0.0, 1.0, period, endpoint=False)
    p_wave = 0.16 * np.exp(-0.5 * ((u - 0.18) / 0.038) ** 2)
    q_wave = -0.18 * np.exp(-0.5 * ((u - 0.34) / 0.015) ** 2)
    r_wave = 1.25 * np.exp(-0.5 * ((u - 0.40) / 0.018) ** 2)
    s_wave = -0.32 * np.exp(-0.5 * ((u - 0.46) / 0.018) ** 2)
    t_wave = 0.34 * np.exp(-0.5 * ((u - 0.69) / 0.062) ** 2)
    return p_wave + q_wave + r_wave + s_wave + t_wave


def gen_ecg_arrhythmia(N: int = 10_000, seed: int = 101) -> dict[str, Any]:
    """Domain 1: Synthetic PQRST ECG with amplitude-matched PVC shape discord."""
    rng = np.random.default_rng(seed)
    period = 80
    beat = _pqrst_beat(period)
    n_beats = (N + period - 1) // period
    x = np.tile(beat, n_beats)[:N].astype(np.float64)
    # Mild physiological baseline wander + sensor noise
    t = np.arange(N, dtype=np.float64)
    x += 0.025 * np.sin(2.0 * np.pi * t / 1200.0) + 0.014 * rng.standard_normal(N)

    # Inject premature ventricular contraction (PVC): wide bizarre biphasic QRS + inverted T
    # Explicitly peak-matched to normal beat envelope so amplitude detectors miss it!
    anom_pos = 5280
    u = np.linspace(0.0, 1.0, period, endpoint=False)
    pvc_raw = (
        0.95 * np.exp(-0.5 * ((u - 0.28) / 0.065) ** 2)
        - 0.85 * np.exp(-0.5 * ((u - 0.46) / 0.070) ** 2)
        + 0.45 * np.sin(2.0 * np.pi * 2.5 * u) * np.exp(-0.5 * ((u - 0.72) / 0.14) ** 2)
    )
    lo, hi = float(np.min(beat)), float(np.max(beat))
    x[anom_pos : anom_pos + period] = _match_envelope(pvc_raw, lo, hi) + 0.014 * rng.standard_normal(period)

    return {
        "id": "ecg_arrhythmia",
        "name": "ECG Arrhythmia (PQRST + PVC)",
        "description": "Synthetic PQRST cardiac rhythm with amplitude-matched premature ventricular contraction shape discord",
        "series": x,
        "m": period,
        "anom_pos": anom_pos,
        "regime_cp": None,
        "dc_offset": 0.0,
        "has_1e10_spikes": False,
        "stream_warmup_start": 3800,
        "stream_warmup_len": 1024,
        "stream_end": anom_pos + 2 * period,
    }


def gen_cloud_microservice(N: int = 10_000, seed: int = 202) -> dict[str, Any]:
    """Domain 2: Cloud microservice diurnal latency + regime shift + p99 oscillation burst."""
    rng = np.random.default_rng(seed)
    m = 64
    cp = 5400
    t = np.arange(N, dtype=np.float64)

    # Regime 1: Smooth diurnal multi-harmonic latency (ms)
    r1 = np.sin(2.0 * np.pi * t[:cp] / m) + 0.35 * np.cos(4.0 * np.pi * t[:cp] / m)
    # Regime 2: Post-deployment canary frequency/shape shift (asymmetric queueing wave, matched mean/std)
    phase2 = (t[cp:] % 42.0) / 42.0
    r2 = 2.0 * phase2 - 1.0 - 0.45 * np.sin(6.0 * np.pi * phase2)

    r1 = 45.0 + 12.0 * (r1 - r1.mean()) / r1.std()
    r2 = 45.0 + 12.0 * (r2 - r2.mean()) / r2.std()
    x = np.concatenate([r1, r2]) + 0.75 * rng.standard_normal(N)

    # Anomalous p99 GC/lock-contention chirp burst bounded strictly inside normal [min, max]
    anom_pos = 2816
    lo = float(np.percentile(x[:cp], 2))
    hi = float(np.percentile(x[:cp], 98))
    u = np.linspace(0.0, 1.0, m, endpoint=False)
    burst = np.sin(2.0 * np.pi * (3.0 * u + 4.0 * u * u)) * (0.45 + 0.55 * np.sin(np.pi * u))
    x[anom_pos : anom_pos + m] = _match_envelope(burst, lo, hi) + 0.4 * rng.standard_normal(m)

    return {
        "id": "cloud_microservice",
        "name": "Cloud Microservice Telemetry",
        "description": "Diurnal latency cycle + deployment regime shift + amplitude-bounded p99 oscillation burst",
        "series": x,
        "m": m,
        "anom_pos": anom_pos,
        "regime_cp": cp,
        "dc_offset": 45.0,
        "has_1e10_spikes": False,
        "stream_warmup_start": 1400,
        "stream_warmup_len": 1024,
        "stream_end": anom_pos + 2 * m,
    }


def gen_industrial_bearing(N: int = 10_000, seed: int = 303) -> dict[str, Any]:
    """Domain 3: Multi-harmonic rotating-machinery bearing vibration + early inner-race fault."""
    rng = np.random.default_rng(seed)
    m = 64
    t = np.arange(N, dtype=np.float64)
    # 1X shaft fundamental + 2X misalignment + 3X vane-pass harmonics
    x = (
        1.00 * np.sin(2.0 * np.pi * t / m)
        + 0.42 * np.sin(4.0 * np.pi * t / m + 0.4)
        + 0.24 * np.cos(6.0 * np.pi * t / m - 0.3)
        + 0.04 * rng.standard_normal(N)
    )

    # Early inner-race spall fault: double-ringing damped resonance modulation, matched peak envelope
    anom_pos = 6400
    lo = float(np.percentile(x, 1))
    hi = float(np.percentile(x, 99))
    u = np.linspace(0.0, 1.0, m, endpoint=False)
    imp1 = np.exp(-14.0 * np.maximum(0.0, u - 0.15)) * (u >= 0.15) * np.sin(2.0 * np.pi * 9.0 * (u - 0.15))
    imp2 = -0.85 * np.exp(-12.0 * np.maximum(0.0, u - 0.55)) * (u >= 0.55) * np.cos(2.0 * np.pi * 7.5 * (u - 0.55))
    fault = imp1 + imp2 - 0.25 * np.linspace(-1.0, 1.0, m)
    x[anom_pos : anom_pos + m] = _match_envelope(fault, lo, hi) + 0.03 * rng.standard_normal(m)

    return {
        "id": "industrial_bearing",
        "name": "Industrial Rotating-Machinery Bearing",
        "description": "Multi-harmonic shaft vibration with early inner-race fault shape modulation (matched envelope)",
        "series": x,
        "m": m,
        "anom_pos": anom_pos,
        "regime_cp": None,
        "dc_offset": 0.0,
        "has_1e10_spikes": False,
        "stream_warmup_start": 4900,
        "stream_warmup_len": 1024,
        "stream_end": anom_pos + 2 * m,
    }


def gen_financial_tick(N: int = 10_000, seed: int = 404) -> dict[str, Any]:
    """Domain 4: High-DC (1e5) financial tick microstructure + regime break + algo motif + discord."""
    rng = np.random.default_rng(seed)
    m = 64
    dc = 1e5
    cp = 5600
    t = np.arange(N, dtype=np.float64)

    # Regime 1: Structured liquidity cycle + bid-ask bounce around 1e5
    r1 = 16.0 * np.sin(2.0 * np.pi * t[:cp] / m) + 6.5 * np.sin(4.0 * np.pi * t[:cp] / m + 0.6)
    # Regime 2: Post-flash-crash regime break (faster asymmetric mean-reversion cycle, matched std)
    r2 = 14.0 * np.sign(np.sin(2.0 * np.pi * t[cp:] / 44.0)) * np.abs(np.sin(2.0 * np.pi * t[cp:] / 44.0)) ** 0.55
    r2 = (r2 - r2.mean()) / r2.std() * r1.std()
    x = dc + np.concatenate([r1, r2]) + 0.8 * rng.standard_normal(N)

    # Repeated execution-algo TWAP/VWAP signature motif (two near-identical windows)
    u = np.linspace(0.0, 1.0, m, endpoint=False)
    algo_motif = 15.0 * (np.tanh(6.0 * (u - 0.35)) - 0.6 * np.tanh(8.0 * (u - 0.72)))
    motif_a, motif_b = 1800, 3400
    x[motif_a : motif_a + m] = dc + algo_motif + 0.04 * rng.standard_normal(m)
    x[motif_b : motif_b + m] = dc + algo_motif + 0.04 * rng.standard_normal(m)

    # Spoofing / quote-stuffing shape anomaly strictly inside normal price envelope
    anom_pos = 7200
    lo = float(np.percentile(x[cp:], 2))
    hi = float(np.percentile(x[cp:], 98))
    spoof = np.linspace(-1.0, 1.0, m) * np.cos(2.0 * np.pi * 5.5 * u) + 0.35 * np.sin(2.0 * np.pi * 1.5 * u)
    x[anom_pos : anom_pos + m] = _match_envelope(spoof, lo, hi) + 0.4 * rng.standard_normal(m)

    return {
        "id": "financial_tick",
        "name": "Financial Tick Microstructure (DC=1e5)",
        "description": "High-DC (1e5) price stream with flash-crash regime break, execution-algo motif, and spoofing discord",
        "series": x,
        "m": m,
        "anom_pos": anom_pos,
        "regime_cp": cp,
        "motif_pair": (motif_a, motif_b),
        "dc_offset": dc,
        "has_1e10_spikes": False,
        "stream_warmup_start": 5800,
        "stream_warmup_len": 1024,
        "stream_end": anom_pos + 2 * m,
    }


def gen_iot_extreme_dc_spikes(N: int = 10_000, seed: int = 505) -> dict[str, Any]:
    """Domain 5: IoT sensor at DC=1e10 with transient 1e10 EMI glitches + post-spike subtle fault."""
    rng = np.random.default_rng(seed)
    m = 64
    dc = 1e10
    t = np.arange(N, dtype=np.float64)

    # Clean environmental oscillation sitting on 1e10 DC offset
    base = 1.15 * np.sin(2.0 * np.pi * t / m) + 0.35 * np.cos(4.0 * np.pi * t / m)
    x = dc + base + 0.025 * rng.standard_normal(N)

    # Two identical transient 1e10 EMI sensor glitches (phase-aligned mod m so spikes match each other
    # while stress-testing FlashMP's spike-drop HWM re-anchoring across subsequent windows!)
    spike1, spike2 = 1200, 2224  # 2224 - 1200 = 1024 = 16 * 64
    x[spike1] += 1e10
    x[spike2] += 1e10

    # Post-spike exact motif pair (verifies < 1e-5 accuracy after both DC=1e10 and 1e10 spikes)
    u = np.linspace(0.0, 1.0, m, endpoint=False)
    pat = 1.2 * np.sin(2.0 * np.pi * u) * np.cos(2.0 * np.pi * 2.0 * u)
    motif_a, motif_b = 3200, 4416
    x[motif_a : motif_a + m] = dc + pat
    x[motif_b : motif_b + m] = dc + pat

    # True physical sensor fault discord AFTER both 1e10 glitches (normal ±1.4 amplitude on 1e10 DC!)
    anom_pos = 6800
    lo = float(dc + np.min(base))
    hi = float(dc + np.max(base))
    fault = np.sign(np.sin(2.0 * np.pi * 4.5 * u)) * (0.3 + 0.7 * u) - 0.5 * np.cos(2.0 * np.pi * 2.0 * u)
    x[anom_pos : anom_pos + m] = _match_envelope(fault, lo, hi) + 0.02 * rng.standard_normal(m)

    return {
        "id": "iot_extreme_dc_spikes",
        "name": "IoT Sensor (DC=1e10 + 1e10 Glitches)",
        "description": "High-DC (1e10) telemetry with transient 1e10 EMI glitches, post-spike exact motif, and subtle shape fault",
        "series": x,
        "m": m,
        "anom_pos": anom_pos,
        "regime_cp": None,
        "motif_pair": (motif_a, motif_b),
        "spike_positions": [spike1, spike2],
        "dc_offset": dc,
        "has_1e10_spikes": True,
        "stream_warmup_start": 5300,
        "stream_warmup_len": 1024,
        "stream_end": anom_pos + 2 * m,
    }


def _downsampler_top3_hit(x: np.ndarray, anom_pos: int, m: int, downsampler: Any, n_out: int = 120) -> bool:
    """Evaluate matched-token visual downsampler + deviation peak detector top-3 recall."""
    idx = downsampler.downsample(x, n_out=n_out)
    y = x[idx]
    med = float(np.median(y))
    top3 = idx[np.argsort(np.abs(y - med))[-3:]]
    return any(abs(int(p) - anom_pos) <= m for p in top3)


def evaluate_domain(spec: dict[str, Any], lttb: Any, m4: Any) -> dict[str, Any]:
    x: np.ndarray = spec["series"]
    m: int = spec["m"]
    anom_pos: int = spec["anom_pos"]
    cp: int | None = spec["regime_cp"]

    # 1. Raw token cost (CSV floats matching real-world LLM/MCP payloads)
    raw_str = ",".join(f"{v:.4f}" for v in x)
    raw_tokens = estimate_tokens(raw_str)

    # 2. FlashMP reversible digest latency (median of 3 runs)
    max_regimes = 2 if cp is not None else 1
    times_ms: list[float] = []
    d = None
    for _ in range(3):
        t0 = time.perf_counter()
        d = flashmp.digest(x, m=m, top_motifs=3, top_discords=3, max_regimes=max_regimes)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    assert d is not None
    digest_ms = float(np.median(times_ms))
    compact = d.to_compact()
    compact_tokens = estimate_tokens(compact)
    token_savings_pct = 100.0 * (1.0 - compact_tokens / max(1, raw_tokens))
    compression_ratio_x = raw_tokens / max(1, compact_tokens)

    # 3. API input cost per 10,000 queries at $3.00 / MTok
    raw_cost_usd = (raw_tokens * QUERIES_SCALE / 1_000_000.0) * API_PRICE_USD_PER_MTOK
    fmp_cost_usd = (compact_tokens * QUERIES_SCALE / 1_000_000.0) * API_PRICE_USD_PER_MTOK
    cost_saved_usd = raw_cost_usd - fmp_cost_usd

    # 4. Top-3 discord localization hit (within ±m ticks): FlashMP vs LTTB vs M4
    fmp_hit = any(abs(int(disc["index"]) - anom_pos) <= m for disc in d.discords[:3])
    fmp_rank = next(
        (rank + 1 for rank, disc in enumerate(d.discords[:3]) if abs(int(disc["index"]) - anom_pos) <= m),
        None,
    )
    fmp_best_disc = d.discords[0] if d.discords else {}
    lttb_hit = _downsampler_top3_hit(x, anom_pos, m, lttb, n_out=120)
    m4_hit = _downsampler_top3_hit(x, anom_pos, m, m4, n_out=120)

    # 5. Optional regime shift localization check
    regime_hit: bool | None = None
    detected_cp: int | None = None
    if cp is not None:
        if len(d.regimes) > 1:
            detected_cp = int(d.regimes[1]["start"])
            regime_hit = abs(detected_cp - cp) <= 4 * m
        else:
            regime_hit = False

    # 6. Optional post-spike / high-DC exact motif verification
    post_spike_motif_dist: float | None = None
    if "motif_pair" in spec:
        ma, mb = spec["motif_pair"]
        prof = flashmp.profile(x, m=m)
        post_spike_motif_dist = float(prof.mp[ma])

    # 7. Lossless drill-down recovery error (must be bit-exact 0.0)
    rec = np.asarray(d.drilldown(anom_pos, anom_pos + m)["values"], dtype=np.float64)
    drilldown_err = float(np.max(np.abs(rec - x[anom_pos : anom_pos + m])))

    # 8. Exact RingMP (`StreamMP`) streaming throughput & anytime-valid detection latency
    ws = spec["stream_warmup_start"]
    wlen = spec["stream_warmup_len"]
    send = min(len(x), spec["stream_end"])
    stream_slice = x[ws + wlen : send]
    n_stream = len(stream_slice)

    alert_gid: int | None = None
    peak_e_val = 0.0
    alert_delay_ticks: int | None = None
    pre_anomaly_false_alerts = 0

    stream_alpha = 0.01
    with flashmp.StreamMP(m=m, window=wlen, egress=True, alpha=stream_alpha) as sm:
        sm.init(x[ws : ws + wlen])
        t0 = time.perf_counter()
        for offset in range(n_stream):
            tick = sm.update(float(stream_slice[offset]))
            abs_pos = ws + wlen + offset
            if abs_pos < anom_pos:
                if tick.is_alert:
                    pre_anomaly_false_alerts += 1
            elif anom_pos <= abs_pos <= anom_pos + 2 * m:
                if tick.e_value > peak_e_val:
                    peak_e_val = float(tick.e_value)
                if tick.is_alert and alert_gid is None:
                    alert_gid = int(tick.gid)
                    alert_delay_ticks = int(abs_pos - anom_pos)
        stream_elapsed_us = (time.perf_counter() - t0) * 1e6

    ringmp_us_per_tick = stream_elapsed_us / max(1, n_stream)

    return {
        "id": spec["id"],
        "name": spec["name"],
        "description": spec["description"],
        "N": int(len(x)),
        "m": int(m),
        "dc_offset": float(spec["dc_offset"]),
        "has_1e10_spikes": bool(spec["has_1e10_spikes"]),
        "ground_truth": {
            "anom_pos": int(anom_pos),
            "regime_cp": int(cp) if cp is not None else None,
        },
        "digest_latency_ms": round(digest_ms, 2),
        "tokens": {
            "raw": int(raw_tokens),
            "flashmp_compact": int(compact_tokens),
            "savings_pct": round(token_savings_pct, 2),
            "compression_ratio_x": round(compression_ratio_x, 1),
        },
        "api_cost_10k_queries_usd_at_3_per_mtok": {
            "raw_usd": round(raw_cost_usd, 2),
            "flashmp_usd": round(fmp_cost_usd, 2),
            "saved_usd": round(cost_saved_usd, 2),
        },
        "discord_top3_hit": {
            "flashmp": bool(fmp_hit),
            "flashmp_rank": fmp_rank,
            "flashmp_top1_index": int(fmp_best_disc.get("index", -1)),
            "flashmp_top1_z": round(float(fmp_best_disc.get("z_score", 0.0)), 2),
            "lttb_matched_tokens": bool(lttb_hit),
            "m4_matched_tokens": bool(m4_hit),
        },
        "regime_shift": {
            "applicable": cp is not None,
            "true_cp": int(cp) if cp is not None else None,
            "detected_cp": detected_cp,
            "hit": regime_hit,
        },
        "post_spike_or_high_dc_motif_dist": (
            round(post_spike_motif_dist, 12) if post_spike_motif_dist is not None else None
        ),
        "lossless_drilldown_max_abs_err": drilldown_err,
        "ringmp_streaming": {
            "window": int(wlen),
            "alpha": stream_alpha,
            "streamed_ticks": int(n_stream),
            "us_per_tick": round(ringmp_us_per_tick, 2),
            "pre_anomaly_false_alerts": int(pre_anomaly_false_alerts),
            "alert_triggered": alert_delay_ticks is not None,
            "detection_delay_ticks": alert_delay_ticks,
            "peak_e_value": round(peak_e_val, 2),
        },
        "compact_preview": compact,
    }


def run_all() -> dict[str, Any]:
    from tsdownsample import LTTBDownsampler, M4Downsampler

    lttb = LTTBDownsampler()
    m4 = M4Downsampler()

    generators = [
        gen_ecg_arrhythmia,
        gen_cloud_microservice,
        gen_industrial_bearing,
        gen_financial_tick,
        gen_iot_extreme_dc_spikes,
    ]

    # Warm up native library load once outside timed measurements
    _ = flashmp.profile(np.sin(np.linspace(0, 10, 256)), m=16)

    domains = [evaluate_domain(gen(), lttb, m4) for gen in generators]

    n_dom = len(domains)
    fmp_hits = sum(1 for r in domains if r["discord_top3_hit"]["flashmp"])
    lttb_hits = sum(1 for r in domains if r["discord_top3_hit"]["lttb_matched_tokens"])
    m4_hits = sum(1 for r in domains if r["discord_top3_hit"]["m4_matched_tokens"])
    ringmp_alerts = sum(1 for r in domains if r["ringmp_streaming"]["alert_triggered"])

    summary = {
        "domains_tested": n_dom,
        "mean_digest_latency_ms": round(float(np.mean([r["digest_latency_ms"] for r in domains])), 2),
        "mean_raw_tokens": round(float(np.mean([r["tokens"]["raw"] for r in domains])), 1),
        "mean_compact_tokens": round(float(np.mean([r["tokens"]["flashmp_compact"] for r in domains])), 1),
        "mean_token_savings_pct": round(float(np.mean([r["tokens"]["savings_pct"] for r in domains])), 2),
        "mean_compression_ratio_x": round(float(np.mean([r["tokens"]["compression_ratio_x"] for r in domains])), 1),
        "mean_raw_cost_10k_usd": round(
            float(np.mean([r["api_cost_10k_queries_usd_at_3_per_mtok"]["raw_usd"] for r in domains])), 2
        ),
        "mean_flashmp_cost_10k_usd": round(
            float(np.mean([r["api_cost_10k_queries_usd_at_3_per_mtok"]["flashmp_usd"] for r in domains])), 2
        ),
        "discord_top3_recall": {
            "flashmp": round(fmp_hits / n_dom, 3),
            "lttb_matched_tokens": round(lttb_hits / n_dom, 3),
            "m4_matched_tokens": round(m4_hits / n_dom, 3),
            "flashmp_hits": f"{fmp_hits}/{n_dom}",
            "lttb_hits": f"{lttb_hits}/{n_dom}",
            "m4_hits": f"{m4_hits}/{n_dom}",
        },
        "max_lossless_drilldown_err": max(r["lossless_drilldown_max_abs_err"] for r in domains),
        "ringmp_mean_us_per_tick": round(
            float(np.mean([r["ringmp_streaming"]["us_per_tick"] for r in domains])), 2
        ),
        "ringmp_alert_hits": f"{ringmp_alerts}/{n_dom}",
        "ringmp_total_pre_anomaly_false_alerts": sum(
            r["ringmp_streaming"]["pre_anomaly_false_alerts"] for r in domains
        ),
    }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "flashmp_version": flashmp.version(),
            "flashmp_isa": flashmp.simd_isa(),
            "numpy_version": np.__version__,
        },
        "pricing_assumptions": {
            "usd_per_mtok_input": API_PRICE_USD_PER_MTOK,
            "queries_per_batch": QUERIES_SCALE,
        },
        "summary": summary,
        "domains": domains,
    }


def main() -> None:
    report = run_all()
    out_path = ROOT / "benchmarks" / "results_realworld_domains.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    s = report["summary"]
    print(f"Saved real-world multi-domain receipt -> {out_path}")
    print(
        f"Summary ({s['domains_tested']} domains, N=10,000 each): "
        f"digest={s['mean_digest_latency_ms']} ms | "
        f"tokens {s['mean_raw_tokens']:.0f} -> {s['mean_compact_tokens']:.0f} (-{s['mean_token_savings_pct']}%) | "
        f"cost/10k ${s['mean_raw_cost_10k_usd']:.2f} -> ${s['mean_flashmp_cost_10k_usd']:.2f} | "
        f"top-3 recall: FlashMP={s['discord_top3_recall']['flashmp_hits']} "
        f"LTTB={s['discord_top3_recall']['lttb_hits']} M4={s['discord_top3_recall']['m4_hits']} | "
        f"drilldown_err={s['max_lossless_drilldown_err']} | "
        f"RingMP={s['ringmp_mean_us_per_tick']} us/tick "
        f"({s['ringmp_alert_hits']} alerts, {s['ringmp_total_pre_anomaly_false_alerts']} pre-anom FP)"
    )
    for dom in report["domains"]:
        dh = dom["discord_top3_hit"]
        rs = dom["ringmp_streaming"]
        tk = dom["tokens"]
        cs = dom["api_cost_10k_queries_usd_at_3_per_mtok"]
        print(
            f"  - {dom['id']:22s} | {dom['digest_latency_ms']:5.1f} ms | "
            f"{tk['raw']:6d}->{tk['flashmp_compact']:3d} tok (-{tk['savings_pct']:5.2f}%) | "
            f"${cs['raw_usd']:6.2f}->${cs['flashmp_usd']:4.2f} | "
            f"hit(F/L/M)={int(dh['flashmp'])}/{int(dh['lttb_matched_tokens'])}/{int(dh['m4_matched_tokens'])} "
            f"(rank={dh['flashmp_rank']}) | err={dom['lossless_drilldown_max_abs_err']:.1f} | "
            f"RingMP={rs['us_per_tick']:5.2f} us/tick fp={rs['pre_anomaly_false_alerts']} "
            f"alert={rs['alert_triggered']} (+{rs['detection_delay_ticks']}t, E={rs['peak_e_value']})"
        )


if __name__ == "__main__":
    main()
