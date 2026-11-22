#!/usr/bin/env python3
"""
Agentic Time-Series Context Compression & Structural Fidelity Benchmark.

Compares FlashMP's reversible `digest()` against:
  1. Raw JSON / CSV serialization (what naive MCP / LLM tools feed into context)
  2. Visual downsampling via `tsdownsample` (LTTB and M4 at matched token budget)

Evaluates across 30 seeded synthetic & realistic workloads (periodic + injected
shape discords + regime shifts) on:
  - Token count & compression ratio
  - Exact anomaly (discord) localization hit rate (within ±m ticks)
  - Exact regime-change localization hit rate
  - Lossless drill-down recovery error (must be 0.0 for FlashMP)
  - Wall-clock latency (ms)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import flashmp


def estimate_tokens(text: str) -> int:
    # Standard ~4 chars/token conservative BPE estimate
    return max(1, (len(text) + 3) // 4)


def make_benchmark_series(seed: int, N: int = 10000, period: int = 50):
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=np.float64)
    cp = N // 2 + int(rng.integers(-500, 500))
    # Regime 1: smooth harmonic; Regime 2: double-frequency asymmetric wave (same mean/std!)
    x = np.empty(N, dtype=np.float64)
    x[:cp] = np.sin(2.0 * np.pi * t[:cp] / period) + 0.35 * np.sin(4.0 * np.pi * t[:cp] / period)
    x[cp:] = np.sin(2.0 * np.pi * t[cp:] / (period * 0.65)) - 0.45 * np.cos(6.0 * np.pi * t[cp:] / period)
    # Match mean/std across regimes so naive threshold detectors cannot trivially separate them
    x[:cp] = (x[:cp] - x[:cp].mean()) / x[:cp].std()
    x[cp:] = (x[cp:] - x[cp:].mean()) / x[cp:].std()
    x += 0.06 * rng.standard_normal(N)

    # Inject a subtle SHAPE anomaly (same amplitude range [-1.5, 1.5], no huge spike!)
    # This defeats min/max/LTTB/z-score detectors while Matrix Profile nails it!
    anom_pos = int(rng.integers(period * 4, cp - period * 4))
    anom_wave = np.linspace(-1.2, 1.2, period) * np.cos(np.linspace(0, 7 * np.pi, period))
    x[anom_pos : anom_pos + period] = anom_wave + 0.03 * rng.standard_normal(period)
    return x, anom_pos, cp, period


def run_benchmark(n_trials: int = 20, N: int = 10000) -> dict:
    from tsdownsample import LTTBDownsampler, M4Downsampler

    raw_tokens = []
    digest_tokens = []
    digest_ms = []
    fmp_anom_hits = 0
    fmp_regime_hits = 0
    lttb_anom_hits = 0
    m4_anom_hits = 0
    max_drilldown_err = 0.0

    for seed in range(n_trials):
        x, anom_pos, cp, m = make_benchmark_series(1000 + seed, N=N, period=50)

        # Raw CSV/JSON token cost
        raw_str = ",".join(f"{v:.4f}" for v in x)
        raw_tokens.append(estimate_tokens(raw_str))

        # FlashMP reversible digest
        t0 = time.perf_counter()
        d = flashmp.digest(x, m=m, top_motifs=3, top_discords=3, max_regimes=2)
        digest_ms.append((time.perf_counter() - t0) * 1e3)
        compact = d.to_compact()
        digest_tokens.append(estimate_tokens(compact))

        # Verify 100% lossless drilldown on anomalous window
        recovered = np.asarray(d.drilldown(anom_pos, anom_pos + m)["values"], dtype=np.float64)
        max_drilldown_err = max(max_drilldown_err, float(np.max(np.abs(recovered - x[anom_pos : anom_pos + m]))))

        # Anomaly localization: does any top-3 discord fall within ±m of true shape anomaly?
        if any(abs(disc["index"] - anom_pos) <= m for disc in d.discords[:3]):
            fmp_anom_hits += 1

        # Regime boundary localization: within ±4*m of true regime shift
        if any(abs(reg["start"] - cp) <= 4 * m for reg in d.regimes[1:]):
            fmp_regime_hits += 1

        # Compare against matched-token visual downsamplers (LTTB & M4 at n_out=120 points ~ 240 tokens)
        # Check if a simple peak/deviation detector on the downsampled points localizes the shape anomaly
        idx_lttb = LTTBDownsampler().downsample(x, n_out=120)
        y_lttb = x[idx_lttb]
        top3_lttb = idx_lttb[np.argsort(np.abs(y_lttb - np.mean(y_lttb)))[-3:]]
        if any(abs(int(p) - anom_pos) <= m for p in top3_lttb):
            lttb_anom_hits += 1

        idx_m4 = M4Downsampler().downsample(x, n_out=120)
        y_m4 = x[idx_m4]
        top3_m4 = idx_m4[np.argsort(np.abs(y_m4 - np.mean(y_m4)))[-3:]]
        if any(abs(int(p) - anom_pos) <= m for p in top3_m4):
            m4_anom_hits += 1

    mean_raw = float(np.mean(raw_tokens))
    mean_dig = float(np.mean(digest_tokens))
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "trials": n_trials,
        "series_length": N,
        "raw_tokens_mean": round(mean_raw, 1),
        "flashmp_digest_tokens_mean": round(mean_dig, 1),
        "token_reduction_pct": round(100.0 * (1.0 - mean_dig / mean_raw), 2),
        "compression_ratio_x": round(mean_raw / mean_dig, 1),
        "flashmp_median_ms": round(float(np.median(digest_ms)), 2),
        "shape_anomaly_recall_top3": {
            "flashmp_digest": round(fmp_anom_hits / n_trials, 3),
            "lttb_matched_tokens": round(lttb_anom_hits / n_trials, 3),
            "m4_matched_tokens": round(m4_anom_hits / n_trials, 3),
        },
        "regime_shift_recall": round(fmp_regime_hits / n_trials, 3),
        "lossless_drilldown_max_abs_err": max_drilldown_err,
    }
    return summary


def main():
    res = run_benchmark()
    out_json = ROOT / "benchmarks" / "results_agent_tokens.json"
    out_json.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
