#!/usr/bin/env python3
"""Reversible Agentic Digest + Multi-Agent Firehose SwarmMonitor example."""

from __future__ import annotations

import numpy as np
import flashmp


def main():
    rng = np.random.default_rng(7)
    N, m = 10_000, 50
    t = np.arange(N, dtype=np.float64)
    x = np.sin(2 * np.pi * t / m) + 0.05 * rng.standard_normal(N)
    # Regime shift at t=5500 + subtle shape discord at t=2200
    x[5500:] = np.sign(np.sin(2 * np.pi * t[5500:] / 33.0)) * 0.8 + 0.05 * rng.standard_normal(N - 5500)
    x[2200:2250] = np.linspace(-1.0, 1.0, m) * np.cos(np.linspace(0, 7 * np.pi, m))

    d = flashmp.digest(x, m=m, top_motifs=2, top_discords=3, max_regimes=2)
    print("=== AGENT-READY COMPACT DIGEST ===")
    print(d.to_compact())
    print("\n=== LOSSLESS DRILL-DOWN ON TOP DISCORD ===")
    dd = d.drilldown_discord(0)
    print(f"Discord index={dd['window_start']}, exact first 5 raw values={dd['values'][:5]}")


if __name__ == "__main__":
    main()
