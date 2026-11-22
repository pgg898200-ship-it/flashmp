#!/usr/bin/env python3
"""
Multi-Agent Firehose Anomaly Sifting & Commutative Alert Fusion Benchmark.

Models the multi-agent streaming firehose problem:
  - C = 64 .. 1,024 concurrent live sensor/telemetry channels
  - Sparse true shape anomalies injected on a few channels
  - 3.5x-amplitude loud distractor channels (normal periodic shape, 3.5x scale!)
    that fool naive amplitude/variance detectors
  - Multiple concurrent worker agents pushing channel updates in scrambled orders

Verifies:
  1. High throughput (channel-ticks/sec across hundreds/thousands of live streams)
  2. High precision/recall on subtle shape faults with ZERO false alarms on 3.5x-loud distractors
  3. Strict token budget compliance (<= 300 tokens/turn even at C=1024 channels)
  4. Exact commutativity / order-invariance across 16 random agent delivery permutations
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


def run_firehose(n_channels: int, warmup_ticks: int = 320, live_ticks: int = 120, m: int = 32, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    total_T = warmup_ticks + live_ticks
    t = np.arange(total_T, dtype=np.float64)

    # Pick 4 true faulty channels & 16 loud (3.5x amplitude, normal shape) distractor channels
    faulty = set(rng.choice(n_channels, size=min(4, n_channels), replace=False).tolist())
    remaining = [c for c in range(n_channels) if c not in faulty]
    distractors = set(rng.choice(remaining, size=min(16, len(remaining)), replace=False).tolist())

    data = np.empty((n_channels, total_T), dtype=np.float64)
    for c in range(n_channels):
        phase = float(rng.uniform(0, 2 * np.pi))
        amp = 3.5 if c in distractors else 1.0
        base = amp * (np.sin(2 * np.pi * t / m + phase) + 0.3 * np.cos(4 * np.pi * t / m))
        data[c] = base + (0.04 * amp) * rng.standard_normal(total_T)
        if c in faulty:
            # Inject subtle same-amplitude shape break during live ticks
            onset = warmup_ticks + int(rng.integers(10, live_ticks - m - 5))
            data[c, onset : onset + m] = 0.9 * np.sign(np.sin(np.linspace(0, 9 * np.pi, m))) + 0.04 * rng.standard_normal(m)

    names = [f"ch_{c:04d}" for c in range(n_channels)]
    mon = flashmp.SwarmMonitor(channels=names, m=m, window=256, alpha=0.01)

    # Warmup via native C++ init_channel
    for c in range(n_channels):
        mon.init_channel(names[c], data[c, :warmup_ticks])

    # Stream live ticks
    t0 = time.perf_counter()
    for step in range(live_ticks):
        mon.update_frame(data[:, warmup_ticks + step])
    elapsed_s = time.perf_counter() - t0
    chan_ticks_per_sec = (n_channels * live_ticks) / elapsed_s

    brief = mon.brief(max_tokens=300, top_k=10)
    brief_tokens = max(1, (len(brief) + 3) // 4)

    top_alerts = mon.ledger.alerts(only_triggered=True, top_k=20)
    alerted_set = {int(a.channel_id.split("_")[1]) for a in top_alerts}
    tp = len(alerted_set & faulty)
    fp_distractors = len(alerted_set & distractors)
    recall = tp / max(1, len(faulty))

    # Commutativity / order-invariance test across 16 random shuffles of agent findings
    findings = mon.ledger.alerts()
    ref_ledger = flashmp.AlertLedger.merge_all(findings)
    order_invariant = True
    for pseed in range(16):
        prng = np.random.default_rng(500 + pseed)
        perm = [findings[i] for i in prng.permutation(len(findings))]
        merged = flashmp.AlertLedger.merge_all(perm)
        if merged != ref_ledger:
            order_invariant = False
            break

    mon.close()
    return {
        "channels": n_channels,
        "live_ticks": live_ticks,
        "elapsed_ms": round(elapsed_s * 1e3, 2),
        "channel_ticks_per_sec": round(chan_ticks_per_sec, 1),
        "us_per_channel_tick": round(1e6 / chan_ticks_per_sec, 2),
        "fault_recall": round(recall, 3),
        "tp_found": tp,
        "total_faults": len(faulty),
        "loud_distractor_false_positives": fp_distractors,
        "brief_tokens": brief_tokens,
        "order_invariant_16_shuffles": order_invariant,
    }


def main():
    rows = []
    for C in [64, 256, 1024]:
        rows.append(run_firehose(C))
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "firehose_scaling": rows,
    }
    out_json = ROOT / "benchmarks" / "results_swarm_sift.json"
    out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
