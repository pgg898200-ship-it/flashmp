#!/usr/bin/env python3
"""Generate benchmarks/RESULTS.md from verified JSON receipts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"


def render() -> str:
    inc = json.loads((BENCH / "results_incumbents.json").read_text())
    tok = json.loads((BENCH / "results_agent_tokens.json").read_text())
    swm = json.loads((BENCH / "results_swarm_sift.json").read_text())
    env = inc["environment"]

    lines = [
        "# FlashMP Verified Silicon Benchmarks",
        "",
        f"> **Measured:** `{inc['timestamp_utc'][:10]}` · **Hardware:** `{env['machine']}` ({env['cpu_count']} vCPU) · **ISA:** `{env['flashmp_isa']}`  ",
        f"> **Versions:** `flashmp {env['flashmp_version']}` · `stumpy {env['stumpy_version']}` · `pyscamp {env['pyscamp_version']}` · `ruptures {env['ruptures_version']}` · `Python {env['python']}`",
        "",
        "All timings are live wall-clock medians on identical inputs and matched thread counts. Every batch and streaming run verifies exact numerical parity against reference outputs.",
        "",
        "## 1. Cold Start (Fresh Subprocess Import + First `N=4,096, m=64` Self-Join)",
        "",
        "| Library | Cold Start | Relative Speed | Runtime Dependencies |",
        "|---|---:|---:|---|",
        f"| **FlashMP (`import flashmp` only)** | **`{inc['cold_start_ms']['flashmp_import_only']:.2f} ms`** | — | **0 required** (pure stdlib + C++20 `.so`) |",
        f"| **FlashMP (import + first self-join)** | **`{inc['cold_start_ms']['flashmp_first_call']:.1f} ms`** | **`{inc['cold_start_ms']['stumpy_first_call']/inc['cold_start_ms']['flashmp_first_call']:,.0f}×`** vs STUMPY | 0 required (optional NumPy) |",
        f"| `pyscamp {env['pyscamp_version']}` | `{inc['cold_start_ms']['pyscamp_first_call']:.1f} ms` | `{inc['cold_start_ms']['stumpy_first_call']/inc['cold_start_ms']['pyscamp_first_call']:,.0f}×` vs STUMPY | C++/pybind11 (source build on Linux) |",
        f"| `stumpy {env['stumpy_version']}` | `{inc['cold_start_ms']['stumpy_first_call']:,.1f} ms` | `1.0×` | NumPy + SciPy + Numba + llvmlite (~109 MiB) |",
        "",
        "## 2. Warm Batch Self-Join (`m=64`, Matched Threads & Exact Parity)",
        "",
        "| Length `N` | FlashMP (2T) | pyscamp (2T, fp64) | STUMPY (2T) | vs pyscamp | vs STUMPY | Max Err vs STUMPY |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in inc["batch_self_join"]:
        lines.append(
            f"| `{r['N']:,}` | **`{r['flashmp_ms']:.2f} ms`** | `{r['pyscamp_ms']:.2f} ms` | `{r['stumpy_ms']:.2f} ms` | **`{r['speedup_vs_pyscamp']:.2f}×`** | **`{r['speedup_vs_stumpy']:.2f}×`** | `{r['max_err_vs_stumpy']:.2e}` |"
        )

    ab = inc["batch_ab_join"]
    lines += [
        "",
        f"## 3. Warm AB-Join / Contamination Scan (`N_A={ab['NA']:,}, N_B={ab['NB']:,}, m={ab['m']}`)",
        "",
        "| Library | Time (2T) | vs STUMPY | vs pyscamp | Max Err vs STUMPY |",
        "|---|---:|---:|---:|---:|",
        f"| **FlashMP** | **`{ab['flashmp_ms']:.2f} ms`** | **`{ab['speedup_vs_stumpy']:.2f}×`** | **`{ab['speedup_vs_pyscamp']:.2f}×`** | `{ab['max_err_vs_stumpy']:.2e}` |",
        f"| `pyscamp` (fp64) | `{ab['pyscamp_ms']:.2f} ms` | `{ab['stumpy_ms']/ab['pyscamp_ms']:.2f}×` | `1.00×` | — |",
        f"| `stumpy` | `{ab['stumpy_ms']:.2f} ms` | `1.00×` | — | `0.00e+00` |",
        "",
        "## 4. Multivariate Matrix Profile (`mstump`, `m=64`, 2T — Unsupported by `pyscamp`)",
        "",
        "| Shape `(D, N)` | FlashMP | STUMPY | Speedup | Max Err vs STUMPY |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in inc["multivariate_mstump"]:
        lines.append(
            f"| `({r['D']}, {r['N']:,})` | **`{r['flashmp_ms']:.2f} ms`** | `{r['stumpy_ms']:,.2f} ms` | **`{r['speedup_vs_stumpy']:.1f}×`** | `{r['max_err_vs_stumpy']:.2e}` |"
        )

    lines += [
        "",
        "## 5. Exact Bounded-Window Streaming (`RingMP` vs `stumpy.stumpi(egress=True)`, `m=64`)",
        "",
        "| Window `W` | FlashMP Full (batch / py-loop) | FlashMP Causal | STUMPY `stumpi` | Speedup (py-loop / batch / causal) | FlashMP Egress Err | STUMPY Egress Err |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in inc["streaming_egress"]:
        lines.append(
            f"| `{r['W']:,}` | **`{r['flashmp_full_us_per_tick']:.2f}` / `{r['flashmp_full_pyloop_us_per_tick']:.2f} µs`** | **`{r['flashmp_causal_us_per_tick']:.2f} µs`** | `{r['stumpy_stumpi_us_per_tick']:.1f} µs` | **`{r['speedup_pyloop_vs_stumpy']:.1f}×` / `{r['speedup_full_vs_stumpy']:.1f}×` / `{r['speedup_causal_vs_stumpy']:.1f}×`** | **`{r['flashmp_egress_max_err']:.2e}`** | `{r['stumpy_egress_max_err']:.3e}` *(stale)* |"
        )

    dc = inc["high_dc_stability"]
    sp = dc["transient_spike_1e10_motif_err"]
    reg = inc["regime_segmentation"]
    lines += [
        "",
        f"## 6. Extreme Numerical Stability (`DC = {dc['dc_offset']:.0e}` & Transient `1e10` Spikes)",
        "",
        "| Library | High-DC Max Abs Err (`DC=1e10`, `N=2,048`) | Post-`1e10`-Spike Exact Motif Err (True = `0.0`) |",
        "|---|---:|---:|",
        f"| **FlashMP** | **`{dc['flashmp_max_err']:.3e}`** | **`{sp['flashmp']:.3e}`** |",
        f"| `pyscamp` (fp64) | `{dc['pyscamp_max_err']:.3e}` | `{sp['pyscamp']:.3e}` |",
        f"| `stumpy` | `{dc['stumpy_max_err']:.3e}` | `{sp['stumpy']:.3e}` |",
        "",
        f"## 7. Unsupervised Regime Segmentation (`N={reg['N']:,}`, True Changepoint `{reg['true_changepoint']}`)",
        "",
        "| Method | Time | Detected CP | Error (ticks) | Speedup |",
        "|---|---:|---:|---:|---:|",
        f"| **FlashMP (MP + FLOSS)** | **`{reg['flashmp_floss_ms']:.2f} ms`** | `{reg['flashmp_detected_cp']}` | `{reg['flashmp_abs_err_ticks']}` | **`{reg['speedup_vs_ruptures_rbf']:.1f}×`** |",
        f"| `ruptures` PELT (`rbf`) | `{reg['ruptures_pelt_rbf_ms']:.2f} ms` | `{reg['ruptures_detected_cps']}` | — | `1.0×` |",
        "",
        f"## 8. Reversible Agentic Context Compression ({tok['trials']} Trials × `N={tok['series_length']:,}`)",
        "",
        "| Representation | Mean Tokens | Token Savings | Top-3 Shape Anomaly Recall | Regime Recall | Lossless Drill-Down Err |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Raw CSV/JSON Floats | `{tok['raw_tokens_mean']:,.0f}` | `0.0%` | — | — | `0.0` |",
        f"| `tsdownsample` LTTB (matched tokens) | `~240` | `~99%` | `{tok['shape_anomaly_recall_top3']['lttb_matched_tokens']*100:.1f}%` | — | Lossy |",
        f"| `tsdownsample` M4 (matched tokens) | `~240` | `~99%` | `{tok['shape_anomaly_recall_top3']['m4_matched_tokens']*100:.1f}%` | — | Lossy |",
        f"| **FlashMP `digest()`** (`{tok['flashmp_median_ms']:.1f} ms`) | **`{tok['flashmp_digest_tokens_mean']:.0f}`** | **`{tok['token_reduction_pct']:.2f}%` (`{tok['compression_ratio_x']:.0f}×`)** | **`{tok['shape_anomaly_recall_top3']['flashmp_digest']*100:.1f}%`** | **`{tok['regime_shift_recall']*100:.1f}%`** | **`{tok['lossless_drilldown_max_abs_err']:.1f}` (exact)** |",
        "",
        "## 9. Multi-Agent Firehose Sifting (`SwarmMonitor`, 3.5× Loud Distractors, 16-Shuffle Commutativity)",
        "",
        "| Channels `C` | Throughput (ch-ticks/s) | Latency / ch-tick | Fault Recall | Loud Distractor FPs | Brief Tokens | Order-Invariant (16 shuffles) |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in swm["firehose_scaling"]:
        lines.append(
            f"| `{r['channels']:,}` | **`{r['channel_ticks_per_sec']:,.0f}`** | `{r['us_per_channel_tick']:.2f} µs` | **`{r['fault_recall']*100:.0f}%`** (`{r['tp_found']}/{r['total_faults']}`) | **`{r['loud_distractor_false_positives']}`** | `{r['brief_tokens']}` | `{'PASS' if r['order_invariant_16_shuffles'] else 'FAIL'}` |"
        )

    rw_path = BENCH / "results_realworld_domains.json"
    if rw_path.is_file():
        rw = json.loads(rw_path.read_text())
        s = rw["summary"]
        rec = s["discord_top3_recall"]
        lines += [
            "",
            "## 10. Five-Domain Real-World Telemetry Evaluation (`N=10,000` Each, `$3/MTok` API Economics)",
            "",
            "| Domain | `m` | `digest()` | Raw → Digest Tokens | Token Savings | Cost / 10k Queries | Top-3 Discord (FlashMP / LTTB / M4) | Drill Err | RingMP (`W=1,024`) | Pre-Anom FPs |",
            "|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|",
        ]
        for d in rw["domains"]:
            tok_d = d["tokens"]
            cost_d = d["api_cost_10k_queries_usd_at_3_per_mtok"]
            loc = d["discord_top3_hit"]
            st = d["ringmp_streaming"]
            lines.append(
                f"| **{d['name']}** | `{d['m']}` | `{d['digest_latency_ms']:.1f} ms` | `{tok_d['raw']:,} → {tok_d['flashmp_compact']}` | **`-{tok_d['savings_pct']:.2f}%`** | `${cost_d['raw_usd']:,.2f} → ${cost_d['flashmp_usd']:.2f}` | **`{'Hit (#%d)' % loc['flashmp_rank'] if loc['flashmp'] else 'Miss'}`** / `{'Hit' if loc['lttb_matched_tokens'] else 'Miss'}` / `{'Hit' if loc['m4_matched_tokens'] else 'Miss'}` | `{d['lossless_drilldown_max_abs_err']:.1f}` | `{st['us_per_tick']:.2f} µs/tick` | `{st['pre_anomaly_false_alerts']}` |"
            )
        lines.append(
            f"| **Mean / Total ({s['domains_tested']} domains)** | — | **`{s['mean_digest_latency_ms']:.1f} ms`** | **`{s['mean_raw_tokens']:,.0f} → {s['mean_compact_tokens']:.1f}`** | **`-{s['mean_token_savings_pct']:.2f}%` (`{s['mean_compression_ratio_x']:.0f}×`)** | **`${s['mean_raw_cost_10k_usd']:,.2f} → ${s['mean_flashmp_cost_10k_usd']:.2f}`** | **`{rec['flashmp_hits']}` (`{rec['flashmp']*100:.0f}%`) / `{rec['lttb_hits']}` / `{rec['m4_hits']}`** | **`{s['max_lossless_drilldown_err']:.1f}`** | **`{s['ringmp_mean_us_per_tick']:.2f} µs/tick`** | **`{s['ringmp_total_pre_anomaly_false_alerts']}`** |"
        )

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python benchmarks/bench_vs_incumbents.py",
        "python benchmarks/bench_agent_tokens.py",
        "python benchmarks/bench_swarm_sift.py",
        "python benchmarks/bench_realworld_domains.py",
        "python benchmarks/render_report.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main():
    md = render()
    out = BENCH / "RESULTS.md"
    out.write_text(md)
    print(f"Wrote {out} ({len(md)} bytes)")


if __name__ == "__main__":
    main()
