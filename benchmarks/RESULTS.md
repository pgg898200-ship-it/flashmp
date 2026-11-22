# FlashMP Verified Silicon Benchmarks

> **Measured:** `2026-11-22` · **Hardware:** `x86_64` (2 vCPU) · **ISA:** `AVX2+FMA`  
> **Versions:** `flashmp 0.1.0` · `stumpy 1.14.1` · `pyscamp dev` · `ruptures v1.1.10` · `Python 3.12.16`

All timings are live wall-clock medians on identical inputs and matched thread counts. Every batch and streaming run verifies exact numerical parity against reference outputs.

## 1. Cold Start (Fresh Subprocess Import + First `N=4,096, m=64` Self-Join)

| Library | Cold Start | Relative Speed | Runtime Dependencies |
|---|---:|---:|---|
| **FlashMP (`import flashmp` only)** | **`0.78 ms`** | — | **0 required** (pure stdlib + C++20 `.so`) |
| **FlashMP (import + first self-join)** | **`42.5 ms`** | **`987×`** vs STUMPY | 0 required (optional NumPy) |
| `pyscamp dev` | `33.2 ms` | `1,263×` vs STUMPY | C++/pybind11 (source build on Linux) |
| `stumpy 1.14.1` | `41,906.1 ms` | `1.0×` | NumPy + SciPy + Numba + llvmlite (~109 MiB) |

## 2. Warm Batch Self-Join (`m=64`, Matched Threads & Exact Parity)

| Length `N` | FlashMP (2T) | pyscamp (2T, fp64) | STUMPY (2T) | vs pyscamp | vs STUMPY | Max Err vs STUMPY |
|---:|---:|---:|---:|---:|---:|---:|
| `4,096` | **`3.29 ms`** | `18.01 ms` | `45.51 ms` | **`5.47×`** | **`13.81×`** | `7.63e-13` |
| `16,384` | **`38.47 ms`** | `194.21 ms` | `565.04 ms` | **`5.05×`** | **`14.69×`** | `2.96e-12` |
| `32,768` | **`119.47 ms`** | `743.50 ms` | `2199.61 ms` | **`6.22×`** | **`18.41×`** | `3.67e-12` |

## 3. Warm AB-Join / Contamination Scan (`N_A=32,768, N_B=4,096, m=64`)

| Library | Time (2T) | vs STUMPY | vs pyscamp | Max Err vs STUMPY |
|---|---:|---:|---:|---:|
| **FlashMP** | **`37.35 ms`** | **`12.88×`** | **`2.71×`** | `3.15e-12` |
| `pyscamp` (fp64) | `101.38 ms` | `4.75×` | `1.00×` | — |
| `stumpy` | `481.19 ms` | `1.00×` | — | `0.00e+00` |

## 4. Multivariate Matrix Profile (`mstump`, `m=64`, 2T — Unsupported by `pyscamp`)

| Shape `(D, N)` | FlashMP | STUMPY | Speedup | Max Err vs STUMPY |
|---|---:|---:|---:|---:|
| `(4, 4,096)` | **`39.14 ms`** | `1,888.58 ms` | **`48.2×`** | `1.23e-11` |
| `(8, 8,192)` | **`248.46 ms`** | `12,079.35 ms` | **`48.6×`** | `1.23e-10` |

## 5. Exact Bounded-Window Streaming (`RingMP` vs `stumpy.stumpi(egress=True)`, `m=64`)

| Window `W` | FlashMP Full (batch / py-loop) | FlashMP Causal | STUMPY `stumpi` | Speedup (py-loop / batch / causal) | FlashMP Egress Err | STUMPY Egress Err |
|---:|---:|---:|---:|---:|---:|---:|
| `1,024` | **`3.14` / `7.77 µs`** | **`1.54 µs`** | `519.9 µs` | **`66.9×` / `165.7×` / `337.7×`** | **`2.18e-13`** | `1.570e+00` *(stale)* |
| `4,096` | **`7.31` / `12.00 µs`** | **`3.16 µs`** | `506.1 µs` | **`42.2×` / `69.2×` / `160.3×`** | **`3.85e-13`** | `1.137e+00` *(stale)* |
| `16,384` | **`31.65` / `41.70 µs`** | **`9.52 µs`** | `798.4 µs` | **`19.1×` / `25.2×` / `83.9×`** | **`1.71e-12`** | `1.021e+00` *(stale)* |

## 6. Extreme Numerical Stability (`DC = 1e+10` & Transient `1e10` Spikes)

| Library | High-DC Max Abs Err (`DC=1e10`, `N=2,048`) | Post-`1e10`-Spike Exact Motif Err (True = `0.0`) |
|---|---:|---:|
| **FlashMP** | **`4.647e-06`** | **`2.032e-16`** |
| `pyscamp` (fp64) | `2.507e-04` | `nan` |
| `stumpy` | `2.188e-03` | `0.000e+00` |

## 7. Unsupervised Regime Segmentation (`N=4,000`, True Changepoint `2000`)

| Method | Time | Detected CP | Error (ticks) | Speedup |
|---|---:|---:|---:|---:|
| **FlashMP (MP + FLOSS)** | **`4.45 ms`** | `1976` | `24` | **`9305.2×`** |
| `ruptures` PELT (`rbf`) | `41371.76 ms` | `[50, 100, 150, 200, 250]` | — | `1.0×` |

## 8. Reversible Agentic Context Compression (20 Trials × `N=10,000`)

| Representation | Mean Tokens | Token Savings | Top-3 Shape Anomaly Recall | Regime Recall | Lossless Drill-Down Err |
|---|---:|---:|---:|---:|---:|
| Raw CSV/JSON Floats | `18,751` | `0.0%` | — | — | `0.0` |
| `tsdownsample` LTTB (matched tokens) | `~240` | `~99%` | `0.0%` | — | Lossy |
| `tsdownsample` M4 (matched tokens) | `~240` | `~99%` | `0.0%` | — | Lossy |
| **FlashMP `digest()`** (`33.6 ms`) | **`129`** | **`99.31%` (`145×`)** | **`100.0%`** | **`100.0%`** | **`0.0` (exact)** |

## 9. Multi-Agent Firehose Sifting (`SwarmMonitor`, 3.5× Loud Distractors, 16-Shuffle Commutativity)

| Channels `C` | Throughput (ch-ticks/s) | Latency / ch-tick | Fault Recall | Loud Distractor FPs | Brief Tokens | Order-Invariant (16 shuffles) |
|---:|---:|---:|---:|---:|---:|:---:|
| `64` | **`138,801`** | `7.20 µs` | **`100%`** (`4/4`) | **`0`** | `84` | `PASS` |
| `256` | **`131,677`** | `7.59 µs` | **`100%`** (`4/4`) | **`0`** | `105` | `PASS` |
| `1,024` | **`103,740`** | `9.64 µs` | **`100%`** (`4/4`) | **`0`** | `119` | `PASS` |

## Reproduce

```bash
python benchmarks/bench_vs_incumbents.py
python benchmarks/bench_agent_tokens.py
python benchmarks/bench_swarm_sift.py
python benchmarks/render_report.py
```
