# FlashMP Verified Silicon Benchmarks

> **Measured:** `2026-11-22` · **Hardware:** `x86_64` (2 vCPU) · **ISA:** `AVX2+FMA`  
> **Versions:** `flashmp 0.2.0` · `stumpy 1.14.1` · `pyscamp dev` · `ruptures v1.1.10` · `Python 3.12.16`

All timings are live wall-clock medians on identical inputs and matched thread counts. Every batch and streaming run verifies exact numerical parity against reference outputs.

## 1. Cold Start (Fresh Subprocess Import + First `N=4,096, m=64` Self-Join)

| Library | Cold Start | Relative Speed | Runtime Dependencies |
|---|---:|---:|---|
| **FlashMP (`import flashmp` only)** | **`0.57 ms`** | — | **0 required** (pure stdlib + C++20 `.so`) |
| **FlashMP (import + first self-join)** | **`43.2 ms`** | **`880×`** vs STUMPY | 0 required (optional NumPy) |
| `pyscamp dev` | `30.0 ms` | `1,266×` vs STUMPY | C++/pybind11 (source build on Linux) |
| `stumpy 1.14.1` | `38,025.1 ms` | `1.0×` | NumPy + SciPy + Numba + llvmlite (~109 MiB) |

## 2. Warm Batch Self-Join (`m=64`, Matched Threads & Exact Parity)

| Length `N` | FlashMP (2T) | pyscamp (2T, fp64) | STUMPY (2T) | vs pyscamp | vs STUMPY | Max Err vs STUMPY |
|---:|---:|---:|---:|---:|---:|---:|
| `4,096` | **`5.30 ms`** | `17.08 ms` | `60.01 ms` | **`3.22×`** | **`11.32×`** | `9.32e-13` |
| `16,384` | **`39.61 ms`** | `275.17 ms` | `576.14 ms` | **`6.95×`** | **`14.54×`** | `2.95e-12` |
| `32,768` | **`87.33 ms`** | `767.63 ms` | `2377.48 ms` | **`8.79×`** | **`27.22×`** | `3.69e-12` |

## 3. Warm AB-Join / Contamination Scan (`N_A=32,768, N_B=4,096, m=64`)

| Library | Time (2T) | vs STUMPY | vs pyscamp | Max Err vs STUMPY |
|---|---:|---:|---:|---:|
| **FlashMP** | **`31.84 ms`** | **`15.79×`** | **`3.25×`** | `3.09e-12` |
| `pyscamp` (fp64) | `103.45 ms` | `4.86×` | `1.00×` | — |
| `stumpy` | `502.80 ms` | `1.00×` | — | `0.00e+00` |

## 4. Multivariate Matrix Profile (`mstump`, `m=64`, 2T — Unsupported by `pyscamp`)

| Shape `(D, N)` | FlashMP | STUMPY | Speedup | Max Err vs STUMPY |
|---|---:|---:|---:|---:|
| `(4, 4,096)` | **`29.03 ms`** | `1,742.62 ms` | **`60.0×`** | `1.23e-11` |
| `(8, 8,192)` | **`154.91 ms`** | `10,609.65 ms` | **`68.5×`** | `1.23e-10` |

## 5. Exact Bounded-Window Streaming (`RingMP` vs `stumpy.stumpi(egress=True)`, `m=64`)

| Window `W` | FlashMP Full (batch / py-loop) | FlashMP Causal | STUMPY `stumpi` | Speedup (py-loop / batch / causal) | FlashMP Egress Err | STUMPY Egress Err |
|---:|---:|---:|---:|---:|---:|---:|
| `1,024` | **`2.54` / `3.63 µs`** | **`1.27 µs`** | `454.5 µs` | **`125.2×` / `179.1×` / `358.5×`** | **`2.18e-13`** | `1.570e+00` *(stale)* |
| `4,096` | **`4.59` / `5.90 µs`** | **`3.01 µs`** | `498.3 µs` | **`84.5×` / `108.7×` / `165.7×`** | **`4.65e-13`** | `1.137e+00` *(stale)* |
| `16,384` | **`12.06` / `13.34 µs`** | **`7.80 µs`** | `712.0 µs` | **`53.4×` / `59.1×` / `91.2×`** | **`3.73e-13`** | `1.021e+00` *(stale)* |

## 6. Extreme Numerical Stability (`DC = 1e+10` & Transient `1e10` Spikes)

| Library | High-DC Max Abs Err (`DC=1e10`, `N=2,048`) | Post-`1e10`-Spike Exact Motif Err (True = `0.0`) |
|---|---:|---:|
| **FlashMP** | **`4.647e-06`** | **`2.032e-16`** |
| `pyscamp` (fp64) | `2.507e-04` | `nan` |
| `stumpy` | `2.188e-03` | `0.000e+00` |

## 7. Unsupervised Regime Segmentation (`N=4,000`, True Changepoint `2000`)

| Method | Time | Detected CP | Error (ticks) | Speedup |
|---|---:|---:|---:|---:|
| **FlashMP (MP + FLOSS)** | **`3.84 ms`** | `1976` | `24` | **`9924.3×`** |
| `ruptures` PELT (`rbf`) | `38076.47 ms` | `[50, 100, 150, 200, 250]` | — | `1.0×` |

## 8. Reversible Agentic Context Compression (20 Trials × `N=10,000`)

| Representation | Mean Tokens | Token Savings | Top-3 Shape Anomaly Recall | Regime Recall | Lossless Drill-Down Err |
|---|---:|---:|---:|---:|---:|
| Raw CSV/JSON Floats | `18,751` | `0.0%` | — | — | `0.0` |
| `tsdownsample` LTTB (matched tokens) | `~240` | `~99%` | `0.0%` | — | Lossy |
| `tsdownsample` M4 (matched tokens) | `~240` | `~99%` | `0.0%` | — | Lossy |
| **FlashMP `digest()`** (`33.5 ms`) | **`129`** | **`99.31%` (`145×`)** | **`100.0%`** | **`100.0%`** | **`0.0` (exact)** |

## 9. Multi-Agent Firehose Sifting (`SwarmMonitor`, 3.5× Loud Distractors, 16-Shuffle Commutativity)

| Channels `C` | Throughput (ch-ticks/s) | Latency / ch-tick | Fault Recall | Loud Distractor FPs | Brief Tokens | Order-Invariant (16 shuffles) |
|---:|---:|---:|---:|---:|---:|:---:|
| `64` | **`275,609`** | `3.63 µs` | **`100%`** (`4/4`) | **`0`** | `84` | `PASS` |
| `256` | **`245,908`** | `4.07 µs` | **`100%`** (`4/4`) | **`0`** | `105` | `PASS` |
| `1,024` | **`205,151`** | `4.87 µs` | **`100%`** (`4/4`) | **`0`** | `119` | `PASS` |

## 10. Five-Domain Real-World Telemetry Evaluation (`N=10,000` Each, `$3/MTok` API Economics)

| Domain | `m` | `digest()` | Raw → Digest Tokens | Token Savings | Cost / 10k Queries | Top-3 Discord (FlashMP / LTTB / M4) | Drill Err | RingMP (`W=1,024`) | Pre-Anom FPs |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| **ECG Arrhythmia (PQRST + PVC)** | `80` | `42.2 ms` | `18,185 → 117` | **`-99.36%`** | `$545.55 → $3.51` | **`Hit (#1)`** / `Miss` / `Miss` | `0.0` | `3.85 µs/tick` | `0` |
| **Cloud Microservice Telemetry** | `64` | `36.5 ms` | `20,000 → 125` | **`-99.38%`** | `$600.00 → $3.75` | **`Hit (#1)`** / `Miss` / `Miss` | `0.0` | `3.37 µs/tick` | `0` |
| **Industrial Rotating-Machinery Bearing** | `64` | `39.6 ms` | `18,741 → 118` | **`-99.37%`** | `$562.23 → $3.54` | **`Hit (#1)`** / `Miss` / `Miss` | `0.0` | `3.21 µs/tick` | `0` |
| **Financial Tick Microstructure (DC=1e5)** | `64` | `34.1 ms` | `28,884 → 133` | **`-99.54%`** | `$866.52 → $3.99` | **`Hit (#1)`** / `Miss` / `Miss` | `0.0` | `3.33 µs/tick` | `0` |
| **IoT Sensor (DC=1e10 + 1e10 Glitches)** | `64` | `50.5 ms` | `41,445 → 120` | **`-99.71%`** | `$1,243.35 → $3.60` | **`Hit (#1)`** / `Miss` / `Miss` | `0.0` | `3.32 µs/tick` | `0` |
| **Mean / Total (5 domains)** | — | **`40.6 ms`** | **`25,451 → 122.6`** | **`-99.47%` (`207×`)** | **`$763.53 → $3.68`** | **`5/5` (`100%`) / `0/5` / `0/5`** | **`0.0`** | **`3.42 µs/tick`** | **`0`** |

## Reproduce

```bash
python benchmarks/bench_vs_incumbents.py
python benchmarks/bench_agent_tokens.py
python benchmarks/bench_swarm_sift.py
python benchmarks/bench_realworld_domains.py
python benchmarks/render_report.py
```
