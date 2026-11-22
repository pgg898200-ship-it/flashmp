# FlashMP vs pyscamp (SCAMP): CPU SIMD, Numerical Stability & Streaming Comparison

> **Headline (`2026-11-22`, x86-64 AVX2+FMA, 2 vCPU):** On CPU, **FlashMP `0.1.0`** runs exact float64 batch self-join **`5.05×–6.22×`** faster than `pyscamp` (SCAMP `4.0.3`), runs bidirectional AB-join **`2.71×`** faster (computing both directions vs `pyscamp`'s one direction), achieves **`54×`** higher accuracy at `DC=1e10`, recovers exact motifs (`2.03e-16`) after transient `1e10` spikes where `pyscamp` returns **`NaN`**, and adds exact bounded-window streaming (`RingMP`), multivariate `mSTUMP`, FLOSS regime segmentation, reversible `-99.3%` token digests, `SwarmMonitor`, and a zero-dependency MCP server.

---

## Verified Benchmark Summary (`2026-11-22`)

Measured live on matched 2-vCPU x86-64 AVX2+FMA silicon (`benchmarks/bench_vs_incumbents.py`):

| Benchmark | **FlashMP `0.1.0`** | `pyscamp` (`4.0.3`, fp64) | FlashMP Edge |
|---|---:|---:|---|
| Batch Self-Join (`N=4,096, m=64, 2T`) | **`3.29 ms`** | `18.01 ms` | **`5.47×`** faster |
| Batch Self-Join (`N=16,384, m=64, 2T`) | **`38.47 ms`** | `194.21 ms` | **`5.05×`** faster |
| Batch Self-Join (`N=32,768, m=64, 2T`) | **`119.47 ms`** | `743.50 ms` | **`6.22×`** faster |
| AB-Join (`32,768 × 4,096, m=64, 2T`) | **`37.35 ms`** *(2-way)* | `101.38 ms` *(1-way)* | **`2.71×`** faster *(with 2× outputs)* |
| Cold import + first `N=4,096` self-join | `42.5 ms` | **`33.2 ms`** | `pyscamp` slightly faster on cold NumPy call; FlashMP bare `import` is `0.78 ms` |
| High-DC Stability (`DC=1e10, N=2,048`) | **`4.65e-6`** max err | `2.51e-4` max err | **`54×`** more accurate |
| Post-`1e10`-Spike Exact Motif (True = `0.0`) | **`2.03e-16`** | **`NaN`** | Spike-drop HWM re-anchoring prevents cancellation blowup |
| Multivariate `mSTUMP` | **`248.5 ms`** `(8, 8,192)` | Unsupported | `48.6×` faster than STUMPY |
| Bounded-Window Streaming (`RingMP`) | **`1.54–3.14 µs/tick`** | Unsupported | Exact egress (`2.2e-13` err) + anytime-valid E-values |
| FLOSS Regimes / Digest / Swarm / MCP | **Included (0 deps)** | Unsupported | `-99.31%` tokens, lossless drill-down, native stdio MCP |

---

## Why FlashMP Outperforms `pyscamp` on CPU & Under Extreme Outliers

1. **16-Diagonal Register Blocking + 512B Slab Staggering:** FlashMP packs 16 consecutive diagonals (`4 × __m256d`) inside L1-resident tiles (`192 × 256`) with 512-byte staggered slabs (`mod 4096 B`) to avoid 4K-aliasing stalls, plus combined-mask write elision (zero L1 stores on `>99.5%` of strips).
2. **Anchor-Shifted + Spike-Drop HWM Recurrence:** Both `pyscamp` and FlashMP use the SCAMP centered `df`/`dg` recurrence. However, `pyscamp` does not perform global DC anchor removal or norm-drop re-anchoring along carried diagonals:
   - At `DC = 1e10`, FlashMP's anchor shift cuts error by **`54×`** (`4.65e-6` vs `2.51e-4`).
   - When two transient `1e10` spikes appear in unit-scale telemetry, carried SCAMP covariance surges to $O(10^{20})$ and leaves $O(10^4)$ cancellation garbage after the spike exits — causing `pyscamp` (`precision="double"`) to output **`NaN`** on an exact downstream motif, whereas FlashMP's $O(1)$-prefix-guarded spike-drop re-anchoring recovers **`2.03e-16`** with zero slowdown on clean data.
3. **Exclusion-Zone Convention Note:** `pyscamp` excludes pairs with `|i - j| < m // 4` (`ez = m // 4 - 1`, i.e., `15` for `m = 64`), whereas `STUMPY` and FlashMP default to `|i - j| <= ceil(m / 4)` (`ez = 16` for `m = 64`). Passing `excl_zone=m // 4 - 1` in FlashMP matches `pyscamp(precision="double")` to `1.1e-6–1.7e-6`.

---

## Honest Limitations: When to Choose pyscamp Instead

Choose **`pyscamp` / SCAMP** if you need:
- **CUDA GPU acceleration** on massive single-series batch joins ($N > 10^6$ to $10^8$) on dedicated NVIDIA GPUs. FlashMP is currently CPU-only (x86-64 AVX2+FMA, ARM64-NEON, scalar C++20).
- Distributed SCAMP server/worker batch orchestration across GPU clusters, or `pyscamp`'s `knn` / `sum` / `matrix` summary profiles.

---

## Reproduce

```bash
python benchmarks/bench_vs_incumbents.py
```
