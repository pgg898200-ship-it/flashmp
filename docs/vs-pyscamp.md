# FlashMP vs pyscamp (SCAMP): CPU SIMD, Numerical Stability & Streaming Comparison

> **Headline (`2026-11-22`, x86-64 AVX2+FMA, 2 vCPU):** On CPU, **FlashMP `0.2.0`** runs exact float64-verified batch self-join **`3.22×–8.79×`** faster than `pyscamp` (SCAMP fp64) via provably-sound 48-wide float32-screened AVX2+FMA kernels, runs bidirectional AB-join **`3.25×`** faster (computing both directions vs `pyscamp`'s one direction), achieves **`54×`** higher accuracy at `DC=1e10`, recovers exact motifs (`2.03e-16`) after transient `1e10` spikes where `pyscamp` returns **`NaN`**, and adds exact bounded-window streaming (`RingMP`), multivariate `mSTUMP` (`68.5×` vs STUMPY), FLOSS regime segmentation, reversible `-99.3%` to `-99.7%` token digests, OpenAI/Anthropic/MCP toolkits, and `SwarmMonitor`.

---

## Verified Benchmark Summary (`2026-11-22`)

Measured live on matched 2-vCPU x86-64 AVX2+FMA silicon (`benchmarks/bench_vs_incumbents.py`):

| Benchmark | **FlashMP `0.2.0`** | `pyscamp` (fp64) | FlashMP Edge |
|---|---:|---:|---|
| Batch Self-Join (`N=4,096, m=64, 2T`) | **`5.30 ms`** | `17.08 ms` | **`3.22×`** faster |
| Batch Self-Join (`N=16,384, m=64, 2T`) | **`39.61 ms`** | `275.17 ms` | **`6.95×`** faster |
| Batch Self-Join (`N=32,768, m=64, 2T`) | **`87.33 ms`** | `767.63 ms` | **`8.79×`** faster (`3.69e-12` exact err) |
| AB-Join (`32,768 × 4,096, m=64, 2T`) | **`31.84 ms`** *(2-way)* | `103.45 ms` *(1-way)* | **`3.25×`** faster *(with 2× outputs)* |
| Cold import + first `N=4,096` self-join | `43.2 ms` | **`30.0 ms`** | `pyscamp` slightly faster on cold NumPy call; FlashMP bare `import` is `0.57 ms` |
| High-DC Stability (`DC=1e10, N=2,048`) | **`4.65e-6`** max err | `2.51e-4` max err | **`54×`** more accurate |
| Post-`1e10`-Spike Exact Motif (True = `0.0`) | **`2.03e-16`** | **`NaN`** | Spike-drop HWM re-anchoring prevents cancellation blowup |
| Multivariate `mSTUMP` | **`154.9 ms`** `(8, 8,192)` | Unsupported | `68.5×` faster than STUMPY |
| Bounded-Window Streaming (`RingMP`) | **`1.27–2.54 µs/tick`** (`W=1,024`) | Unsupported | Exact egress (`2.2e-13` err) + anytime-valid E-values |
| FLOSS Regimes / Digest / Swarm / AI Toolkits | **Included (0 deps)** | Unsupported | `-99.47%` 5-domain tokens, lossless drill-down, OpenAI/Anthropic/MCP |

---

## Why FlashMP Outperforms `pyscamp` on CPU & Under Extreme Outliers

1. **48-Wide Provably-Sound Float32-Screened Exact Float64 Engine:** On clean tiles (verified in $O(1)$ via per-block `StatBlock` scale-ratio bounds), FlashMP carries **48 consecutive diagonals** (`6 × __m256`) in float32 — re-anchored from exact float64 at tile tops — screening against provably sound lower-bounded float32 thresholds (`best - delta_tile`) with deferred exact float64 candidate verification (`< 4e-12` error). Any tile with drops, NaN/Inf/flats, extreme scale ratios, or `delta_tile > 0.025f` transparently executes the 16-wide exact float64 AVX2+FMA kernel.
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
