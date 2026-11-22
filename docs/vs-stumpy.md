# FlashMP vs STUMPY: Head-to-Head CPU & Streaming Comparison

> **Headline (`2026-11-22`, x86-64 AVX2+FMA, 2 vCPU):** FlashMP `0.1.0` is a drop-in replacement for `STUMPY 1.14.1` (`stump`, `mstump`, `stumpi`, `abjoin`) that eliminates STUMPY's **41.9-second** Numba JIT cold start (**`987×`** faster first call), runs batch self-join **`13.8×–18.4×`** faster, accelerates multivariate `mSTUMP` by **`48.6×`**, streams bounded windows **`19×–338×`** faster while fixing `stumpy.stumpi(egress=True)`'s **`1.02–1.57`** stale-neighbor error, and improves `DC=1e10` accuracy by **`471×`**.

---

## Verified Benchmark Summary (`2026-11-22`)

All numbers measured live on identical inputs, matched thread counts (`2T`), and exact parity checks (`benchmarks/bench_vs_incumbents.py`):

| Benchmark | **FlashMP `0.1.0`** | `STUMPY 1.14.1` | Speedup / Improvement | Max Abs Err |
|---|---:|---:|---:|---:|
| Cold `import` only | **`0.78 ms`** | — | **0 required deps** (vs ~109 MiB) | — |
| Cold import + first `N=4,096` self-join | **`42.5 ms`** | `41,906.1 ms` | **`987×`** faster | `7.63e-13` |
| Batch Self-Join (`N=4,096, m=64`) | **`3.29 ms`** | `45.51 ms` | **`13.81×`** faster | `7.63e-13` |
| Batch Self-Join (`N=16,384, m=64`) | **`38.47 ms`** | `565.04 ms` | **`14.69×`** faster | `2.96e-12` |
| Batch Self-Join (`N=32,768, m=64`) | **`119.47 ms`** | `2,199.61 ms` | **`18.41×`** faster | `3.67e-12` |
| AB-Join (`32,768 × 4,096, m=64`) | **`37.35 ms`** *(2-way)* | `481.19 ms` *(1-way)* | **`12.88×`** faster | `3.15e-12` |
| Multivariate `mSTUMP` (`D=4, N=4,096`) | **`39.14 ms`** | `1,888.58 ms` | **`48.25×`** faster | `1.23e-11` |
| Multivariate `mSTUMP` (`D=8, N=8,192`) | **`248.46 ms`** | `12,079.35 ms` | **`48.62×`** faster | `1.23e-10` |
| Streaming `W=1,024` (`egress=True`) | **`3.14 µs`** *(batch)* / **`7.77 µs`** *(py-loop)* / **`1.54 µs`** *(causal)* | `519.9 µs` | **`66.9×`–`337.7×`** faster | **`2.18e-13`** vs **`1.570`** |
| Streaming `W=4,096` (`egress=True`) | **`7.31 µs`** *(batch)* / **`12.00 µs`** *(py-loop)* / **`3.16 µs`** *(causal)* | `506.1 µs` | **`42.2×`–`160.3×`** faster | **`3.85e-13`** vs **`1.137`** |
| Streaming `W=16,384` (`egress=True`) | **`31.65 µs`** *(batch)* / **`41.70 µs`** *(py-loop)* / **`9.52 µs`** *(causal)* | `798.4 µs` | **`19.1×`–`83.9×`** faster | **`1.71e-12`** vs **`1.021`** |
| High-DC Stability (`DC=1e10, N=2,048`) | **`4.65e-6`** max err | `2.19e-3` max err | **`471×`** more accurate | — |

---

## Key Technical Differences

### 1. Exact Bounded-Window Streaming (`RingMP` vs `stumpy.stumpi`)
When `egress=True`, `stumpy.stumpi` drops the oldest subsequence from the front of the sliding window, but surviving subsequences whose nearest neighbor was that expired subsequence retain **stale distances and indices**. Across `W = 1,024..16,384`, this produces **`1.02–1.57`** maximum absolute error relative to the true sliding-window Matrix Profile.

FlashMP's `RingMP` solves egress mathematically via a **Monotone Suffix-Max Left-Stack** (expected depth $\ln W \approx 8.9$ at $W = 4{,}096$, capped at 64 with exact fallback rebuild). Expiring a subsequence pops invalidated left-neighbors in $O(1)$ amortized time using 8-wide AVX2 index scans, maintaining **`< 2e-12`** exactness on every tick while running **`19×–338×`** faster.

### 2. Zero Numba / LLVM Cold Start & Zero Mandatory Dependencies
`STUMPY` depends on NumPy, SciPy, Numba, and llvmlite (~109 MiB) and compiles LLVM IR on first call (`41.9 s` in a fresh container/serverless process). FlashMP ships a precompiled C++20 shared library with `0` required Python dependencies: `import flashmp` completes in **`0.78 ms`** with `numpy` never loaded unless the caller passes NumPy arrays.

### 3. Drop-In Migration (1-Line Swap)
```python
# Before
import stumpy

# After
import flashmp as stumpy

mp = stumpy.stump(x, m=64)             # [P, I, left_I, right_I], shape (n, 4)
P, I = stumpy.mstump(X, m=64)          # shape (D, n), matches stumpy.mstump
stream = stumpy.stumpi(T, m=64, egress=True)
stream.update(new_tick)                # Exact egress, .P_, .I_, .left_P_, .left_I_
```

---

## Honest Limitations: When to Choose STUMPY Instead

Choose **STUMPY** if you need:
- Distributed multi-node cluster execution via Dask (`stumpy.stumped`) or Ray.
- Specialized STUMPY routines not yet in FlashMP: `ostinato` (consensus motifs), `mpdist`, `pan` matrix profiles, `mmotifs`, `snippets`, or `aamp` (non-z-normalized Euclidean MP).
- STUMPY's flat-window convention in univariate `stump` (`STUMPY` returns distance `0.0` between two constant subsequences; FlashMP intentionally masks constant univariate subsequences to `+inf` to avoid false-positive motifs/leaks on flat telemetry, while matching `STUMPY` in `mstump`).

---

## Reproduce

```bash
python benchmarks/bench_vs_incumbents.py
```
