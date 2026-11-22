# FlashMP vs STUMPY: Head-to-Head CPU & Streaming Comparison

> **Headline (`2026-11-22`, x86-64 AVX2+FMA, 2 vCPU):** FlashMP `0.2.0` is a drop-in replacement for `STUMPY 1.14.1` (`stump`, `mstump`, `stumpi`, `abjoin`) that eliminates STUMPY's **38.0-second** Numba JIT cold start (**`880×`** faster first call; **`0.57 ms`** bare import), runs batch self-join **`11.3×–27.2×`** faster, accelerates multivariate `mSTUMP` by **`60.0×–68.5×`**, streams bounded windows **`53×–358×`** faster (including **`53×–125×`** in a plain Python `for` loop via the zero-copy `_fast` C-API extension) while fixing `stumpy.stumpi(egress=True)`'s **`1.02–1.57`** stale-neighbor error, and improves `DC=1e10` accuracy by **`471×`**.

---

## Verified Benchmark Summary (`2026-11-22`)

All numbers measured live on identical inputs, matched thread counts (`2T`), and exact parity checks (`benchmarks/bench_vs_incumbents.py`):

| Benchmark | **FlashMP `0.2.0`** | `STUMPY 1.14.1` | Speedup / Improvement | Max Abs Err |
|---|---:|---:|---:|---:|
| Cold `import` only | **`0.57 ms`** | — | **0 required deps** (vs ~109 MiB) | — |
| Cold import + first `N=4,096` self-join | **`43.2 ms`** | `38,025.1 ms` | **`880×`** faster | `9.32e-13` |
| Batch Self-Join (`N=4,096, m=64`) | **`5.30 ms`** | `60.01 ms` | **`11.32×`** faster | `9.32e-13` |
| Batch Self-Join (`N=16,384, m=64`) | **`39.61 ms`** | `576.14 ms` | **`14.54×`** faster | `2.95e-12` |
| Batch Self-Join (`N=32,768, m=64`) | **`87.33 ms`** | `2,377.48 ms` | **`27.22×`** faster | `3.69e-12` |
| AB-Join (`32,768 × 4,096, m=64`) | **`31.84 ms`** *(2-way)* | `502.80 ms` *(1-way)* | **`15.79×`** faster | `3.09e-12` |
| Multivariate `mSTUMP` (`D=4, N=4,096`) | **`29.03 ms`** | `1,742.62 ms` | **`60.02×`** faster | `1.23e-11` |
| Multivariate `mSTUMP` (`D=8, N=8,192`) | **`154.91 ms`** | `10,609.65 ms` | **`68.49×`** faster | `1.23e-10` |
| Streaming `W=1,024` (`egress=True`) | **`2.54 µs`** *(batch)* / **`3.63 µs`** *(py-loop)* / **`1.27 µs`** *(causal)* | `454.5 µs` | **`125.2×`–`358.5×`** faster | **`2.18e-13`** vs **`1.570`** |
| Streaming `W=4,096` (`egress=True`) | **`4.59 µs`** *(batch)* / **`5.90 µs`** *(py-loop)* / **`3.01 µs`** *(causal)* | `498.3 µs` | **`84.5×`–`165.7×`** faster | **`4.65e-13`** vs **`1.137`** |
| Streaming `W=16,384` (`egress=True`) | **`12.06 µs`** *(batch)* / **`13.34 µs`** *(py-loop)* / **`7.80 µs`** *(causal)* | `712.0 µs` | **`53.4×`–`91.2×`** faster | **`3.73e-13`** vs **`1.021`** |
| High-DC Stability (`DC=1e10, N=2,048`) | **`4.65e-6`** max err | `2.19e-3` max err | **`471×`** more accurate | — |

---

## Key Technical Differences

### 1. Exact Bounded-Window Streaming (`RingMP` vs `stumpy.stumpi`)
When `egress=True`, `stumpy.stumpi` drops the oldest subsequence from the front of the sliding window, but surviving subsequences whose nearest neighbor was that expired subsequence retain **stale distances and indices**. Across `W = 1,024..16,384`, this produces **`1.02–1.57`** maximum absolute error relative to the true sliding-window Matrix Profile.

FlashMP's `RingMP` solves egress mathematically via a **Monotone Suffix-Max Left-Stack** + **$O(1)$-amortized intrusive expiration buckets** + **Hybrid 32-wide Mature-Float32 screening** (for L3-bound windows `W >= 6,144`) + zero-copy PEP 3118 `_fast` C-API dispatch, maintaining **`< 5e-13`** exactness on every tick while running **`53×–358×`** faster.

### 2. Zero Numba / LLVM Cold Start & Zero Mandatory Dependencies
`STUMPY` depends on NumPy, SciPy, Numba, and llvmlite (~109 MiB) and compiles LLVM IR on first call (`38.0 s` in a fresh container/serverless process). FlashMP ships prebuilt C++20 native binaries with `0` required Python dependencies: `import flashmp` completes in **`0.57 ms`** with `numpy` never loaded unless the caller passes NumPy arrays, while seamlessly duck-typing Pandas and Polars Series/DataFrames without importing them.

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
