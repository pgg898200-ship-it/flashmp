# FlashMP Architecture & Mathematical Guarantees

FlashMP is a zero-dependency C++20/SIMD engine (AVX2+FMA on x86-64, NEON on ARM64, portable scalar fallback) paired with a zero-copy PEP 3118 CPython C-API accelerator (`_fast`), engineered for exact batch and bounded-window streaming Matrix Profile on CPU, reversible time-series compaction (`digest()`), and commutative multi-agent monitoring (`SwarmMonitor`).

---

## 1. Exact DC-, Scale-, and Spike-Invariant Centered Recurrence

For a time series $T \in \mathbb{R}^N$ and subsequence length $m \ge 3$, let $n = N - m + 1$.

Conventional Matrix Profile implementations (`STUMPY`, naive `STOMP`) maintain raw dot products $QT_{i,j} = \sum_{k=0}^{m-1} T_{i+k} T_{j+k}$ and compute Pearson correlation via:
$$r_{i,j} = \frac{QT_{i,j} - m \mu_i \mu_j}{m \sigma_i \sigma_j}$$

This formulation fails in three common production regimes:
1. **High DC offsets** (e.g. $10^{10}$ timestamps/telemetry): $QT_{i,j}$ and $m\mu_i\mu_j$ suffer catastrophic subtractive cancellation.
2. **Extreme uniform scales** ($10^{-160}$ or $10^{160}$): squaring raw centered deviations underflows to `0` or overflows to `+inf`.
3. **Transient giant spikes** (e.g. two $10^{10}$ sensor glitches in unit-scale data): carried recurrences briefly surge to $O(10^{20})$ and leave $O(10^4)$ rounding residue after the spike exits, corrupting downstream distances across thousands of diagonals.

### FlashMP's Four-Layer Numerical Defense
1. **Overflow-Safe Global Anchor Shift & Extreme-Scale Normalization:** FlashMP shifts finite values by the first finite sample $b$ (verifying `isfinite(T[i] - b)` across all finite samples; falling back to $b = 0$ on opposing $\pm 10^{308}$ extremes) and applies exact positive rescaling whenever all finite values are uniformly tiny ($< 10^{-100}$) or predominantly huge ($> 10^{100}$).
2. **Neumaier-Compensated Centered Variance + Exact Re-Anchor:** Window means $\mu_i$ and centered sums of squares $S_i = \sum_{k=0}^{m-1}(u_{i+k} - \mu_i)^2$ use compensated rolling updates re-anchored every 64 steps and immediately whenever `run_var <= peak_var * 1e-4` or non-finite, paired with a scale-free flatness guard.
3. **Exact Centered Covariance Recurrence (SCAMP Identity) + Spike-Drop HWM Re-Anchoring:** Defining
   $$df_i = \frac{u_{i+m} - u_i}{2}, \qquad dg_i = (u_{i+m} - \mu_{i+1}) + (u_i - \mu_i),$$
   the centered inner product $\operatorname{cov}_{i,j} = \sum_{k=0}^{m-1}(u_{i+k}-\mu_i)(u_{j+k}-\mu_j)$ satisfies the exact offset-invariant identity:
   $$\operatorname{cov}_{i+1, j+1} = \operatorname{cov}_{i, j} + df_i \cdot dg_j + dg_i \cdot df_j.$$
   Along carried diagonals (`kCovReanchor = 4096`), FlashMP tracks a per-series running norm high-water mark (`norm_hwm`). Whenever a valid subsequence norm drops by $> 32\times$ from `norm_hwm` (or recovers from a NaN/Inf/flat segment), it marks `is_drop[i] = 1` and resets `norm_hwm`. An $O(1)$ prefix-sum check (`drop_prefix`) ensures **zero hot-path overhead on clean tiles**, while any tile crossing a drop triggers exact covariance re-anchoring so $\max |\operatorname{cov}_{i',j'}| / (\|A_i\|_2 \|B_j\|_2) \le 32^2 = 1024$ everywhere — bounding floating-point drift to $< 10^{-11}$ even across $10^{10}$ transient spikes.
4. **Branchless NaN/Flat Masking & Near-Zero Distance Refinement:** Invalid/flat subsequences receive `inv_norm = quiet_NaN()`, so ordered IEEE-754 comparisons (`_CMP_GT_OQ`) reject them branchlessly. Any winning correlation $r > 1 - 10^{-10}$ is recomputed directly via centered unit-vector Euclidean difference $\sqrt{\sum (\hat{a}_k - \hat{b}_k)^2}$, eliminating $\sqrt{1 - r}$ cancellation on verbatim/affine copies (`< 1e-14` error).

---

## 2. Microarchitecture: 48-Wide Provably-Sound Float32-Screened Exact Float64 Engine

On x86-64 CPUs with AVX2+FMA (`v0.2.0`):
- **48-Diagonal Float32-Screened Batch Kernels (`fmp_self_join`, `fmp_ab_join`, `fmp_leak_scan`):** `SeriesPrep` precomputes float32 copies (`inv_f`, `df_f`, `dg_f`) alongside 32-wide block `StatBlock` envelopes (`min_norm`, `max_norm`, `max_df`, `max_dg`, `bad`). Before scanning any tile (`kTileH = 192`, `kTileW = 288`), FlashMP computes a **strict, provably sound IEEE-754 float32 rounding error bound** $\delta_{\text{tile}}$ across all carried FMA updates in the tile:
  - *Clean tiles* ($\delta_{\text{tile}} \le 0.025$ and zero drops/NaNs/flats/extreme ratios) execute a **48-wide (`6 × __m256`)** float32 carried covariance strip re-anchored from exact float64 at tile tops, screening each lane against the strictly sound lower-bounded threshold $\text{best} - \delta_{\text{tile}}$. Because $|\tilde{r}_{i,j} - r_{i,j}| \le \delta_{\text{tile}}$ is guaranteed, **no true exact optimum can ever be pruned**. Passing candidates are verified in exact float64 (`exact_cov_scalar` + parallel `merge_and_verify_f32_states`), achieving **`8.79×`** vs `pyscamp` and **`27.22×`** vs `STUMPY` at **`< 4e-12`** exact double-precision parity!
  - *Adversarial tiles* (spikes, drops, NaNs, flats, extreme scale ratios, or $\delta_{\text{tile}} > 0.025$) transparently execute the **16-wide (`4 × __m256d`) exact float64** AVX2+FMA kernel (`eval_strip_16_carry`) with 512-byte staggered slabs (`mod 4096 B`) and combined-mask write elision.
- **L1d-Resident 8-Diagonal Float32-Screened `mSTUMP` (`68.5×` vs STUMPY):** `eval_mstump_strip_8_f32_avx2<FixedD>` evaluates 8 diagonals across all $D$ channels using pointer-incremented B streams (full Haswell load-op micro-fusion) and non-negative IEEE-754 int32-domain `vpminsd`/`vpmaxsd`/`vpcmpgtd` thresholding against L2-resident AoS `mp_hi_f` packed via AVX2 `8×8` / SSE `4×4` transposes into `col_mp_tile` (`21.5 KiB`, L1d-resident), paired with in-register AVX2 8-element / SSE 4-element bitonic sorting networks (`sort8_nonneg_f32_avx2` + `prefix_sum8_f32_avx2`), shared 1-band warm-start, and exact float64 verification (`< 1e-13` exact ref parity).
- **Zero-Copy PEP 3118 CPython C-API Accelerator (`_fast`):** Links dynamically to `libflashmp` using only `<Python.h>` and `flashmp.hpp` (zero NumPy C-API dependency), acquires contiguous aligned buffers directly via `PyObject_GetBuffer`, and releases the Python GIL (`Py_BEGIN_ALLOW_THREADS`) across all batch, streaming, leakage, mSTUMP, and FLOSS calls — eliminating `ctypes` overhead (`13.8×` lower wrapper cost) and scaling multi-threaded Python callers cleanly across CPU cores.

---

## 3. RingMP: Exact $O(1)$-Amortized-Egress Bounded-Window Streaming MP

Standard streaming Matrix Profile (`stumpy.stumpi(egress=True)`) shifts arrays in Python/Numba (~450–712 µs/tick) and retains **stale expired neighbors** after egress (accumulating errors `1.02–1.57` vs. the true sliding-window Matrix Profile).

### Theorem (Exact Monotone Suffix-Max Left-Stack + $O(1)$ Expiration Buckets + Hybrid Mature-F32)
Fix any subsequence $k$ arriving at time $t = k$. Its candidate left neighbors lie in $[L_0, k - \text{ez} - 1]$ where $L_0 = \max(0, k - W + 1)$. After $k$ arrives, **no new left neighbor can ever appear for $k$**; old left neighbors only **expire monotonically from the left** as $L(t') = \max(0, t' - W + 1)$ advances by $+1$ per tick.

1. **Monotone Suffix-Max Stack:** Scanning $j$ strictly right-to-left ($k - \text{ez} - 1$ down to $L_0$) in-place in the staggered covariance ring upon arrival and pushing only strictly increasing correlation record-holders $(j_1, c_1), \dots, (j_r, c_r)$ guarantees that at every future tick $t'$, the stack top with $j_{\text{top}} \ge L(t')$ is the **exact** Left-MP optimum over $[L(t'), k - \text{ez} - 1]$. Expected stack depth is $H_W \approx \ln W$ (~8.9 at $W = 4{,}096$), stored inline (`kStackCap = 64`) with exact fallback rebuild (`rebuild_left_stack_exact`).
2. **$O(1)$-Amortized Intrusive Expiration Buckets (`v0.2.0`):** Each surviving slot is linked into the intrusive expiration list (`exp_head[j_top & mask]`, `exp_next`) keyed by its current top neighbor `j_top`. Expiring $g_{\text{exp}} = g - W$ traverses ONLY the slots whose top neighbor actually equals $g_{\text{exp}}$ (expected count $\approx 1$ per tick!), eliminating the $O(W)$ egress scan completely.
3. **Hybrid 16-Wide Float64 + 32-Wide Mature-F32 Segment Kernel (`v0.2.0`):** L1/L2-resident windows (`W < 6,144`) and young lags (`<= 512`) run a 16-wide (`4 × __m256d`) float64 kernel, while L3-bound mature windows (`W >= 6,144`) run a provably-sound 32-wide (`4 × __m256`) float32-screened kernel (`process_segment_rtl_f32`) with block-level `StreamStatBlk` bounds, deferred exact float64 candidate verification, and automatic float64 fallback on spikes/NaNs/flats/extreme scale — cutting `W = 16,384` full-MP streaming from `31.65 µs` to **`12.06 µs/tick`** (**`59.1×`** vs `stumpy.stumpi`) with **`3.73e-13`** exact egress parity!

---

## 4. Anytime-Valid Streaming Discord Alerts (Hölder-Tempered Exchangeable e-CUSUM)

Raw Matrix Profile distances lack calibrated false-alarm control, and naïve i.i.d. e-processes over-fire on $m$-overlapping windows due to strong lag-$m$ autocorrelation. On each tick $g \ge 2m$, RingMP scores the strictly causal Left-MP distance $d_g$ against a trailing $C = 256$ calibration ring delayed by $m$ ticks (strictly non-overlapping with the current window) via a single-pass fused AVX2 `update_eprocess` kernel with exact-reanchored running `cal_sum`:
1. **Causal Conformal $p$-Value & Robust $z$-Score:** Computed in AVX2 over the non-overlapping calibration ring `cal_ring`.
2. **Exact Finite-Sample Exchangeable 4th-Power E-Value:** With $\bar{d}_{\text{all}} = \frac{1}{C+1}\big(d_g + \sum_{j=1}^C X_j\big)$ and $w(x) = \max(0, x - \bar{d}_{\text{all}})^4$,
   $$e_g = \frac{(C + 1)\, w(d_g)}{w(d_g) + \sum_{j=1}^C w(X_j)}, \qquad \mathbb{E}_{H_0}[e_g] = 1 \text{ identically under exchangeability.}$$
3. **$1/m$ Hölder-Tempered 3-Bet e-CUSUM Mixture:** Over three fixed betting fractions $\lambda_b \in \{0.10, 0.30, 0.60\}$,
   $$W_g^{(b)} = \max\!\big(1,\, W_{g-1}^{(b)}\big)\cdot \big(1 + \lambda_b (e_g - 1)\big)^{1/m}, \qquad E_g = \frac{1}{3}\sum_{b=1}^3 W_g^{(b)}.$$
   By Hölder's inequality across the $m$ overlapping subsequence phases, $E_g$ controls false alarms under stationary $m$-dependent streams while accumulating exponential evidence ($E_g \ge 1/\alpha$) immediately on sustained shape anomalies (`5/5` alerts triggered with `0` pre-anomaly false alerts across all 5 real-world telemetry domains).

---

## 5. Reversible Agentic Digest, Commutative `SwarmMonitor` & AI Agent Ecosystem

- **Reversible Digest (`digest()`):** Compresses $N$-sample univariate or multivariate telemetry (NumPy, lists, `array.array`, dicts, or duck-typed Pandas/Polars DataFrames & Series) into a fixed-point token-counted compact representation (~117–133 tokens, **`-99.31%` to `-99.71%`** savings on `N = 10,000`) preserving dominant period $m$, FLOSS semantic regimes, top motifs (with exact per-channel Pearson correlation), and top shape discords — while owning a private `array.array('d')` copy for bit-exact lossless `drilldown(start, end)`, `drilldown_motif(rank)`, and `drilldown_discord(rank)` (`max_abs_err = 0.0`).
- **Commutative `SwarmMonitor`:** Runs lazy causal `RingMP` + exchangeable e-CUSUM via `_fast` across up to thousands of concurrent channels (`205k–276k ch-ticks/s`, immune to $3.5\times$ high-amplitude normal-shape distractors) and fuses multi-agent alerts via a provably commutative, associative, idempotent total-order join-semilattice (`AlertLedger`), emitting strictly token-bounded briefs (`len(brief) <= max_tokens * 4`).
- **Zero-SDK AI Integrations & Sandboxed MCP Server:** `openai_tools()`, `anthropic_tools()`, `AgentToolkit`, `mcp_client_config()`, `install_skill()`, and the native stdio JSON-RPC 2.0 MCP server (`flashmp-mcp`) all share the same bounded 64-session LRU engine, multi-header LSP framing, DoS ceilings, and strict `allowed_dirs` path/symlink confinement (`FLASHMP_MCP_ALLOWED_DIRS`).
