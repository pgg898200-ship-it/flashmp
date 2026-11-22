# FlashMP Architecture & Mathematical Guarantees

FlashMP is a zero-dependency C++20/SIMD engine (AVX2+FMA on x86-64, NEON on ARM64, portable scalar fallback) engineered for exact batch and bounded-window streaming Matrix Profile on CPU, paired with reversible time-series compaction (`digest()`) and commutative multi-agent monitoring (`SwarmMonitor`).

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

## 2. Microarchitecture: L1-Tiled 16-Diagonal AVX2+FMA & Two-Stage 8-Diagonal `mSTUMP`

On x86-64 CPUs with AVX2+FMA:
- **16-Diagonal Register-Blocked Batch Kernels:** `eval_strip_16_carry` carries 16 consecutive diagonals (`4 × __m256d` accumulators `c0..c3`) inside L1-resident tiles (`kTileH = 192`, `kTileW = 256`) with 512-byte staggered slabs (`mod 4096 B`) to eliminate 4K-aliasing stalls and L1d set conflicts. Sentinel-padded slabs (`kSeriesPad = 32`) eliminate scalar corner peel loops.
- **Combined-Mask Write Elision:** Row-max and column-max checks combine all 4 comparison masks via `_mm256_or_pd` into one `_mm256_movemask_pd` branch. In steady state, `>99.5%` of 16-cell strips execute zero L1 stores and zero index loads.
- **Single-Pass Full + Left + Right MP:** Every upper-triangle pair $(i, j)$ ($j > i + \text{ez}$) updates `right_mp[i]` and causal `left_mp[j]` simultaneously, emitting Full-MP, strictly causal Left-MP, Right-MP, and FLOSS arc curves in one pass.
- **Two-Stage Exact 8-Diagonal AVX2 `mSTUMP`:** Processes 8 diagonals simultaneously across all $D$ channels using packed B-tiles (`kMsBStride = 264`):
  1. *Fast Provable Lower-Bound Filter:* Computes a strictly sound `float32` lower bound on per-channel half-distance using `_mm256_rsqrt_ps` (`vrsqrtps`), sorts channels in-register via compile-time optimal 8-wide SIMD sorting networks (`D = 1..8, 12, 16`) or iterative Batcher odd-even merge sort (`D > 16`), and tests prefix sums against `upper_bound_f32(best_mp)`.
  2. *Cold Exact Float64 Verifier:* Only lanes whose lower bound beats the current `float32` upper bound invoke the cold noinline float64 verifier (`_mm256_sqrt_pd` + near-zero refinement), preserving `< 1e-14` double-precision exactness at ~40×+ speedup over `stumpy.mstump`.

---

## 3. RingMP: Exact $O(1)$-Amortized-Egress Bounded-Window Streaming MP

Standard streaming Matrix Profile (`stumpy.stumpi(egress=True)`) shifts arrays in Python/Numba (~400–700 µs/tick) and retains **stale expired neighbors** after egress (accumulating errors `> 1.0` vs. the true sliding-window Matrix Profile).

### Theorem (Exact Monotone Suffix-Max Left-Stack)
Fix any subsequence $k$ arriving at time $t = k$. Its candidate left neighbors lie in $[L_0, k - \text{ez} - 1]$ where $L_0 = \max(0, k - W + 1)$. After $k$ arrives, **no new left neighbor can ever appear for $k$**; old left neighbors only **expire monotonically from the left** as $L(t') = \max(0, t' - W + 1)$ advances by $+1$ per tick.

Scanning $j$ strictly right-to-left ($k - \text{ez} - 1$ down to $L_0$) in-place in the staggered covariance ring upon arrival and pushing only strictly increasing correlation record-holders $(j_1, c_1), \dots, (j_r, c_r)$ (with $j_1 > \dots > j_r$ and $c_1 < \dots < c_r$):
- At every future tick $t'$, the stack top with $j_{\text{top}} \ge L(t')$ is the **exact** Left-MP optimum over $[L(t'), k - \text{ez} - 1]$.
- Expiring $g_{\text{exp}} = g - W$ scans active `left_top_idx` in 8-wide AVX2 (`_mm256_cmpeq_epi64`) and pops in $O(1)$ whenever $j_{\text{top}} \le g_{\text{exp}}$.
- Expected stack depth is the harmonic number $H_W \approx \ln W$ (~8.9 entries at $W = 4{,}096$), stored inline with cap `kStackCap = 64` and exact fallback rebuild (`rebuild_left_stack_exact`) if an adversarial monotone stream ever exhausts all 64 slots.
- **Right-MP neighbors never expire** before $k$ itself expires (since $j > k \ge L(t')$).
- Combined with mirrored $2\times$ raw ring buffers, round-robin exact re-anchoring (1 lag per tick), and `active_drops` spike-exit re-anchoring, RingMP delivers `< 1e-11` exact sliding-window parity in bounded $O(W)$ memory with zero per-tick heap allocations.

---

## 4. Anytime-Valid Streaming Discord Alerts (Hölder-Tempered Exchangeable e-CUSUM)

Raw Matrix Profile distances lack calibrated false-alarm control, and naïve i.i.d. e-processes over-fire on $m$-overlapping windows due to strong lag-$m$ autocorrelation. On each tick $g \ge 2m$, RingMP scores the strictly causal Left-MP distance $d_g$ against a trailing $C = 256$ calibration ring delayed by $m$ ticks (so every calibration sample is strictly non-overlapping with the current window):
1. **Causal Conformal $p$-Value & Robust $z$-Score:** Computed in AVX2 over the non-overlapping calibration ring `cal_ring`.
2. **Exact Finite-Sample Exchangeable 4th-Power E-Value:** With $\bar{d}_{\text{all}} = \frac{1}{C+1}\big(d_g + \sum_{j=1}^C X_j\big)$ and $w(x) = \max(0, x - \bar{d}_{\text{all}})^4$,
   $$e_g = \frac{(C + 1)\, w(d_g)}{w(d_g) + \sum_{j=1}^C w(X_j)}, \qquad \mathbb{E}_{H_0}[e_g] = 1 \text{ identically under exchangeability.}$$
3. **$1/m$ Hölder-Tempered 3-Bet e-CUSUM Mixture:** Over three fixed betting fractions $\lambda_b \in \{0.10, 0.30, 0.60\}$,
   $$W_g^{(b)} = \max\!\big(1,\, W_{g-1}^{(b)}\big)\cdot \big(1 + \lambda_b (e_g - 1)\big)^{1/m}, \qquad E_g = \frac{1}{3}\sum_{b=1}^3 W_g^{(b)}.$$
   By Hölder's inequality across the $m$ overlapping subsequence phases, $E_g$ controls false alarms under stationary $m$-dependent streams while accumulating exponential evidence ($E_g \ge 1/\alpha$) immediately on sustained shape anomalies.

---

## 5. Reversible Agentic Digest & Commutative Multi-Agent `SwarmMonitor`

- **Reversible Digest (`digest()`):** Compresses $N$-sample univariate or multivariate telemetry into a fixed-point token-counted compact representation (~130–250 tokens, `>99%` savings on `N = 10,000`) preserving dominant period $m$, FLOSS semantic regimes, top motifs (with exact per-channel Pearson correlation), and top shape discords — while owning a private `array.array('d')` copy for bit-exact lossless `drilldown(start, end)`, `drilldown_motif(rank)`, and `drilldown_discord(rank)`.
- **Commutative `SwarmMonitor`:** Runs lazy causal `RingMP` + exchangeable e-CUSUM across up to thousands of concurrent channels (z-normalization makes it immune to $3.5\times$–$4\times$ high-amplitude normal-shape distractors) and fuses multi-agent alerts via a provably commutative, associative, idempotent total-order join-semilattice (`AlertLedger`), emitting strictly token-bounded briefs (`len(brief) <= max_tokens * 4`).
- **Sandboxed Zero-Dependency MCP Server (`flashmp mcp` / `flashmp-mcp`):** Native stdio JSON-RPC 2.0 server supporting both newline-delimited and multi-header HTTP/LSP framing (`Content-Length` / `Content-Type`), bounded 64-session LRU native cleanup, strict `allowed_dirs` path/symlink confinement (`FLASHMP_MCP_ALLOWED_DIRS`), and DoS resource ceilings.
