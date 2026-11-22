#include "flashmp.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define FMP_HAS_AVX2 1
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define FMP_HAS_NEON 1
#endif

namespace {

constexpr double kNegInf = -2.0;
constexpr double kQuietNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPosInf = std::numeric_limits<double>::infinity();

// L1-resident tile dimensions (empirically tuned for 32 KiB L1d on AVX2+FMA)
constexpr int64_t kTileH = 192;
constexpr int64_t kTileW = 256;
constexpr int64_t kStripW = 16;
constexpr int64_t kSeriesPad = 32;    // Left/right sentinel pad (>= kStripW)
constexpr int64_t kCovReanchor = 4096; // Exact covariance re-anchor interval along carried diagonals
constexpr int64_t kReanchorStats = 64;

inline double corr_to_dist(double corr, int64_t m) {
    if (!(corr > -1.5)) {
        return kPosInf;
    }
    double c = corr;
    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;
    double d2 = 2.0 * static_cast<double>(m) * (1.0 - c);
    return d2 > 0.0 ? std::sqrt(d2) : 0.0;
}

inline int resolve_threads(int n_threads) {
    unsigned hw_u = std::thread::hardware_concurrency();
    int hw = hw_u > 0 ? static_cast<int>(hw_u) : 1;
    int max_t = std::min(256, std::max(4, hw * 4));
    if (n_threads > 0) return std::min(n_threads, max_t);
    return std::min(hw, max_t);
}

struct SeriesPrep {
    int64_t N = 0;
    int64_t m = 0;
    int64_t n = 0;       // N - m + 1
    int64_t stride = 0;  // ((n + 2*kSeriesPad + 511) & ~511) + 64 -> 512B stagger mod 4096B
    double base = 0.0;
    double scale = 1.0;
    bool has_any_flat = false;
    bool has_any_invalid = false;
    bool has_drops = false;
    std::vector<double> u_buf;     // length N + 2*kSeriesPad, DC-shifted, scaled & NaN-sanitized
    std::vector<double> slab;      // 4 * stride: [inv_norm | df | dg | mu], each padded by kSeriesPad
    std::vector<uint8_t> valid;    // length n, 1 iff NaN-free and non-flat
    std::vector<uint8_t> is_nan;   // length n, 1 iff window contains non-finite value
    std::vector<uint8_t> is_flat;  // length n, 1 iff NaN-free and flat/constant
    std::vector<uint8_t> is_drop;  // length n, 1 iff norm dropped >32x from running HWM (spike exit)
    std::vector<int64_t> drop_prefix; // length n + 1, prefix sum of is_drop

    inline double* u_ptr()   { return u_buf.data() + kSeriesPad; }
    inline double* inv_ptr() { return slab.data() + 0 * stride + kSeriesPad; }
    inline double* df_ptr()  { return slab.data() + 1 * stride + kSeriesPad; }
    inline double* dg_ptr()  { return slab.data() + 2 * stride + kSeriesPad; }
    inline double* mu_ptr()  { return slab.data() + 3 * stride + kSeriesPad; }
    inline const double* u_ptr()   const { return u_buf.data() + kSeriesPad; }
    inline const double* inv_ptr() const { return slab.data() + 0 * stride + kSeriesPad; }
    inline const double* df_ptr()  const { return slab.data() + 1 * stride + kSeriesPad; }
    inline const double* dg_ptr()  const { return slab.data() + 2 * stride + kSeriesPad; }
    inline const double* mu_ptr()  const { return slab.data() + 3 * stride + kSeriesPad; }
};

// Exact DC-shifted, scale-normalized, re-anchored centered statistics
// (immune to DC=1e12, extreme 1e-160/1e160/1e308 scales, and transient 1e12 spikes).
SeriesPrep precompute_series(const double* T, int64_t N, int64_t m) {
    SeriesPrep p;
    p.N = N;
    p.m = m;
    p.n = N - m + 1;
    p.stride = ((std::max<int64_t>(0, p.n) + 2 * kSeriesPad + 511) & ~511LL) + 64;
    p.u_buf.assign(static_cast<size_t>(std::max<int64_t>(0, N) + 2 * kSeriesPad), 0.0);
    p.slab.assign(static_cast<size_t>(4 * p.stride), 0.0);
    // Entire inv_norm plane (including left/right kSeriesPad sentinel zones) starts as NaN
    std::fill_n(p.slab.data(), p.stride, kQuietNaN);
    if (p.n > 0) {
        p.valid.resize(static_cast<size_t>(p.n), 0);
        p.is_nan.resize(static_cast<size_t>(p.n), 1);
        p.is_flat.resize(static_cast<size_t>(p.n), 0);
        p.is_drop.resize(static_cast<size_t>(p.n), 0);
        p.drop_prefix.resize(static_cast<size_t>(p.n + 1), 0);
    } else {
        p.drop_prefix.resize(1, 0);
    }

    double* u_w = p.u_ptr();
    double* mu_a = p.mu_ptr();
    double* inv_a = p.inv_ptr();
    double* df_a = p.df_ptr();
    double* dg_a = p.dg_ptr();

    // Find first finite sample as global DC anchor (eliminates huge-DC cancellation).
    double anchor = 0.0;
    bool has_finite = false;
    for (int64_t i = 0; i < N; ++i) {
        if (std::isfinite(T[i])) {
            anchor = T[i];
            has_finite = true;
            break;
        }
    }

    // Verify T[i] - anchor never overflows to ±Inf on extreme opposing signs (e.g. +1e308, -1e308).
    if (has_finite && anchor != 0.0) {
        for (int64_t i = 0; i < N; ++i) {
            if (std::isfinite(T[i]) && !std::isfinite(T[i] - anchor)) {
                anchor = 0.0;
                break;
            }
        }
    }
    p.base = anchor;

    std::vector<int64_t> bad_prefix(static_cast<size_t>(N + 1), 0);
    double max_abs_u = 0.0;
    int64_t n_finite_u = 0;
    int64_t n_huge_u = 0;
    for (int64_t i = 0; i < N; ++i) {
        if (std::isfinite(T[i])) {
            double v = T[i] - anchor;
            u_w[i] = v;
            double av = std::abs(v);
            if (av > max_abs_u) max_abs_u = av;
            if (av > 1e50) ++n_huge_u;
            ++n_finite_u;
            bad_prefix[static_cast<size_t>(i + 1)] = bad_prefix[static_cast<size_t>(i)];
        } else {
            u_w[i] = 0.0;
            bad_prefix[static_cast<size_t>(i + 1)] = bad_prefix[static_cast<size_t>(i)] + 1;
        }
    }

    // Exact positive rescaling when the series magnitude is uniformly tiny (<1e-100) or
    // predominantly huge (>1e100 with majority >1e50), avoiding crushing O(1) tails when only
    // isolated ±1e308 outliers are present.
    double scale = 1.0;
    if (has_finite &&
        ((max_abs_u > 0.0 && max_abs_u < 1e-100) ||
         (max_abs_u > 1e100 && n_huge_u * 2 >= n_finite_u))) {
        scale = max_abs_u;
        const double inv_scale = 1.0 / scale;
        for (int64_t i = 0; i < N; ++i) {
            u_w[i] *= inv_scale;
        }
    }
    p.scale = scale;

    if (N > 0) {
        double u0 = u_w[0];
        for (int64_t k = -kSeriesPad; k < 0; ++k) {
            u_w[k] = u0;
        }
    }

    if (!has_finite || p.n <= 0) {
        p.has_any_invalid = (p.n > 0);
        return p;
    }

    const double inv_m = 1.0 / static_cast<double>(m);
    const double* u = p.u_ptr();

    // Rolling Neumaier-compensated mean + periodic exact re-anchor every kReanchorStats.
    double run_sum = 0.0;
    double run_comp = 0.0;
    double run_var = 0.0;

    auto recompute_exact = [&](int64_t i) {
        // Two-pass exact local-shifted mean & centered sum of squares for subsequence i.
        double s = 0.0;
        double c = 0.0;
        for (int64_t k = 0; k < m; ++k) {
            double v = u[i + k];
            double t = s + v;
            if (std::abs(s) >= std::abs(v)) {
                c += (s - t) + v;
            } else {
                c += (v - t) + s;
            }
            s = t;
        }
        run_sum = s;
        run_comp = c;
        double mean_i = (s + c) * inv_m;
        double ss = 0.0;
        double sc = 0.0;
        for (int64_t k = 0; k < m; ++k) {
            double d = u[i + k] - mean_i;
            double d2 = d * d;
            double t = ss + d2;
            if (std::abs(ss) >= std::abs(d2)) {
                sc += (ss - t) + d2;
            } else {
                sc += (d2 - t) + ss;
            }
            ss = t;
        }
        run_var = ss + sc;
        mu_a[i] = mean_i;
    };

    double peak_var = 0.0;
    double norm_hwm = 0.0;
    for (int64_t i = 0; i < p.n; ++i) {
        if ((i % kReanchorStats) == 0 || !std::isfinite(run_sum) || !std::isfinite(run_var)) {
            recompute_exact(i);
            peak_var = run_var;
        } else {
            double old_mu = mu_a[i - 1];
            double x_out = u[i - 1];
            double x_in = u[i + m - 1];
            double delta = x_in - x_out;
            double t = run_sum + delta;
            if (std::abs(run_sum) >= std::abs(delta)) {
                run_comp += (run_sum - t) + delta;
            } else {
                run_comp += (delta - t) + run_sum;
            }
            run_sum = t;
            double new_mu = (run_sum + run_comp) * inv_m;
            mu_a[i] = new_mu;
            run_var += (x_in - new_mu) * (x_in - old_mu) - (x_out - new_mu) * (x_out - old_mu);
            if (run_var > peak_var) peak_var = run_var;
            if (run_var < 0.0 || !std::isfinite(run_var)) {
                recompute_exact(i);
                peak_var = run_var;
            }
        }

        bool no_nan = (bad_prefix[static_cast<size_t>(i + m)] == bad_prefix[static_cast<size_t>(i)]);
        if (!no_nan) {
            p.valid[static_cast<size_t>(i)] = 0;
            p.is_nan[static_cast<size_t>(i)] = 1;
            p.is_flat[static_cast<size_t>(i)] = 0;
            inv_a[i] = kQuietNaN;
            p.has_any_invalid = true;
            p.drop_prefix[static_cast<size_t>(i + 1)] = p.drop_prefix[static_cast<size_t>(i)];
            continue;
        }
        p.is_nan[static_cast<size_t>(i)] = 0;

        // Relative flatness floor in scaled u-units (invariant to global rescaling by `scale`).
        double abs_level_scaled = std::abs(mu_a[i] + (anchor / scale));
        double rel_floor = abs_level_scaled * 1e-13;
        double var_floor = std::max(1e-300, static_cast<double>(m) * rel_floor * rel_floor);
        double recheck_floor = std::max(var_floor, peak_var * 1e-4);
        if (run_var <= recheck_floor || !std::isfinite(run_var)) {
            recompute_exact(i);
            peak_var = run_var;
            if (run_var <= var_floor || !std::isfinite(run_var)) {
                p.valid[static_cast<size_t>(i)] = 0;
                p.is_flat[static_cast<size_t>(i)] = 1;
                inv_a[i] = kQuietNaN;
                p.has_any_flat = true;
                p.has_any_invalid = true;
                p.drop_prefix[static_cast<size_t>(i + 1)] = p.drop_prefix[static_cast<size_t>(i)];
                continue;
            }
        }

        p.valid[static_cast<size_t>(i)] = 1;
        double norm_i = std::sqrt(run_var);
        inv_a[i] = 1.0 / norm_i;
        bool recovered_from_invalid = (i > 0 && p.valid[static_cast<size_t>(i - 1)] == 0);
        if (norm_hwm > 32.0 * norm_i || recovered_from_invalid) {
            p.is_drop[static_cast<size_t>(i)] = 1;
            p.has_drops = true;
            norm_hwm = norm_i;
        } else if (norm_i > norm_hwm) {
            norm_hwm = norm_i;
        }
        p.drop_prefix[static_cast<size_t>(i + 1)] =
            p.drop_prefix[static_cast<size_t>(i)] + p.is_drop[static_cast<size_t>(i)];
    }

    // Exact left-pad means so SCAMP covariance recurrence is algebraically exact crossing j = -1 -> 0
    for (int64_t j = -kSeriesPad; j < 0; ++j) {
        double s = 0.0;
        for (int64_t k = 0; k < m; ++k) {
            s += u[j + k];
        }
        mu_a[j] = s * inv_m;
    }
    for (int64_t i = -kSeriesPad; i + 1 < p.n; ++i) {
        double x_out = u[i];
        double x_in = u[i + m];
        df_a[i] = 0.5 * (x_in - x_out);
        dg_a[i] = (x_in - mu_a[i + 1]) + (x_out - mu_a[i]);
    }
    return p;
}

// Exact centered dot product between subsequence A[i..i+m-1] and B[j..j+m-1]
inline double exact_cov_scalar(
    const double* uA, double muA,
    const double* uB, double muB,
    int64_t m
) {
#if defined(FMP_HAS_AVX2)
    __m256d v_muA = _mm256_set1_pd(muA);
    __m256d v_muB = _mm256_set1_pd(muB);
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    int64_t k = 0;
    for (; k + 8 <= m; k += 8) {
        __m256d a0 = _mm256_sub_pd(_mm256_loadu_pd(uA + k), v_muA);
        __m256d b0 = _mm256_sub_pd(_mm256_loadu_pd(uB + k), v_muB);
        __m256d a1 = _mm256_sub_pd(_mm256_loadu_pd(uA + k + 4), v_muA);
        __m256d b1 = _mm256_sub_pd(_mm256_loadu_pd(uB + k + 4), v_muB);
        acc0 = _mm256_fmadd_pd(a0, b0, acc0);
        acc1 = _mm256_fmadd_pd(a1, b1, acc1);
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    for (; k + 4 <= m; k += 4) {
        __m256d a0 = _mm256_sub_pd(_mm256_loadu_pd(uA + k), v_muA);
        __m256d b0 = _mm256_sub_pd(_mm256_loadu_pd(uB + k), v_muB);
        acc0 = _mm256_fmadd_pd(a0, b0, acc0);
    }
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, acc0);
    double sum = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
    for (; k < m; ++k) {
        sum += (uA[k] - muA) * (uB[k] - muB);
    }
    return sum;
#elif defined(FMP_HAS_NEON)
    float64x2_t v_muA = vdupq_n_f64(muA);
    float64x2_t v_muB = vdupq_n_f64(muB);
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    int64_t k = 0;
    for (; k + 4 <= m; k += 4) {
        float64x2_t a0 = vsubq_f64(vld1q_f64(uA + k), v_muA);
        float64x2_t b0 = vsubq_f64(vld1q_f64(uB + k), v_muB);
        float64x2_t a1 = vsubq_f64(vld1q_f64(uA + k + 2), v_muA);
        float64x2_t b1 = vsubq_f64(vld1q_f64(uB + k + 2), v_muB);
        acc0 = vfmaq_f64(acc0, a0, b0);
        acc1 = vfmaq_f64(acc1, a1, b1);
    }
    acc0 = vaddq_f64(acc0, acc1);
    double sum = vgetq_lane_f64(acc0, 0) + vgetq_lane_f64(acc0, 1);
    for (; k < m; ++k) {
        sum += (uA[k] - muA) * (uB[k] - muB);
    }
    return sum;
#else
    double sum = 0.0;
    for (int64_t k = 0; k < m; ++k) {
        sum += (uA[k] - muA) * (uB[k] - muB);
    }
    return sum;
#endif
}

// Exact two-pass unit-vector half-L2 distance sqrt(0.5 * ||zA - zB||^2 / m) == sqrt(1 - corr(a, b)).
// Avoids catastrophic cancellation in (1 - corr) when corr ~ 1.0 (verbatim or affine copies).
static inline double exact_unit_half_dist_scalar(const double* a, const double* b, int64_t m) {
    double sum_a = 0.0, sum_b = 0.0;
    for (int64_t k = 0; k < m; ++k) {
        sum_a += a[k];
        sum_b += b[k];
    }
    const double inv_m = 1.0 / static_cast<double>(m);
    const double ma = sum_a * inv_m;
    const double mb = sum_b * inv_m;
    double va = 0.0, vb = 0.0;
    for (int64_t k = 0; k < m; ++k) {
        double da = a[k] - ma;
        double db = b[k] - mb;
        va += da * da;
        vb += db * db;
    }
    if (!(va > 0.0) || !(vb > 0.0)) return 0.0;
    const double inv_a = 1.0 / std::sqrt(va);
    const double inv_b = 1.0 / std::sqrt(vb);
    double s2 = 0.0;
    for (int64_t k = 0; k < m; ++k) {
        double d = (a[k] - ma) * inv_a - (b[k] - mb) * inv_b;
        s2 += d * d;
    }
    return std::sqrt(0.5 * s2);
}

static inline double exact_z_dist_scalar(const double* a, const double* b, int64_t m) {
    return std::sqrt(2.0 * static_cast<double>(m)) * exact_unit_half_dist_scalar(a, b, m);
}

#if defined(__GNUC__) || defined(__clang__)
#define FMP_COLD_NOINLINE __attribute__((noinline, cold))
#define FMP_NOINLINE __attribute__((noinline))
#else
#define FMP_COLD_NOINLINE
#define FMP_NOINLINE
#endif

// Compute W (multiple of 4) consecutive centered covariances cov(i0, j0 + 0..W-1)
// simultaneously in SIMD registers (reuses uA[i0+k] - muA across all W columns).
template <int W>
inline void exact_cov_W(
    const double* uA_i0, double muA,
    const double* uB, const double* muB, int64_t j0,
    int64_t m,
    double out_cov[W]
) {
    static_assert((W % 4) == 0 && W >= 4, "W must be a positive multiple of 4");
#if defined(FMP_HAS_AVX2)
    if constexpr (W > 24) {
        constexpr int H = W / 2;
        exact_cov_W<H>(uA_i0, muA, uB, muB, j0, m, out_cov);
        exact_cov_W<W - H>(uA_i0, muA, uB, muB, j0 + H, m, out_cov + H);
    } else {
        constexpr int V = W / 4;
        __m256d c[V];
        __m256d mb[V];
        for (int v = 0; v < V; ++v) {
            c[v] = _mm256_setzero_pd();
            mb[v] = _mm256_loadu_pd(muB + j0 + 4 * v);
        }
        const double* bptr = uB + j0;
        for (int64_t k = 0; k < m; ++k) {
            __m256d va = _mm256_set1_pd(uA_i0[k] - muA);
            const double* bk = bptr + k;
            for (int v = 0; v < V; ++v) {
                c[v] = _mm256_fmadd_pd(va, _mm256_sub_pd(_mm256_loadu_pd(bk + 4 * v), mb[v]), c[v]);
            }
        }
        for (int v = 0; v < V; ++v) {
            _mm256_storeu_pd(out_cov + 4 * v, c[v]);
        }
    }
#elif defined(FMP_HAS_NEON)
    if constexpr (W > 16) {
        constexpr int H = W / 2;
        exact_cov_W<H>(uA_i0, muA, uB, muB, j0, m, out_cov);
        exact_cov_W<W - H>(uA_i0, muA, uB, muB, j0 + H, m, out_cov + H);
    } else {
        constexpr int V = W / 2;
        float64x2_t c[V];
        float64x2_t mb[V];
        for (int v = 0; v < V; ++v) {
            c[v] = vdupq_n_f64(0.0);
            mb[v] = vld1q_f64(muB + j0 + 2 * v);
        }
        const double* bptr = uB + j0;
        for (int64_t k = 0; k < m; ++k) {
            float64x2_t va = vdupq_n_f64(uA_i0[k] - muA);
            const double* bk = bptr + k;
            for (int v = 0; v < V; ++v) {
                c[v] = vfmaq_f64(c[v], va, vsubq_f64(vld1q_f64(bk + 2 * v), mb[v]));
            }
        }
        for (int v = 0; v < V; ++v) {
            vst1q_f64(out_cov + 2 * v, c[v]);
        }
    }
#else
    for (int lane = 0; lane < W; ++lane) {
        out_cov[lane] = exact_cov_scalar(uA_i0, muA, uB + j0 + lane, muB[j0 + lane], m);
    }
#endif
}

// ---------------------------------------------------------------------------
// Core 16-Diagonal-Wide Zero-Spill AVX2/FMA/NEON Tiled Join Microkernel
// with Persistent Band Covariance Carry-Over across L1-Resident Row Tiles
//
// Evaluates all (i0 + r, j0 + r + lane) for r in [0, rows), lane in [0, 16),
// reading initial cov(i0, j0 + 0..15) from 32B-aligned cov_io and writing
// advanced cov(i0 + rows, j0 + rows + 0..15) back to cov_io.
// - UpdateCol == true: maximizes true corr for both row_best[i] & col_best[j],
//   combining row+col checks via min(v_rb, col_best) + single movemask.
// - UpdateCol == false: factors invA[i] out of the hot loop! Maximizes
//   c * invB[col] (or |c * invB[col]| if CheckNeg) against row_best[i]
//   initialized to (A.valid[i] ? -inf : NaN); caller multiplies final best by invA[i].
// ---------------------------------------------------------------------------
template <bool UpdateCol, bool CheckNeg>
FMP_NOINLINE void eval_strip_16_carry(
    const double* __restrict__ invA, const double* __restrict__ dfA, const double* __restrict__ dgA,
    const double* __restrict__ invB, const double* __restrict__ dfB, const double* __restrict__ dgB,
    int64_t i0, int64_t j0, int64_t rows,
    double* __restrict__ cov_io,
    double* __restrict__ row_best, int64_t* __restrict__ row_idx,
    double* __restrict__ col_best, int64_t* __restrict__ col_idx
) {
#if defined(FMP_HAS_AVX2)
    __m256d c0 = _mm256_load_pd(cov_io);
    __m256d c1 = _mm256_load_pd(cov_io + 4);
    __m256d c2 = _mm256_load_pd(cov_io + 8);
    __m256d c3 = _mm256_load_pd(cov_io + 12);

    const double* __restrict__ p_invA = invA + i0;
    const double* __restrict__ p_dfA = dfA + i0;
    const double* __restrict__ p_dgA = dgA + i0;
    double* __restrict__ p_rb = row_best + i0;
    int64_t* __restrict__ p_ri = row_idx + i0;
    const double* __restrict__ p_invB = invB + j0;
    const double* __restrict__ p_dfB = dfB + j0;
    const double* __restrict__ p_dgB = dgB + j0;
    double* __restrict__ p_cb = UpdateCol ? (col_best + j0) : nullptr;
    int64_t* __restrict__ p_ci = UpdateCol ? (col_idx + j0) : nullptr;
    if constexpr (!UpdateCol) {
        (void)p_invA;
        (void)p_cb;
        (void)p_ci;
    }

    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    if constexpr (!CheckNeg) {
        (void)sign_mask;
    }

    for (int64_t r = 0; r < rows; ++r) {
        __m256d v_rb = _mm256_set1_pd(p_rb[r]);
        __m256d corr0, corr1, corr2, corr3;
        __m256d m_any;

        if constexpr (UpdateCol) {
            __m256d v_invi = _mm256_set1_pd(p_invA[r]);
            corr0 = _mm256_mul_pd(_mm256_mul_pd(c0, _mm256_loadu_pd(p_invB + r)), v_invi);
            corr1 = _mm256_mul_pd(_mm256_mul_pd(c1, _mm256_loadu_pd(p_invB + r + 4)), v_invi);
            corr2 = _mm256_mul_pd(_mm256_mul_pd(c2, _mm256_loadu_pd(p_invB + r + 8)), v_invi);
            corr3 = _mm256_mul_pd(_mm256_mul_pd(c3, _mm256_loadu_pd(p_invB + r + 12)), v_invi);
            if constexpr (CheckNeg) {
                corr0 = _mm256_andnot_pd(sign_mask, corr0);
                corr1 = _mm256_andnot_pd(sign_mask, corr1);
                corr2 = _mm256_andnot_pd(sign_mask, corr2);
                corr3 = _mm256_andnot_pd(sign_mask, corr3);
            }
            // row_best and col_best are never NaN when UpdateCol==true, so
            // (corr > v_rb) || (corr > cb) iff corr > min(v_rb, cb) (exact IEEE-754!).
            __m256d m0 = _mm256_cmp_pd(corr0, _mm256_min_pd(v_rb, _mm256_loadu_pd(p_cb + r)), _CMP_GT_OQ);
            __m256d m1 = _mm256_cmp_pd(corr1, _mm256_min_pd(v_rb, _mm256_loadu_pd(p_cb + r + 4)), _CMP_GT_OQ);
            __m256d m2 = _mm256_cmp_pd(corr2, _mm256_min_pd(v_rb, _mm256_loadu_pd(p_cb + r + 8)), _CMP_GT_OQ);
            __m256d m3 = _mm256_cmp_pd(corr3, _mm256_min_pd(v_rb, _mm256_loadu_pd(p_cb + r + 12)), _CMP_GT_OQ);
            m_any = _mm256_or_pd(_mm256_or_pd(m0, m1), _mm256_or_pd(m2, m3));
        } else {
            corr0 = _mm256_mul_pd(c0, _mm256_loadu_pd(p_invB + r));
            corr1 = _mm256_mul_pd(c1, _mm256_loadu_pd(p_invB + r + 4));
            corr2 = _mm256_mul_pd(c2, _mm256_loadu_pd(p_invB + r + 8));
            corr3 = _mm256_mul_pd(c3, _mm256_loadu_pd(p_invB + r + 12));
            if constexpr (CheckNeg) {
                corr0 = _mm256_andnot_pd(sign_mask, corr0);
                corr1 = _mm256_andnot_pd(sign_mask, corr1);
                corr2 = _mm256_andnot_pd(sign_mask, corr2);
                corr3 = _mm256_andnot_pd(sign_mask, corr3);
            }
            __m256d m0 = _mm256_cmp_pd(corr0, v_rb, _CMP_GT_OQ);
            __m256d m1 = _mm256_cmp_pd(corr1, v_rb, _CMP_GT_OQ);
            __m256d m2 = _mm256_cmp_pd(corr2, v_rb, _CMP_GT_OQ);
            __m256d m3 = _mm256_cmp_pd(corr3, v_rb, _CMP_GT_OQ);
            m_any = _mm256_or_pd(_mm256_or_pd(m0, m1), _mm256_or_pd(m2, m3));
        }

        if (__builtin_expect(_mm256_movemask_pd(m_any) != 0, 0)) {
            alignas(32) double vals[16];
            _mm256_store_pd(vals, corr0);
            _mm256_store_pd(vals + 4, corr1);
            _mm256_store_pd(vals + 8, corr2);
            _mm256_store_pd(vals + 12, corr3);
            double rb = p_rb[r];
            int64_t ri = p_ri[r];
            const int64_t col = j0 + r;
            const int64_t i = i0 + r;
            if constexpr (UpdateCol) {
                double* cb_r = p_cb + r;
                int64_t* ci_r = p_ci + r;
                for (int lane = 0; lane < 16; ++lane) {
                    double v = vals[lane];
                    if (v > rb) {
                        rb = v;
                        ri = col + lane;
                    }
                    if (v > cb_r[lane]) {
                        cb_r[lane] = v;
                        ci_r[lane] = i;
                    }
                }
            } else {
                for (int lane = 0; lane < 16; ++lane) {
                    double v = vals[lane];
                    if (v > rb) {
                        rb = v;
                        ri = col + lane;
                    }
                }
            }
            p_rb[r] = rb;
            p_ri[r] = ri;
        }

        // Unconditional advance (sentinel-padded slab guarantees in-bounds & exact carry-over)
        __m256d v_dfi = _mm256_set1_pd(p_dfA[r]);
        __m256d v_dgi = _mm256_set1_pd(p_dgA[r]);
        c0 = _mm256_fmadd_pd(v_dfi, _mm256_loadu_pd(p_dgB + r),
             _mm256_fmadd_pd(v_dgi, _mm256_loadu_pd(p_dfB + r), c0));
        c1 = _mm256_fmadd_pd(v_dfi, _mm256_loadu_pd(p_dgB + r + 4),
             _mm256_fmadd_pd(v_dgi, _mm256_loadu_pd(p_dfB + r + 4), c1));
        c2 = _mm256_fmadd_pd(v_dfi, _mm256_loadu_pd(p_dgB + r + 8),
             _mm256_fmadd_pd(v_dgi, _mm256_loadu_pd(p_dfB + r + 8), c2));
        c3 = _mm256_fmadd_pd(v_dfi, _mm256_loadu_pd(p_dgB + r + 12),
             _mm256_fmadd_pd(v_dgi, _mm256_loadu_pd(p_dfB + r + 12), c3));
    }

    _mm256_store_pd(cov_io, c0);
    _mm256_store_pd(cov_io + 4, c1);
    _mm256_store_pd(cov_io + 8, c2);
    _mm256_store_pd(cov_io + 12, c3);
#elif defined(FMP_HAS_NEON)
    float64x2_t c[8];
    for (int v = 0; v < 8; ++v) {
        c[v] = vld1q_f64(cov_io + 2 * v);
    }

    const double* __restrict__ p_invA = invA + i0;
    const double* __restrict__ p_dfA = dfA + i0;
    const double* __restrict__ p_dgA = dgA + i0;
    double* __restrict__ p_rb = row_best + i0;
    int64_t* __restrict__ p_ri = row_idx + i0;
    const double* __restrict__ p_invB = invB + j0;
    const double* __restrict__ p_dfB = dfB + j0;
    const double* __restrict__ p_dgB = dgB + j0;
    double* __restrict__ p_cb = UpdateCol ? (col_best + j0) : nullptr;
    int64_t* __restrict__ p_ci = UpdateCol ? (col_idx + j0) : nullptr;
    if constexpr (!UpdateCol) {
        (void)p_invA;
        (void)p_cb;
        (void)p_ci;
    }

    for (int64_t r = 0; r < rows; ++r) {
        float64x2_t v_rb = vdupq_n_f64(p_rb[r]);
        float64x2_t corr[8];
        uint64x2_t m_any = vdupq_n_u64(0);

        if constexpr (UpdateCol) {
            float64x2_t v_invi = vdupq_n_f64(p_invA[r]);
            for (int v = 0; v < 8; ++v) {
                corr[v] = vmulq_f64(vmulq_f64(c[v], vld1q_f64(p_invB + r + 2 * v)), v_invi);
                if constexpr (CheckNeg) corr[v] = vabsq_f64(corr[v]);
                float64x2_t thr = vminq_f64(v_rb, vld1q_f64(p_cb + r + 2 * v));
                uint64x2_t mv = vcgtq_f64(corr[v], thr);
                m_any = (v == 0) ? mv : vorrq_u64(m_any, mv);
            }
        } else {
            for (int v = 0; v < 8; ++v) {
                corr[v] = vmulq_f64(c[v], vld1q_f64(p_invB + r + 2 * v));
                if constexpr (CheckNeg) corr[v] = vabsq_f64(corr[v]);
                uint64x2_t mv = vcgtq_f64(corr[v], v_rb);
                m_any = (v == 0) ? mv : vorrq_u64(m_any, mv);
            }
        }

        if (__builtin_expect((vgetq_lane_u64(m_any, 0) | vgetq_lane_u64(m_any, 1)) != 0, 0)) {
            double vals[16];
            for (int v = 0; v < 8; ++v) vst1q_f64(vals + 2 * v, corr[v]);
            double rb = p_rb[r];
            int64_t ri = p_ri[r];
            const int64_t col = j0 + r;
            const int64_t i = i0 + r;
            if constexpr (UpdateCol) {
                double* cb_r = p_cb + r;
                int64_t* ci_r = p_ci + r;
                for (int lane = 0; lane < 16; ++lane) {
                    double v = vals[lane];
                    if (v > rb) { rb = v; ri = col + lane; }
                    if (v > cb_r[lane]) { cb_r[lane] = v; ci_r[lane] = i; }
                }
            } else {
                for (int lane = 0; lane < 16; ++lane) {
                    double v = vals[lane];
                    if (v > rb) { rb = v; ri = col + lane; }
                }
            }
            p_rb[r] = rb;
            p_ri[r] = ri;
        }

        float64x2_t v_dfi = vdupq_n_f64(p_dfA[r]);
        float64x2_t v_dgi = vdupq_n_f64(p_dgA[r]);
        for (int v = 0; v < 8; ++v) {
            c[v] = vfmaq_f64(vfmaq_f64(c[v], v_dgi, vld1q_f64(p_dfB + r + 2 * v)),
                             v_dfi, vld1q_f64(p_dgB + r + 2 * v));
        }
    }

    for (int v = 0; v < 8; ++v) {
        vst1q_f64(cov_io + 2 * v, c[v]);
    }
#else
    double c[16];
    std::memcpy(c, cov_io, sizeof(c));
    if constexpr (!UpdateCol) {
        (void)invA;
        (void)col_best;
        (void)col_idx;
    }
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t i = i0 + r;
        const int64_t col = j0 + r;
        double rb = row_best[i];
        int64_t ri = row_idx[i];
        if constexpr (UpdateCol) {
            const double invi = invA[i];
            for (int lane = 0; lane < 16; ++lane) {
                double corr = (c[lane] * invB[col + lane]) * invi;
                if constexpr (CheckNeg) corr = std::abs(corr);
                if (corr > rb) {
                    rb = corr;
                    ri = col + lane;
                }
                if (corr > col_best[col + lane]) {
                    col_best[col + lane] = corr;
                    col_idx[col + lane] = i;
                }
            }
        } else {
            for (int lane = 0; lane < 16; ++lane) {
                double scaled = c[lane] * invB[col + lane];
                if constexpr (CheckNeg) scaled = std::abs(scaled);
                if (scaled > rb) {
                    rb = scaled;
                    ri = col + lane;
                }
            }
        }
        row_best[i] = rb;
        row_idx[i] = ri;
        const double dfi = dfA[i];
        const double dgi = dgA[i];
        for (int lane = 0; lane < 16; ++lane) {
            c[lane] += dfi * dgB[col + lane] + dgi * dfB[col + lane];
        }
    }
    std::memcpy(cov_io, c, sizeof(c));
#endif
}

template <bool UpdateCol, bool CheckNeg>
inline void eval_strip_16_guarded(
    const SeriesPrep& A, const SeriesPrep& B,
    int64_t i0, int64_t j0, int64_t rows,
    double* __restrict__ cov_io,
    double* __restrict__ row_best, int64_t* __restrict__ row_idx,
    double* __restrict__ col_best, int64_t* __restrict__ col_idx
) {
    const int64_t nB = B.n;
    const int64_t m = A.m;
    const double* uA = A.u_ptr();
    const double* muA = A.mu_ptr();
    const double* invA = A.inv_ptr();
    const double* dfA = A.df_ptr();
    const double* dgA = A.dg_ptr();
    const double* uB = B.u_ptr();
    const double* muB = B.mu_ptr();
    const double* invB = B.inv_ptr();
    const double* dfB = B.df_ptr();
    const double* dgB = B.dg_ptr();
    const uint8_t* dropA = A.has_drops ? A.is_drop.data() : nullptr;
    const uint8_t* dropB = B.has_drops ? B.is_drop.data() : nullptr;

    double c[16];
    std::memcpy(c, cov_io, sizeof(c));
    if constexpr (!UpdateCol) {
        (void)invA;
        (void)col_best;
        (void)col_idx;
    }
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t i = i0 + r;
        const int64_t col = j0 + r;
        if (dropA && dropA[static_cast<size_t>(i)]) {
            exact_cov_W<kStripW>(uA + i, muA[i], uB, muB, col, m, c);
        } else if (dropB) {
            for (int lane = 0; lane < 16; ++lane) {
                int64_t cidx = col + lane;
                if (cidx >= 0 && cidx < nB && dropB[static_cast<size_t>(cidx)]) {
                    c[lane] = exact_cov_scalar(uA + i, muA[i], uB + cidx, muB[cidx], m);
                }
            }
        }
        double rb = row_best[i];
        int64_t ri = row_idx[i];
        if constexpr (UpdateCol) {
            const double invi = invA[i];
            for (int lane = 0; lane < 16; ++lane) {
                double corr = (c[lane] * invB[col + lane]) * invi;
                if constexpr (CheckNeg) corr = std::abs(corr);
                if (corr > rb) {
                    rb = corr;
                    ri = col + lane;
                }
                if (corr > col_best[col + lane]) {
                    col_best[col + lane] = corr;
                    col_idx[col + lane] = i;
                }
            }
        } else {
            for (int lane = 0; lane < 16; ++lane) {
                double scaled = c[lane] * invB[col + lane];
                if constexpr (CheckNeg) scaled = std::abs(scaled);
                if (scaled > rb) {
                    rb = scaled;
                    ri = col + lane;
                }
            }
        }
        row_best[i] = rb;
        row_idx[i] = ri;
        const double dfi = dfA[i];
        const double dgi = dgA[i];
        for (int lane = 0; lane < 16; ++lane) {
            c[lane] += dfi * dgB[col + lane] + dgi * dfB[col + lane];
        }
    }
    std::memcpy(cov_io, c, sizeof(c));
}

template <bool UpdateCol, bool CheckNeg>
inline void eval_diag_scalar(
    const SeriesPrep& A, const SeriesPrep& B,
    int64_t i0, int64_t j0, int64_t rows,
    double* row_best, int64_t* row_idx,
    double* col_best, int64_t* col_idx
) {
    const int64_t m = A.m;
    const double* uA = A.u_ptr();
    const double* muA = A.mu_ptr();
    const double* invA = A.inv_ptr();
    const double* dfA = A.df_ptr();
    const double* dgA = A.dg_ptr();
    const double* uB = B.u_ptr();
    const double* muB = B.mu_ptr();
    const double* invB = B.inv_ptr();
    const double* dfB = B.df_ptr();
    const double* dgB = B.dg_ptr();
    const uint8_t* dropA = A.has_drops ? A.is_drop.data() : nullptr;
    const uint8_t* dropB = B.has_drops ? B.is_drop.data() : nullptr;

    if constexpr (!UpdateCol) {
        (void)invA;
        (void)col_best;
        (void)col_idx;
    }
    double c = exact_cov_scalar(uA + i0, muA[i0], uB + j0, muB[j0], m);
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t i = i0 + r;
        const int64_t col = j0 + r;
        if (r > 0 && ((dropA && dropA[static_cast<size_t>(i)]) ||
                      (dropB && dropB[static_cast<size_t>(col)]))) {
            c = exact_cov_scalar(uA + i, muA[i], uB + col, muB[col], m);
        }
        if constexpr (UpdateCol) {
            double corr = (c * invB[col]) * invA[i];
            if constexpr (CheckNeg) corr = std::abs(corr);
            if (corr > row_best[i]) {
                row_best[i] = corr;
                row_idx[i] = col;
            }
            if (corr > col_best[col]) {
                col_best[col] = corr;
                col_idx[col] = i;
            }
        } else {
            double scaled = c * invB[col];
            if constexpr (CheckNeg) scaled = std::abs(scaled);
            if (scaled > row_best[i]) {
                row_best[i] = scaled;
                row_idx[i] = col;
            }
        }
        c += dfA[i] * dgB[col] + dgA[i] * dfB[col];
    }
}

// Process a single L1-resident diagonal band [db, db_end), tiled by kTileH.
// Sentinel padding (kSeriesPad=32, invB=NaN outside [0, nB), exact left mu/df/dg)
// runs 100% of diagonal strips at full 16-wide SIMD with zero scalar corners,
// and persists band_cov[kTileW] (2 KiB L1d) across row tiles with periodic re-anchor.
template <bool UpdateCol, bool CheckNeg>
void run_diag_band(
    const SeriesPrep& A, const SeriesPrep& B,
    int64_t db, int64_t db_end,
    double* row_best, int64_t* row_idx,
    double* col_best, int64_t* col_idx
) {
    const int64_t nA = A.n;
    const int64_t nB = B.n;
    const int64_t m = A.m;
    const double* uA = A.u_ptr();
    const double* muA = A.mu_ptr();
    const double* invA = A.inv_ptr();
    const double* dfA = A.df_ptr();
    const double* dgA = A.dg_ptr();
    const double* uB = B.u_ptr();
    const double* muB = B.mu_ptr();
    const double* invB = B.inv_ptr();
    const double* dfB = B.df_ptr();
    const double* dgB = B.dg_ptr();
    const bool any_drops = A.has_drops || B.has_drops;

    int64_t i_min = std::max<int64_t>(0, -(db_end - 1));
    int64_t i_max = std::min<int64_t>(nA, nB - db);
    if (i_min >= i_max) return;

    constexpr int64_t kMaxStrips = (kTileW + kStripW - 1) / kStripW;
    alignas(32) double band_cov[kMaxStrips * kStripW];
    int64_t next_row[kMaxStrips];
    int64_t since_reanchor[kMaxStrips];
    for (int64_t s = 0; s < kMaxStrips; ++s) {
        next_row[s] = -1;
        since_reanchor[s] = 0;
    }

    for (int64_t ti = i_min; ti < i_max; ti += kTileH) {
        int64_t ti_end = std::min(i_max, ti + kTileH);
        int64_t d = db;
        int64_t s = 0;
        while (d < db_end) {
            if (d + kStripW <= db_end || db_end == nB) {
                int64_t i_lo = std::max<int64_t>(ti, -(d + kStripW - 1));
                int64_t i_hi = std::min<int64_t>(ti_end, nB - d);
                if (i_lo < i_hi) {
                    double* cptr = band_cov + s * kStripW;
                    int64_t rows = i_hi - i_lo;
                    int64_t j0 = i_lo + d;
                    if (next_row[s] != i_lo || since_reanchor[s] >= kCovReanchor) {
                        exact_cov_W<kStripW>(uA + i_lo, muA[i_lo], uB, muB, j0, m, cptr);
                        since_reanchor[s] = rows;
                    } else {
                        since_reanchor[s] += rows;
                    }
                    bool strip_has_drop = false;
                    if (__builtin_expect(any_drops, 0)) {
                        if (A.has_drops && A.drop_prefix[static_cast<size_t>(i_hi)] > A.drop_prefix[static_cast<size_t>(i_lo)]) {
                            strip_has_drop = true;
                        } else if (B.has_drops) {
                            int64_t bj_lo = std::clamp<int64_t>(j0, 0, nB);
                            int64_t bj_hi = std::clamp<int64_t>(j0 + rows + kStripW - 1, 0, nB);
                            if (B.drop_prefix[static_cast<size_t>(bj_hi)] > B.drop_prefix[static_cast<size_t>(bj_lo)]) {
                                strip_has_drop = true;
                            }
                        }
                    }
                    if (__builtin_expect(strip_has_drop, 0)) {
                        eval_strip_16_guarded<UpdateCol, CheckNeg>(
                            A, B, i_lo, j0, rows, cptr,
                            row_best, row_idx, col_best, col_idx);
                    } else {
                        eval_strip_16_carry<UpdateCol, CheckNeg>(
                            invA, dfA, dgA, invB, dfB, dgB,
                            i_lo, j0, rows, cptr,
                            row_best, row_idx, col_best, col_idx);
                    }
                    next_row[s] = i_hi;
                }
                d += kStripW;
                ++s;
                continue;
            }
            int64_t di_lo = std::max<int64_t>(ti, -d);
            int64_t di_hi = std::min<int64_t>(ti_end, nB - d);
            if (di_lo < di_hi) {
                eval_diag_scalar<UpdateCol, CheckNeg>(
                    A, B, di_lo, di_lo + d, di_hi - di_lo,
                    row_best, row_idx, col_best, col_idx);
            }
            ++d;
        }
    }
}

// Static contiguous diagonal range runner (ideal L2 locality for AB-join & leak-scan)
template <bool UpdateCol, bool CheckNeg>
void run_diag_static(
    const SeriesPrep& A, const SeriesPrep& B,
    int64_t d_lo, int64_t d_hi,
    double* row_best, int64_t* row_idx,
    double* col_best, int64_t* col_idx
) {
    for (int64_t db = d_lo; db < d_hi; db += kTileW) {
        int64_t db_end = std::min(d_hi, db + kTileW);
        run_diag_band<UpdateCol, CheckNeg>(A, B, db, db_end, row_best, row_idx, col_best, col_idx);
    }
}

// Compute equal-cell-count diagonal cut points aligned to kStripW for static AB/leak threads
std::vector<int64_t> compute_balanced_diag_cuts(
    int64_t nA, int64_t nB, int64_t d_min, int64_t d_max, int actual_nt
) {
    std::vector<int64_t> cuts(static_cast<size_t>(actual_nt + 1), d_max);
    cuts[0] = d_min;
    if (actual_nt <= 1 || d_min >= d_max) return cuts;
    const double total_cells = static_cast<double>(nA) * static_cast<double>(nB);
    double acc = 0.0;
    int next_t = 1;
    for (int64_t d = d_min; d < d_max && next_t < actual_nt; ++d) {
        int64_t lo_i = std::max<int64_t>(0, -d);
        int64_t hi_i = std::min<int64_t>(nA, nB - d);
        if (hi_i > lo_i) acc += static_cast<double>(hi_i - lo_i);
        while (next_t < actual_nt &&
               acc * static_cast<double>(actual_nt) >= total_cells * static_cast<double>(next_t)) {
            int64_t cut = d_min + (((d + 1 - d_min) + kStripW - 1) / kStripW) * kStripW;
            if (cut > d_max) cut = d_max;
            if (cut < cuts[static_cast<size_t>(next_t - 1)]) {
                cut = cuts[static_cast<size_t>(next_t - 1)];
            }
            cuts[static_cast<size_t>(next_t)] = cut;
            ++next_t;
        }
    }
    return cuts;
}

// Dynamic work-stealing diagonal band runner across [d_min, d_max) (for self-join)
template <bool UpdateCol, bool CheckNeg>
void run_diag_dynamic(
    const SeriesPrep& A, const SeriesPrep& B,
    std::atomic<int64_t>& next_db, int64_t d_max,
    double* row_best, int64_t* row_idx,
    double* col_best, int64_t* col_idx
) {
    while (true) {
        int64_t db = next_db.fetch_add(kTileW, std::memory_order_relaxed);
        if (db >= d_max) break;
        int64_t db_end = std::min(d_max, db + kTileW);
        run_diag_band<UpdateCol, CheckNeg>(A, B, db, db_end, row_best, row_idx, col_best, col_idx);
    }
}

// ---------------------------------------------------------------------------
// High-Performance Tiled SIMD Multivariate Matrix Profile (mSTUMP) Kernels
// ---------------------------------------------------------------------------
constexpr int64_t kMsTileH = 128;
constexpr int64_t kMsTileW = 128;
constexpr int64_t kMsBStride = kMsTileH + kMsTileW + 8; // 264 (non-4K-aliasing stride)

inline void exact_cov_8(
    const double* uA_i0, double muA,
    const double* uB, const double* muB, int64_t j0,
    int64_t m,
    double out_cov[8]
) {
#if defined(FMP_HAS_AVX2)
    __m256d c0 = _mm256_setzero_pd();
    __m256d c1 = _mm256_setzero_pd();
    __m256d mb0 = _mm256_loadu_pd(muB + j0);
    __m256d mb1 = _mm256_loadu_pd(muB + j0 + 4);
    const double* bptr = uB + j0;
    for (int64_t k = 0; k < m; ++k) {
        __m256d va = _mm256_set1_pd(uA_i0[k] - muA);
        const double* bk = bptr + k;
        c0 = _mm256_fmadd_pd(va, _mm256_sub_pd(_mm256_loadu_pd(bk), mb0), c0);
        c1 = _mm256_fmadd_pd(va, _mm256_sub_pd(_mm256_loadu_pd(bk + 4), mb1), c1);
    }
    _mm256_storeu_pd(out_cov, c0);
    _mm256_storeu_pd(out_cov + 4, c1);
#elif defined(FMP_HAS_NEON)
    float64x2_t c0 = vdupq_n_f64(0.0);
    float64x2_t c1 = vdupq_n_f64(0.0);
    float64x2_t c2 = vdupq_n_f64(0.0);
    float64x2_t c3 = vdupq_n_f64(0.0);
    float64x2_t mb0 = vld1q_f64(muB + j0);
    float64x2_t mb1 = vld1q_f64(muB + j0 + 2);
    float64x2_t mb2 = vld1q_f64(muB + j0 + 4);
    float64x2_t mb3 = vld1q_f64(muB + j0 + 6);
    const double* bptr = uB + j0;
    for (int64_t k = 0; k < m; ++k) {
        float64x2_t va = vdupq_n_f64(uA_i0[k] - muA);
        const double* bk = bptr + k;
        c0 = vfmaq_f64(c0, va, vsubq_f64(vld1q_f64(bk), mb0));
        c1 = vfmaq_f64(c1, va, vsubq_f64(vld1q_f64(bk + 2), mb1));
        c2 = vfmaq_f64(c2, va, vsubq_f64(vld1q_f64(bk + 4), mb2));
        c3 = vfmaq_f64(c3, va, vsubq_f64(vld1q_f64(bk + 6), mb3));
    }
    vst1q_f64(out_cov, c0);
    vst1q_f64(out_cov + 2, c1);
    vst1q_f64(out_cov + 4, c2);
    vst1q_f64(out_cov + 6, c3);
#else
    for (int64_t lane = 0; lane < 8; ++lane) {
        out_cov[lane] = exact_cov_scalar(uA_i0, muA, uB + j0 + lane, muB[j0 + lane], m);
    }
#endif
}

template <typename Vec, typename CmpSwap>
inline void batcher_oddeven_sort(Vec* v, int64_t D, CmpSwap&& cs) {
    for (int64_t p = 1; p < D; p <<= 1) {
        const int64_t blk_mask = ~((p << 1) - 1);
        for (int64_t k = p; k >= 1; k >>= 1) {
            for (int64_t j = k % p; j + k < D; j += (k << 1)) {
                int64_t lim = std::min(k, D - j - k);
                for (int64_t i = 0; i < lim; ++i) {
                    int64_t a = i + j;
                    int64_t b = a + k;
                    if (((a ^ b) & blk_mask) == 0) {
                        cs(v[a], v[b]);
                    }
                }
            }
        }
    }
}

template <int FixedD, typename Vec, typename CmpSwap>
inline void sort_network_fixed(Vec* v, CmpSwap&& cs) {
    if constexpr (FixedD <= 1) {
        (void)v;
        (void)cs;
    } else if constexpr (FixedD == 2) {
        cs(v[0], v[1]);
    } else if constexpr (FixedD == 3) {
        cs(v[0], v[1]); cs(v[1], v[2]); cs(v[0], v[1]);
    } else if constexpr (FixedD == 4) {
        cs(v[0], v[1]); cs(v[2], v[3]);
        cs(v[0], v[2]); cs(v[1], v[3]);
        cs(v[1], v[2]);
    } else if constexpr (FixedD == 5) {
        cs(v[0], v[1]); cs(v[3], v[4]);
        cs(v[2], v[4]); cs(v[2], v[3]);
        cs(v[0], v[3]); cs(v[0], v[2]);
        cs(v[1], v[4]); cs(v[1], v[3]);
        cs(v[1], v[2]);
    } else if constexpr (FixedD == 6) {
        cs(v[1], v[2]); cs(v[4], v[5]);
        cs(v[0], v[2]); cs(v[3], v[5]);
        cs(v[0], v[1]); cs(v[3], v[4]);
        cs(v[2], v[5]); cs(v[0], v[3]);
        cs(v[1], v[4]); cs(v[2], v[4]);
        cs(v[1], v[3]); cs(v[2], v[3]);
    } else if constexpr (FixedD == 7) {
        cs(v[1], v[2]); cs(v[3], v[4]); cs(v[5], v[6]);
        cs(v[0], v[2]); cs(v[3], v[5]); cs(v[4], v[6]);
        cs(v[0], v[1]); cs(v[4], v[5]); cs(v[2], v[6]);
        cs(v[0], v[4]); cs(v[1], v[5]);
        cs(v[0], v[3]); cs(v[2], v[5]);
        cs(v[1], v[3]); cs(v[2], v[4]);
        cs(v[2], v[3]);
    } else if constexpr (FixedD == 8) {
        // Optimal 19-comparator Batcher odd-even network (depth 6)
        cs(v[0], v[1]); cs(v[2], v[3]); cs(v[4], v[5]); cs(v[6], v[7]);
        cs(v[0], v[2]); cs(v[1], v[3]); cs(v[4], v[6]); cs(v[5], v[7]);
        cs(v[1], v[2]); cs(v[5], v[6]);
        cs(v[0], v[4]); cs(v[1], v[5]); cs(v[2], v[6]); cs(v[3], v[7]);
        cs(v[2], v[4]); cs(v[3], v[5]);
        cs(v[1], v[2]); cs(v[3], v[4]); cs(v[5], v[6]);
    } else {
        batcher_oddeven_sort(v, FixedD, cs);
    }
}

inline void sort_small_doubles(double* a, int64_t D) {
    auto cs = [](double& x, double& y) {
        double lo = x < y ? x : y;
        double hi = x < y ? y : x;
        x = lo;
        y = hi;
    };
    switch (D) {
        case 1: return;
        case 2: sort_network_fixed<2>(a, cs); return;
        case 3: sort_network_fixed<3>(a, cs); return;
        case 4: sort_network_fixed<4>(a, cs); return;
        case 5: sort_network_fixed<5>(a, cs); return;
        case 6: sort_network_fixed<6>(a, cs); return;
        case 7: sort_network_fixed<7>(a, cs); return;
        case 8: sort_network_fixed<8>(a, cs); return;
        default: break;
    }
    if (D <= 32) {
        for (int64_t i = 1; i < D; ++i) {
            double key = a[i];
            int64_t j = i - 1;
            while (j >= 0 && a[j] > key) {
                a[j + 1] = a[j];
                --j;
            }
            a[j + 1] = key;
        }
    } else {
        std::sort(a, a + D);
    }
}

inline float upper_bound_f32(double x) {
    if (x <= 0.0) return 0.0f;
    if (!std::isfinite(x)) return std::numeric_limits<float>::infinity();
    float f = static_cast<float>(x);
    if (static_cast<double>(f) < x) {
        f = std::nextafter(f, std::numeric_limits<float>::infinity());
    }
    return f;
}

struct MstumpChannels {
    int64_t D = 0;
    int64_t n = 0;
    int64_t m = 0;
    bool clean_f32 = false; // true iff !any_flat && !any_nan && inv_norm in [1e-15, 1e15]
    const double* const* u = nullptr;
    const double* const* mu = nullptr;
    const double* const* inv = nullptr;
    const double* const* df = nullptr;
    const double* const* dg = nullptr;
    const double* const* flat = nullptr;
    const uint8_t* const* is_drop = nullptr;
    const int64_t* any_drop_prefix = nullptr;
};

struct MstumpWorkerBuf {
    double* mp = nullptr;         // [D][n], exact double sum of sqrt(u_(p))
    int64_t* idx = nullptr;       // [D][n]
    float* row_mp_f = nullptr;    // [n][D], upper-bounded float32 copy
    float* col_mp_f = nullptr;    // [D][n_pad_f], upper-bounded float32 copy
    float* col_mp_tile = nullptr; // [D][kMsBStride], tile-local upper-bounded float32 copy
    int64_t n_pad_f = 0;
    int64_t tile_b_base = 0;
    int64_t tile_b_len = 0;
    double* dists = nullptr;      // [D] scratch
    double* cov_dyn_a = nullptr;  // [D * 8] aligned scratch
    double* cov_dyn_b = nullptr;  // [D * 8] aligned scratch
    float* lb_dyn = nullptr;      // [D * 8] aligned scratch
};

template <bool HasFlat>
inline void eval_mstump_exact_cell(
    const MstumpChannels& ch,
    const double* cov_cell,
    int64_t cov_stride,
    int64_t i,
    int64_t j,
    const double* invA_row,
    const double* flatA_row,
    MstumpWorkerBuf& wb
) {
    const int64_t D = ch.D;
    const int64_t n = ch.n;
    double* dists = wb.dists;
    int64_t c = 0;
#if defined(FMP_HAS_AVX2)
    const __m256d v_one = _mm256_set1_pd(1.0);
    const __m256d v_zero = _mm256_setzero_pd();
    const __m256d v_two = _mm256_set1_pd(2.0);
    const __m256d v_inf = _mm256_set1_pd(kPosInf);
    const __m256d v_near = _mm256_set1_pd(1e-10);
    for (; c + 4 <= D; c += 4) {
        __m256d vc = _mm256_set_pd(
            cov_cell[(c + 3) * cov_stride], cov_cell[(c + 2) * cov_stride],
            cov_cell[(c + 1) * cov_stride], cov_cell[(c + 0) * cov_stride]);
        __m256d via = invA_row
            ? _mm256_loadu_pd(invA_row + c)
            : _mm256_set_pd(ch.inv[c + 3][i], ch.inv[c + 2][i], ch.inv[c + 1][i], ch.inv[c + 0][i]);
        __m256d vib = _mm256_set_pd(
            ch.inv[c + 3][j], ch.inv[c + 2][j], ch.inv[c + 1][j], ch.inv[c + 0][j]);
        __m256d vu = _mm256_fnmadd_pd(vc, _mm256_mul_pd(via, vib), v_one);
        if constexpr (HasFlat) {
            __m256d vfa = flatA_row
                ? _mm256_loadu_pd(flatA_row + c)
                : _mm256_set_pd(ch.flat[c + 3][i], ch.flat[c + 2][i], ch.flat[c + 1][i], ch.flat[c + 0][i]);
            __m256d vfb = _mm256_set_pd(
                ch.flat[c + 3][j], ch.flat[c + 2][j], ch.flat[c + 1][j], ch.flat[c + 0][j]);
            vu = _mm256_sub_pd(_mm256_sub_pd(vu, vfa), vfb);
        }
        // Preserve NaN in src2 through clamps, then min(sqrt(u), +inf) maps NaN -> +inf!
        vu = _mm256_min_pd(v_two, _mm256_max_pd(v_zero, vu));
        __m256d vd = _mm256_min_pd(_mm256_sqrt_pd(vu), v_inf);
        _mm256_storeu_pd(dists + c, vd);
        int nm = _mm256_movemask_pd(_mm256_cmp_pd(vu, v_near, _CMP_LT_OQ));
        if (__builtin_expect(nm != 0, 0)) {
            while (nm) {
                int lane = __builtin_ctz(static_cast<unsigned>(nm));
                nm &= (nm - 1);
                int64_t cc = c + lane;
                if (!HasFlat || (ch.flat[cc][i] == 0.0 && ch.flat[cc][j] == 0.0)) {
                    dists[cc] = exact_unit_half_dist_scalar(ch.u[cc] + i, ch.u[cc] + j, ch.m);
                }
            }
        }
    }
#endif
    for (; c < D; ++c) {
        double inv_prod = (invA_row ? invA_row[c] : ch.inv[c][i]) * ch.inv[c][j];
        double u = std::fma(-cov_cell[c * cov_stride], inv_prod, 1.0);
        if constexpr (HasFlat) {
            double fa = flatA_row ? flatA_row[c] : ch.flat[c][i];
            u = (u - fa) - ch.flat[c][j];
        }
        if (std::isfinite(u)) {
            if (__builtin_expect(u < 1e-10, 0)) {
                if (!HasFlat || (ch.flat[c][i] == 0.0 && ch.flat[c][j] == 0.0)) {
                    dists[c] = exact_unit_half_dist_scalar(ch.u[c] + i, ch.u[c] + j, ch.m);
                } else {
                    dists[c] = 0.0;
                }
            } else {
                if (u > 2.0) u = 2.0;
                dists[c] = std::sqrt(u);
            }
        } else {
            dists[c] = kPosInf;
        }
    }
    sort_small_doubles(dists, D);
    double psum = 0.0;
    double* mp = wb.mp;
    int64_t* idx = wb.idx;
    float* row_mp_f = wb.row_mp_f;
    float* col_mp_f = wb.col_mp_f;
    float* col_mp_tile = wb.col_mp_tile;
    const int64_t n_pad_f = wb.n_pad_f;
    const int64_t b_base = wb.tile_b_base;
    const int64_t b_end = b_base + wb.tile_b_len;
    const bool i_in_tile = (col_mp_tile != nullptr && i >= b_base && i < b_end);
    const bool j_in_tile = (col_mp_tile != nullptr && j >= b_base && j < b_end);
    for (int64_t k = 0; k < D; ++k) {
        psum += dists[k];
        int64_t off_i = k * n + i;
        if (psum < mp[off_i]) {
            mp[off_i] = psum;
            idx[off_i] = j;
            if (row_mp_f) {
                float ub = upper_bound_f32(psum);
                row_mp_f[i * D + k] = ub;
                col_mp_f[k * n_pad_f + i] = ub;
                if (i_in_tile) col_mp_tile[k * kMsBStride + (i - b_base)] = ub;
            }
        }
        int64_t off_j = k * n + j;
        if (psum < mp[off_j]) {
            mp[off_j] = psum;
            idx[off_j] = i;
            if (row_mp_f) {
                float ub = upper_bound_f32(psum);
                row_mp_f[j * D + k] = ub;
                col_mp_f[k * n_pad_f + j] = ub;
                if (j_in_tile) col_mp_tile[k * kMsBStride + (j - b_base)] = ub;
            }
        }
    }
}

template <bool HasFlat>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline, cold))
#endif
void eval_mstump_exact_hit_mask(
    const MstumpChannels& ch,
    const double* cov_cur,
    int64_t i,
    int64_t col,
    int hit_mask,
    const double* invA_row,
    const double* flatA_row,
    MstumpWorkerBuf& wb
) {
    while (hit_mask) {
        int lane = __builtin_ctz(static_cast<unsigned>(hit_mask));
        hit_mask &= (hit_mask - 1);
        eval_mstump_exact_cell<HasFlat>(
            ch, cov_cur + lane, 8, i, col + lane, invA_row, flatA_row, wb);
    }
}

template <bool HasFlat>
inline void eval_mstump_diag_scalar(
    const MstumpChannels& ch,
    int64_t i0, int64_t j0, int64_t rows,
    MstumpWorkerBuf& wb
) {
    const int64_t D = ch.D;
    double* cov = wb.cov_dyn_a;
    for (int64_t c = 0; c < D; ++c) {
        cov[c] = exact_cov_scalar(
            ch.u[c] + i0, ch.mu[c][i0],
            ch.u[c] + j0, ch.mu[c][j0],
            ch.m);
    }
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t i = i0 + r;
        const int64_t j = j0 + r;
        if (r > 0 && ch.is_drop) {
            for (int64_t c = 0; c < D; ++c) {
                if (ch.is_drop[c] && (ch.is_drop[c][i] || ch.is_drop[c][j])) {
                    cov[c] = exact_cov_scalar(
                        ch.u[c] + i, ch.mu[c][i],
                        ch.u[c] + j, ch.mu[c][j],
                        ch.m);
                }
            }
        }
        eval_mstump_exact_cell<HasFlat>(ch, cov, 1, i, j, nullptr, nullptr, wb);
        if (r + 1 < rows) {
            for (int64_t c = 0; c < D; ++c) {
                cov[c] += ch.df[c][i] * ch.dg[c][j] + ch.dg[c][i] * ch.df[c][j];
            }
        }
    }
}

#if defined(FMP_HAS_AVX2)
// Two-stage exact AVX2 8-diagonal mSTUMP microkernel (v3):
// - Packed B_tile (compile-time constant stride kMsBStride = 264): zero pointer chasing, zero 4K aliasing
// - Ping-pong cov_cur/cov_nxt: reads cov ONCE per row, fuses cov recurrence + float32 lower-bound filter
// - CleanF32 fast path (!HasFlat && !HasNaN && safe_f32): omits redundant NaN/upper clamps (saves 16 p1 ops/row!)
// - Cold noinline exact verifier using AVX2 _mm256_sqrt_pd across channels
template <int FixedD, bool HasFlat, bool CleanF32>
inline void eval_mstump_strip_8_avx2(
    const MstumpChannels& ch,
    int64_t i0, int64_t j0, int64_t rows,
    int64_t b_off0,
    const double* __restrict__ invA_tile,
    const float* __restrict__ invA_f_tile,
    const double* __restrict__ dfA_tile,
    const double* __restrict__ dgA_tile,
    const double* __restrict__ flatA_tile,
    const double* __restrict__ dfB_tile,
    const double* __restrict__ dgB_tile,
    const double* __restrict__ invB_tile,
    const float* __restrict__ invB_f_tile,
    const double* __restrict__ flatB_tile,
    MstumpWorkerBuf& wb
) {
    const int64_t D_val = (FixedD > 0) ? FixedD : ch.D;
    float* row_mp_f = wb.row_mp_f;
    float* col_mp_tile = wb.col_mp_tile;

    alignas(32) double cov_stack_a[(FixedD > 0 ? FixedD : 1) * 8];
    alignas(32) double cov_stack_b[(FixedD > 0 ? FixedD : 1) * 8];
    double* cov_cur = (FixedD > 0) ? cov_stack_a : wb.cov_dyn_a;
    double* cov_nxt = (FixedD > 0) ? cov_stack_b : wb.cov_dyn_b;

    for (int64_t c = 0; c < D_val; ++c) {
        exact_cov_8(ch.u[c] + i0, ch.mu[c][i0], ch.u[c], ch.mu[c], j0, ch.m, cov_cur + c * 8);
    }

    const __m256d v_one = _mm256_set1_pd(1.0);
    const __m256d v_zero = _mm256_setzero_pd();
    const __m256 vf_zero = _mm256_setzero_ps();
    const __m256 vf_one_eps = _mm256_set1_ps(0.99999f);
    const __m256 vf_two = _mm256_set1_ps(2.0f);
    const __m256 vf_tiny = _mm256_set1_ps(1e-34f);
    const float lb_s = (D_val <= 1024) ? 0.998f : ((D_val <= 8192) ? 0.99f : 0.95f);
    const __m256 vf_lb_scale = _mm256_set1_ps(lb_s);
    const __m256 vf_inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());

    (void)v_one; (void)v_zero; (void)vf_zero; (void)vf_one_eps; (void)vf_two; (void)vf_inf;
    (void)invA_f_tile; (void)invB_tile; (void)invB_f_tile; (void)flatB_tile;

    auto cmp_swap_8f = [](__m256& a, __m256& b) {
        __m256 lo = _mm256_min_ps(a, b);
        __m256 hi = _mm256_max_ps(a, b);
        a = lo;
        b = hi;
    };

    for (int64_t r = 0; r < rows; ++r) {
        const int64_t i = i0 + r;
        const int64_t col = j0 + r;
        const int64_t boff = b_off0 + r;
        const double* dfA_row = dfA_tile + r * D_val;
        const double* dgA_row = dgA_tile + r * D_val;
        const float* row_f = row_mp_f + i * D_val;

        if constexpr (FixedD > 0) {
            __m256 lb[FixedD];
            for (int64_t c = 0; c < FixedD; ++c) {
                __m256d c0 = _mm256_load_pd(cov_cur + c * 8);
                __m256d c1 = _mm256_load_pd(cov_cur + c * 8 + 4);

                __m256 lb_c;
                if constexpr (CleanF32) {
                    __m256 cf = _mm256_insertf128_ps(
                        _mm256_castps128_ps256(_mm256_cvtpd_ps(c0)),
                        _mm256_cvtpd_ps(c1), 1);
                    __m256 v_ia = _mm256_set1_ps(invA_f_tile[r * FixedD + c]);
                    __m256 inv_p = _mm256_mul_ps(_mm256_loadu_ps(invB_f_tile + c * kMsBStride + boff), v_ia);
                    __m256 uf = _mm256_max_ps(vf_zero, _mm256_fnmadd_ps(cf, inv_p, vf_one_eps));
                    __m256 yf = _mm256_rsqrt_ps(_mm256_max_ps(vf_tiny, uf));
                    lb_c = _mm256_mul_ps(_mm256_mul_ps(uf, vf_lb_scale), yf);
                } else {
                    __m256d v_ia = _mm256_set1_pd(invA_tile[r * FixedD + c]);
                    const double* invB_c = invB_tile + c * kMsBStride + boff;
                    __m256d u0 = _mm256_fnmadd_pd(c0, _mm256_mul_pd(_mm256_loadu_pd(invB_c), v_ia), v_one);
                    __m256d u1 = _mm256_fnmadd_pd(c1, _mm256_mul_pd(_mm256_loadu_pd(invB_c + 4), v_ia), v_one);
                    if constexpr (HasFlat) {
                        __m256d v_fa = _mm256_set1_pd(flatA_tile[r * FixedD + c]);
                        const double* fB_c = flatB_tile + c * kMsBStride + boff;
                        u0 = _mm256_sub_pd(_mm256_sub_pd(u0, v_fa), _mm256_loadu_pd(fB_c));
                        u1 = _mm256_sub_pd(_mm256_sub_pd(u1, v_fa), _mm256_loadu_pd(fB_c + 4));
                    }
                    u0 = _mm256_max_pd(v_zero, u0);
                    u1 = _mm256_max_pd(v_zero, u1);
                    __m256 uf = _mm256_insertf128_ps(
                        _mm256_castps128_ps256(_mm256_cvtpd_ps(u0)),
                        _mm256_cvtpd_ps(u1), 1);
                    uf = _mm256_min_ps(vf_two, uf);
                    __m256 yf = _mm256_rsqrt_ps(_mm256_max_ps(vf_tiny, uf));
                    lb_c = _mm256_min_ps(_mm256_mul_ps(_mm256_mul_ps(uf, vf_lb_scale), yf), vf_inf);
                }
                lb[c] = lb_c;

                // Fused recurrence advance while c0, c1 are already in YMM registers
                __m256d v_dfi = _mm256_set1_pd(dfA_row[c]);
                __m256d v_dgi = _mm256_set1_pd(dgA_row[c]);
                const double* dfB_c = dfB_tile + c * kMsBStride + boff;
                const double* dgB_c = dgB_tile + c * kMsBStride + boff;
                __m256d n0 = _mm256_fmadd_pd(v_dfi, _mm256_loadu_pd(dgB_c),
                             _mm256_fmadd_pd(v_dgi, _mm256_loadu_pd(dfB_c), c0));
                __m256d n1 = _mm256_fmadd_pd(v_dfi, _mm256_loadu_pd(dgB_c + 4),
                             _mm256_fmadd_pd(v_dgi, _mm256_loadu_pd(dfB_c + 4), c1));
                _mm256_store_pd(cov_nxt + c * 8, n0);
                _mm256_store_pd(cov_nxt + c * 8 + 4, n1);
            }

            sort_network_fixed<FixedD>(lb, cmp_swap_8f);

            __m256 psum = lb[0];
            __m256 any_hit = _mm256_or_ps(
                _mm256_cmp_ps(psum, _mm256_set1_ps(row_f[0]), _CMP_LT_OQ),
                _mm256_cmp_ps(psum, _mm256_loadu_ps(col_mp_tile + boff), _CMP_LT_OQ));
            for (int64_t k = 1; k < FixedD; ++k) {
                psum = _mm256_add_ps(psum, lb[k]);
                __m256 mr = _mm256_cmp_ps(psum, _mm256_set1_ps(row_f[k]), _CMP_LT_OQ);
                __m256 mc = _mm256_cmp_ps(psum, _mm256_loadu_ps(col_mp_tile + k * kMsBStride + boff), _CMP_LT_OQ);
                any_hit = _mm256_or_ps(any_hit, _mm256_or_ps(mr, mc));
            }

            int hit_mask = _mm256_movemask_ps(any_hit);
            if (__builtin_expect(hit_mask != 0, 0)) {
                eval_mstump_exact_hit_mask<HasFlat>(
                    ch, cov_cur, i, col, hit_mask,
                    invA_tile + r * FixedD,
                    HasFlat ? (flatA_tile + r * FixedD) : nullptr, wb);
            }
        } else {
            float* lb_arr = wb.lb_dyn;
            for (int64_t c = 0; c < D_val; ++c) {
                __m256d c0 = _mm256_load_pd(cov_cur + c * 8);
                __m256d c1 = _mm256_load_pd(cov_cur + c * 8 + 4);

                __m256 lb_c;
                if constexpr (CleanF32) {
                    __m256 cf = _mm256_insertf128_ps(
                        _mm256_castps128_ps256(_mm256_cvtpd_ps(c0)),
                        _mm256_cvtpd_ps(c1), 1);
                    __m256 v_ia = _mm256_set1_ps(invA_f_tile[r * D_val + c]);
                    __m256 inv_p = _mm256_mul_ps(_mm256_loadu_ps(invB_f_tile + c * kMsBStride + boff), v_ia);
                    __m256 uf = _mm256_max_ps(vf_zero, _mm256_fnmadd_ps(cf, inv_p, vf_one_eps));
                    __m256 yf = _mm256_rsqrt_ps(_mm256_max_ps(vf_tiny, uf));
                    lb_c = _mm256_mul_ps(_mm256_mul_ps(uf, vf_lb_scale), yf);
                } else {
                    __m256d v_ia = _mm256_set1_pd(invA_tile[r * D_val + c]);
                    const double* invB_c = invB_tile + c * kMsBStride + boff;
                    __m256d u0 = _mm256_fnmadd_pd(c0, _mm256_mul_pd(_mm256_loadu_pd(invB_c), v_ia), v_one);
                    __m256d u1 = _mm256_fnmadd_pd(c1, _mm256_mul_pd(_mm256_loadu_pd(invB_c + 4), v_ia), v_one);
                    if constexpr (HasFlat) {
                        __m256d v_fa = _mm256_set1_pd(flatA_tile[r * D_val + c]);
                        const double* fB_c = flatB_tile + c * kMsBStride + boff;
                        u0 = _mm256_sub_pd(_mm256_sub_pd(u0, v_fa), _mm256_loadu_pd(fB_c));
                        u1 = _mm256_sub_pd(_mm256_sub_pd(u1, v_fa), _mm256_loadu_pd(fB_c + 4));
                    }
                    u0 = _mm256_max_pd(v_zero, u0);
                    u1 = _mm256_max_pd(v_zero, u1);
                    __m256 uf = _mm256_insertf128_ps(
                        _mm256_castps128_ps256(_mm256_cvtpd_ps(u0)),
                        _mm256_cvtpd_ps(u1), 1);
                    uf = _mm256_min_ps(vf_two, uf);
                    __m256 yf = _mm256_rsqrt_ps(_mm256_max_ps(vf_tiny, uf));
                    lb_c = _mm256_min_ps(_mm256_mul_ps(_mm256_mul_ps(uf, vf_lb_scale), yf), vf_inf);
                }
                _mm256_storeu_ps(lb_arr + c * 8, lb_c);

                __m256d v_dfi = _mm256_set1_pd(dfA_row[c]);
                __m256d v_dgi = _mm256_set1_pd(dgA_row[c]);
                const double* dfB_c = dfB_tile + c * kMsBStride + boff;
                const double* dgB_c = dgB_tile + c * kMsBStride + boff;
                __m256d n0 = _mm256_fmadd_pd(v_dfi, _mm256_loadu_pd(dgB_c),
                             _mm256_fmadd_pd(v_dgi, _mm256_loadu_pd(dfB_c), c0));
                __m256d n1 = _mm256_fmadd_pd(v_dfi, _mm256_loadu_pd(dgB_c + 4),
                             _mm256_fmadd_pd(v_dgi, _mm256_loadu_pd(dfB_c + 4), c1));
                _mm256_store_pd(cov_nxt + c * 8, n0);
                _mm256_store_pd(cov_nxt + c * 8 + 4, n1);
            }

            // Iterative Batcher odd-even merge sort on 8-wide float vectors in lb_arr
            for (int64_t p = 1; p < D_val; p <<= 1) {
                const int64_t blk_mask = ~((p << 1) - 1);
                for (int64_t k = p; k >= 1; k >>= 1) {
                    for (int64_t j = k % p; j + k < D_val; j += (k << 1)) {
                        int64_t lim = std::min(k, D_val - j - k);
                        for (int64_t ii = 0; ii < lim; ++ii) {
                            int64_t a = ii + j;
                            int64_t b = a + k;
                            if (((a ^ b) & blk_mask) == 0) {
                                __m256 va = _mm256_loadu_ps(lb_arr + a * 8);
                                __m256 vb = _mm256_loadu_ps(lb_arr + b * 8);
                                _mm256_storeu_ps(lb_arr + a * 8, _mm256_min_ps(va, vb));
                                _mm256_storeu_ps(lb_arr + b * 8, _mm256_max_ps(va, vb));
                            }
                        }
                    }
                }
            }

            __m256 psum = _mm256_loadu_ps(lb_arr);
            __m256 any_hit = _mm256_or_ps(
                _mm256_cmp_ps(psum, _mm256_set1_ps(row_f[0]), _CMP_LT_OQ),
                _mm256_cmp_ps(psum, _mm256_loadu_ps(col_mp_tile + boff), _CMP_LT_OQ));
            for (int64_t k = 1; k < D_val; ++k) {
                psum = _mm256_add_ps(psum, _mm256_loadu_ps(lb_arr + k * 8));
                __m256 mr = _mm256_cmp_ps(psum, _mm256_set1_ps(row_f[k]), _CMP_LT_OQ);
                __m256 mc = _mm256_cmp_ps(psum, _mm256_loadu_ps(col_mp_tile + k * kMsBStride + boff), _CMP_LT_OQ);
                any_hit = _mm256_or_ps(any_hit, _mm256_or_ps(mr, mc));
            }

            int hit_mask = _mm256_movemask_ps(any_hit);
            if (__builtin_expect(hit_mask != 0, 0)) {
                eval_mstump_exact_hit_mask<HasFlat>(
                    ch, cov_cur, i, col, hit_mask,
                    invA_tile + r * D_val,
                    HasFlat ? (flatA_tile + r * D_val) : nullptr, wb);
            }
        }

        std::swap(cov_cur, cov_nxt);
    }
}
#endif

template <bool HasFlat>
inline void eval_mstump_strip_8_generic(
    const MstumpChannels& ch,
    int64_t i0, int64_t j0, int64_t rows,
    const double* invA_tile,
    const double* dfA_tile,
    const double* dgA_tile,
    const double* flatA_tile,
    MstumpWorkerBuf& wb
) {
    const int64_t D = ch.D;
    double* cov_buf = wb.cov_dyn_a;
    for (int64_t c = 0; c < D; ++c) {
        exact_cov_8(ch.u[c] + i0, ch.mu[c][i0], ch.u[c], ch.mu[c], j0, ch.m, cov_buf + c * 8);
    }
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t i = i0 + r;
        const int64_t col = j0 + r;
        const double* invA_row = invA_tile + r * D;
        const double* flatA_row = HasFlat ? (flatA_tile + r * D) : nullptr;
        for (int64_t lane = 0; lane < 8; ++lane) {
            eval_mstump_exact_cell<HasFlat>(
                ch, cov_buf + lane, 8, i, col + lane, invA_row, flatA_row, wb);
        }
        const double* dfA_row = dfA_tile + r * D;
        const double* dgA_row = dgA_tile + r * D;
        for (int64_t c = 0; c < D; ++c) {
            const double dfi = dfA_row[c];
            const double dgi = dgA_row[c];
            const double* dfB_c = ch.df[c] + col;
            const double* dgB_c = ch.dg[c] + col;
            double* cv = cov_buf + c * 8;
            for (int64_t lane = 0; lane < 8; ++lane) {
                cv[lane] += dfi * dgB_c[lane] + dgi * dfB_c[lane];
            }
        }
    }
}

template <int FixedD, bool HasFlat>
void run_mstump_worker_dynamic(
    const MstumpChannels& ch,
    std::atomic<int64_t>& next_db,
    int64_t diag_max,
    double* mp_buf,
    int64_t* idx_buf
) {
    const int64_t D = (FixedD > 0) ? FixedD : ch.D;
    const int64_t n = ch.n;
    const int64_t n_pad_f = ((n + 31) & ~31LL) + 16;

    std::vector<double> dists(static_cast<size_t>(D), 0.0);
    std::vector<double> cov_raw_a(static_cast<size_t>(D * 8 + 8), 0.0);
    std::vector<double> cov_raw_b(static_cast<size_t>(D * 8 + 8), 0.0);
    uintptr_t ca = reinterpret_cast<uintptr_t>(cov_raw_a.data());
    uintptr_t cb = reinterpret_cast<uintptr_t>(cov_raw_b.data());
    double* cov_aligned_a = reinterpret_cast<double*>((ca + 31u) & ~uintptr_t(31u));
    double* cov_aligned_b = reinterpret_cast<double*>((cb + 31u) & ~uintptr_t(31u));

#if defined(FMP_HAS_AVX2)
    const float kPosInfF = std::numeric_limits<float>::infinity();
    std::vector<float> row_mp_f(static_cast<size_t>(n * D), kPosInfF);
    std::vector<float> col_mp_f(static_cast<size_t>(D * n_pad_f), kPosInfF);
    std::vector<float> col_mp_tile(static_cast<size_t>(D * kMsBStride), kPosInfF);
    std::vector<float> lb_raw(static_cast<size_t>(D * 8 + 16), 0.0f);
    uintptr_t lb_addr = reinterpret_cast<uintptr_t>(lb_raw.data());
    float* lb_aligned = reinterpret_cast<float*>((lb_addr + 31u) & ~uintptr_t(31u));

    std::vector<float> invA_f_tile(static_cast<size_t>(ch.clean_f32 ? (kMsTileH * D) : 1));
    std::vector<double> dfB_tile(static_cast<size_t>(D * kMsBStride));
    std::vector<double> dgB_tile(static_cast<size_t>(D * kMsBStride));
    std::vector<double> invB_tile(static_cast<size_t>(ch.clean_f32 ? 1 : (D * kMsBStride)));
    std::vector<float> invB_f_tile(static_cast<size_t>(ch.clean_f32 ? (D * kMsBStride) : 1));
    std::vector<double> flatB_tile(static_cast<size_t>(HasFlat ? (D * kMsBStride) : 1));
#endif

    MstumpWorkerBuf wb;
    wb.mp = mp_buf;
    wb.idx = idx_buf;
#if defined(FMP_HAS_AVX2)
    wb.row_mp_f = row_mp_f.data();
    wb.col_mp_f = col_mp_f.data();
    wb.col_mp_tile = col_mp_tile.data();
    wb.n_pad_f = n_pad_f;
    wb.lb_dyn = lb_aligned;
#else
    wb.row_mp_f = nullptr;
    wb.col_mp_f = nullptr;
    wb.col_mp_tile = nullptr;
    wb.n_pad_f = n_pad_f;
    wb.lb_dyn = nullptr;
#endif
    wb.dists = dists.data();
    wb.cov_dyn_a = cov_aligned_a;
    wb.cov_dyn_b = cov_aligned_b;

    std::vector<double> invA_tile(static_cast<size_t>(kMsTileH * D));
    std::vector<double> dfA_tile(static_cast<size_t>(kMsTileH * D));
    std::vector<double> dgA_tile(static_cast<size_t>(kMsTileH * D));
    std::vector<double> flatA_tile(static_cast<size_t>(HasFlat ? (kMsTileH * D) : 1));

    while (true) {
        int64_t db = next_db.fetch_add(kMsTileW, std::memory_order_relaxed);
        if (db >= diag_max) break;
        int64_t db_end = std::min(diag_max, db + kMsTileW);
        int64_t i_max = n - db;
        if (i_max <= 0) continue;

        for (int64_t ti = 0; ti < i_max; ti += kMsTileH) {
            int64_t ti_end = std::min(i_max, ti + kMsTileH);
            int64_t tile_rows = ti_end - ti;

            for (int64_t r = 0; r < tile_rows; ++r) {
                int64_t i = ti + r;
                double* inv_dst = invA_tile.data() + r * D;
                double* df_dst = dfA_tile.data() + r * D;
                double* dg_dst = dgA_tile.data() + r * D;
                for (int64_t c = 0; c < D; ++c) {
                    double iv = ch.inv[c][i];
                    inv_dst[c] = iv;
                    df_dst[c] = ch.df[c][i];
                    dg_dst[c] = ch.dg[c][i];
#if defined(FMP_HAS_AVX2)
                    if (ch.clean_f32) {
                        invA_f_tile[static_cast<size_t>(r * D + c)] = static_cast<float>(iv);
                    }
#endif
                }
                if constexpr (HasFlat) {
                    double* f_dst = flatA_tile.data() + r * D;
                    for (int64_t c = 0; c < D; ++c) {
                        f_dst[c] = ch.flat[c][i];
                    }
                }
            }

#if defined(FMP_HAS_AVX2)
            int64_t b_base = ti + db;
            int64_t b_end = std::min(n, ti_end + db_end - 1);
            int64_t b_len = b_end > b_base ? (b_end - b_base) : 0;
            wb.tile_b_base = b_base;
            wb.tile_b_len = b_len;
            for (int64_t c = 0; c < D; ++c) {
                std::memcpy(dfB_tile.data() + c * kMsBStride, ch.df[c] + b_base,
                            static_cast<size_t>(b_len) * sizeof(double));
                std::memcpy(dgB_tile.data() + c * kMsBStride, ch.dg[c] + b_base,
                            static_cast<size_t>(b_len) * sizeof(double));
                std::memcpy(col_mp_tile.data() + c * kMsBStride, col_mp_f.data() + c * n_pad_f + b_base,
                            static_cast<size_t>(b_len) * sizeof(float));
                if (ch.clean_f32) {
                    const double* src = ch.inv[c] + b_base;
                    float* dst = invB_f_tile.data() + c * kMsBStride;
                    for (int64_t q = 0; q < b_len; ++q) dst[q] = static_cast<float>(src[q]);
                } else {
                    std::memcpy(invB_tile.data() + c * kMsBStride, ch.inv[c] + b_base,
                                static_cast<size_t>(b_len) * sizeof(double));
                    if constexpr (HasFlat) {
                        std::memcpy(flatB_tile.data() + c * kMsBStride, ch.flat[c] + b_base,
                                    static_cast<size_t>(b_len) * sizeof(double));
                    }
                }
            }
#endif

            int64_t d = db;
            while (d < db_end) {
                if (d + 8 <= db_end) {
                    int64_t i_stop = std::min(ti_end, n - (d + 7));
                    if (i_stop - ti >= 16) {
                        bool strip_has_drop = false;
                        if (__builtin_expect(ch.any_drop_prefix != nullptr, 0)) {
                            int64_t j_lo = ti + d;
                            int64_t j_hi = std::min(n, i_stop + d + 7);
                            if (ch.any_drop_prefix[i_stop] > ch.any_drop_prefix[ti] ||
                                ch.any_drop_prefix[j_hi] > ch.any_drop_prefix[j_lo]) {
                                strip_has_drop = true;
                            }
                        }
                        if (__builtin_expect(strip_has_drop, 0)) {
                            for (int64_t dd = d; dd < d + 8; ++dd) {
                                int64_t di_hi = std::min(ti_end, n - dd);
                                if (ti < di_hi) {
                                    eval_mstump_diag_scalar<HasFlat>(
                                        ch, ti, ti + dd, di_hi - ti, wb);
                                }
                            }
                            d += 8;
                            continue;
                        }
#if defined(FMP_HAS_AVX2)
                        int64_t b_off0 = d - db;
                        if constexpr (!HasFlat) {
                            if (ch.clean_f32) {
                                eval_mstump_strip_8_avx2<FixedD, false, true>(
                                    ch, ti, ti + d, i_stop - ti, b_off0,
                                    invA_tile.data(), invA_f_tile.data(),
                                    dfA_tile.data(), dgA_tile.data(),
                                    nullptr,
                                    dfB_tile.data(), dgB_tile.data(),
                                    nullptr, invB_f_tile.data(),
                                    nullptr, wb);
                            } else {
                                eval_mstump_strip_8_avx2<FixedD, false, false>(
                                    ch, ti, ti + d, i_stop - ti, b_off0,
                                    invA_tile.data(), nullptr,
                                    dfA_tile.data(), dgA_tile.data(),
                                    nullptr,
                                    dfB_tile.data(), dgB_tile.data(),
                                    invB_tile.data(), nullptr,
                                    nullptr, wb);
                            }
                        } else {
                            eval_mstump_strip_8_avx2<FixedD, true, false>(
                                ch, ti, ti + d, i_stop - ti, b_off0,
                                invA_tile.data(), nullptr,
                                dfA_tile.data(), dgA_tile.data(),
                                flatA_tile.data(),
                                dfB_tile.data(), dgB_tile.data(),
                                invB_tile.data(), nullptr,
                                flatB_tile.data(), wb);
                        }
#else
                        eval_mstump_strip_8_generic<HasFlat>(
                            ch, ti, ti + d, i_stop - ti,
                            invA_tile.data(), dfA_tile.data(), dgA_tile.data(),
                            HasFlat ? flatA_tile.data() : nullptr, wb);
#endif
                        for (int64_t dd = d; dd < d + 8; ++dd) {
                            int64_t di_hi = std::min(ti_end, n - dd);
                            if (di_hi > i_stop) {
                                eval_mstump_diag_scalar<HasFlat>(
                                    ch, i_stop, i_stop + dd, di_hi - i_stop, wb);
                            }
                        }
                        d += 8;
                        continue;
                    }
                }
                int64_t di_hi = std::min(ti_end, n - d);
                if (ti < di_hi) {
                    eval_mstump_diag_scalar<HasFlat>(ch, ti, ti + d, di_hi - ti, wb);
                }
                ++d;
            }
        }
    }
}

template <bool HasFlat>
void dispatch_mstump_worker(
    const MstumpChannels& ch,
    std::atomic<int64_t>& next_db,
    int64_t diag_max,
    double* mp_buf,
    int64_t* idx_buf
) {
    switch (ch.D) {
        case 1: run_mstump_worker_dynamic<1, HasFlat>(ch, next_db, diag_max, mp_buf, idx_buf); break;
        case 2: run_mstump_worker_dynamic<2, HasFlat>(ch, next_db, diag_max, mp_buf, idx_buf); break;
        case 3: run_mstump_worker_dynamic<3, HasFlat>(ch, next_db, diag_max, mp_buf, idx_buf); break;
        case 4: run_mstump_worker_dynamic<4, HasFlat>(ch, next_db, diag_max, mp_buf, idx_buf); break;
        case 5: run_mstump_worker_dynamic<5, HasFlat>(ch, next_db, diag_max, mp_buf, idx_buf); break;
        case 6: run_mstump_worker_dynamic<6, HasFlat>(ch, next_db, diag_max, mp_buf, idx_buf); break;
        case 7: run_mstump_worker_dynamic<7, HasFlat>(ch, next_db, diag_max, mp_buf, idx_buf); break;
        case 8: run_mstump_worker_dynamic<8, HasFlat>(ch, next_db, diag_max, mp_buf, idx_buf); break;
        case 12: run_mstump_worker_dynamic<12, HasFlat>(ch, next_db, diag_max, mp_buf, idx_buf); break;
        case 16: run_mstump_worker_dynamic<16, HasFlat>(ch, next_db, diag_max, mp_buf, idx_buf); break;
        default: run_mstump_worker_dynamic<0, HasFlat>(ch, next_db, diag_max, mp_buf, idx_buf); break;
    }
}

// ---------------------------------------------------------------------------
// RingMP Internal Implementation
// ---------------------------------------------------------------------------
constexpr int kStackCap = 64;

struct StackEntry {
    int64_t idx;
    double corr;
};

} // namespace

struct fmp_stream {
    int64_t m = 0;
    int64_t W = 0;           // Max active subsequences
    int64_t ez = 0;          // Exclusion zone (valid pairs satisfy |i-j| > ez)
    int egress = 1;
    double alpha = 0.05;
    int track_full_mp = 1;

    int64_t cap = 0;         // Power of 2 >= W + 4 (for subsequence ring)
    int64_t mask = 0;
    int64_t slab_stride = 0; // cap + 64 (staggered 512B mod 4096B -> zero 4K aliasing!)
    int64_t raw_cap = 0;     // Power of 2 >= W + m + 4 (mirrored 2*raw_cap)
    int64_t raw_mask = 0;

    int64_t n_raw = 0;       // Total raw ticks ingested
    int64_t n_sub = 0;       // Total subsequences formed (= max(0, n_raw - m + 1))
    double base = 0.0;       // DC anchor set on first finite tick
    int has_base = 0;

    // Mirrored raw ring buffers (length 2 * raw_cap)
    std::vector<double> u_ring;
    std::vector<uint8_t> bad_ring;
    int64_t bad_in_cur_win = 0;

    // Single staggered double slab (7 * slab_stride) & int64 slab (2 * slab_stride)
    // Eliminates Haswell 4K-aliasing stalls & L1d set conflicts across all hot ring streams.
    std::vector<double> d_slab;
    std::vector<int64_t> i_slab;
    double* cov = nullptr;           // In-place covariance vector cov(gid_cur, k)
    double* df = nullptr;
    double* dg = nullptr;
    double* inv_norm = nullptr;
    double* right_corr = nullptr;
    double* mu = nullptr;
    double* left_top_corr = nullptr;
    int64_t* right_idx = nullptr;
    int64_t* left_top_idx = nullptr;

    std::vector<uint8_t> valid;

    // Left-MP Monotone Suffix-Max Stack per active subsequence (length cap * kStackCap)
    // Bottom (0) = rightmost/newest record-holder, top (len-1) = leftmost/highest record-holder.
    std::vector<StackEntry> left_stacks;
    std::vector<uint8_t> stack_len;
    std::vector<uint8_t> stack_trunc;

    // Causal calibration ring + m-tick delay buffer for conformal p-value & exchangeable e-CUSUM
    static constexpr int64_t kCalCap = 256;
    alignas(32) double cal_ring[kCalCap] = {};
    int64_t cal_count = 0;
    int64_t cal_head = 0;
    std::vector<double> cal_delay;
    int64_t cal_delay_cnt = 0;
    int64_t cal_delay_head = 0;
    double e_cusum[3] = {1.0, 1.0, 1.0};
    double e_decay[3] = {1.0, 1.0, 1.0};
    double e_wealth = 1.0;

    // Rolling Welford stats on u_ring for m > 256 fast path
    double last_sum = 0.0;
    double last_var = 0.0;
    double last_peak_var = 0.0;

    // Spike-drop covariance cancellation guard (zero cost when active_drops is empty)
    double norm_hwm = 0.0;
    uint8_t prev_valid = 1;
    std::vector<int64_t> active_drops;
    std::vector<int64_t> split_scratch;

    void bind_slab_ptrs() {
        double* db = d_slab.data();
        cov = db + 0 * slab_stride;
        df = db + 1 * slab_stride;
        dg = db + 2 * slab_stride;
        inv_norm = db + 3 * slab_stride;
        right_corr = db + 4 * slab_stride;
        mu = db + 5 * slab_stride;
        left_top_corr = db + 6 * slab_stride;
        int64_t* ib = i_slab.data();
        right_idx = ib + 0 * slab_stride;
        left_top_idx = ib + 1 * slab_stride;
    }

    void init_slabs(int64_t c) {
        cap = c;
        mask = c - 1;
        slab_stride = c + 64;
        d_slab.assign(static_cast<size_t>(7 * slab_stride), 0.0);
        i_slab.assign(static_cast<size_t>(2 * slab_stride), -1);
        bind_slab_ptrs();
        std::fill_n(inv_norm, c, kQuietNaN);
        std::fill_n(right_corr, c, kNegInf);
        std::fill_n(left_top_corr, c, kNegInf);
        valid.assign(static_cast<size_t>(c), 0);
        if (track_full_mp && egress) {
            left_stacks.resize(static_cast<size_t>(c * kStackCap));
        } else {
            left_stacks.clear();
        }
        stack_len.assign(static_cast<size_t>(c), 0);
        stack_trunc.assign(static_cast<size_t>(c), 0);
    }

    void grow_subseq_capacity(int64_t g) {
        int64_t new_cap = cap * 2;
        int64_t new_mask = new_cap - 1;
        int64_t new_stride = new_cap + 64;
        std::vector<double> new_d(static_cast<size_t>(7 * new_stride), 0.0);
        std::vector<int64_t> new_i(static_cast<size_t>(2 * new_stride), -1);

        double* n_cov = new_d.data() + 0 * new_stride;
        double* n_df = new_d.data() + 1 * new_stride;
        double* n_dg = new_d.data() + 2 * new_stride;
        double* n_inv = new_d.data() + 3 * new_stride;
        double* n_rc = new_d.data() + 4 * new_stride;
        double* n_mu = new_d.data() + 5 * new_stride;
        double* n_lc = new_d.data() + 6 * new_stride;
        int64_t* n_ri = new_i.data() + 0 * new_stride;
        int64_t* n_li = new_i.data() + 1 * new_stride;

        std::fill_n(n_inv, new_cap, kQuietNaN);
        std::fill_n(n_rc, new_cap, kNegInf);
        std::fill_n(n_lc, new_cap, kNegInf);

        std::vector<uint8_t> new_valid(static_cast<size_t>(new_cap), 0);
        std::vector<uint8_t> new_slen(static_cast<size_t>(new_cap), 0);
        std::vector<uint8_t> new_strunc(static_cast<size_t>(new_cap), 0);
        std::vector<StackEntry> new_stacks;
        const bool has_stacks = (track_full_mp && egress);
        if (has_stacks) {
            new_stacks.resize(static_cast<size_t>(new_cap * kStackCap));
        }

        for (int64_t k = 0; k < g; ++k) {
            int64_t os = k & mask;
            int64_t ns = k & new_mask;
            n_cov[ns] = cov[os];
            n_df[ns] = df[os];
            n_dg[ns] = dg[os];
            n_inv[ns] = inv_norm[os];
            n_rc[ns] = right_corr[os];
            n_mu[ns] = mu[os];
            n_lc[ns] = left_top_corr[os];
            n_ri[ns] = right_idx[os];
            n_li[ns] = left_top_idx[os];
            new_valid[static_cast<size_t>(ns)] = valid[static_cast<size_t>(os)];
            new_slen[static_cast<size_t>(ns)] = stack_len[static_cast<size_t>(os)];
            new_strunc[static_cast<size_t>(ns)] = stack_trunc[static_cast<size_t>(os)];
            if (has_stacks && stack_len[static_cast<size_t>(os)] > 0) {
                std::memcpy(new_stacks.data() + ns * kStackCap,
                            left_stacks.data() + os * kStackCap,
                            sizeof(StackEntry) * stack_len[static_cast<size_t>(os)]);
            }
        }

        d_slab.swap(new_d);
        i_slab.swap(new_i);
        valid.swap(new_valid);
        stack_len.swap(new_slen);
        stack_trunc.swap(new_strunc);
        left_stacks.swap(new_stacks);
        cap = new_cap;
        mask = new_mask;
        slab_stride = new_stride;
        W = new_cap - 4;
        bind_slab_ptrs();
    }

    void grow_raw_capacity() {
        int64_t new_rc = raw_cap * 2;
        int64_t new_rm = new_rc - 1;
        std::vector<double> nu(static_cast<size_t>(2 * new_rc), 0.0);
        std::vector<uint8_t> nb(static_cast<size_t>(2 * new_rc), 0);
        for (int64_t i = 0; i < n_raw; ++i) {
            int64_t op = i & raw_mask;
            int64_t np = i & new_rm;
            nu[static_cast<size_t>(np)] = u_ring[static_cast<size_t>(op)];
            nu[static_cast<size_t>(np + new_rc)] = u_ring[static_cast<size_t>(op)];
            nb[static_cast<size_t>(np)] = bad_ring[static_cast<size_t>(op)];
            nb[static_cast<size_t>(np + new_rc)] = bad_ring[static_cast<size_t>(op)];
        }
        u_ring.swap(nu);
        bad_ring.swap(nb);
        raw_cap = new_rc;
        raw_mask = new_rm;
    }

    // Compute exact centered stats for subsequence g (starting at raw index g)
    void compute_subseq_stats(int64_t g) {
        int64_t raw_pos = g & raw_mask;
        const double* x = u_ring.data() + raw_pos;
        int64_t slot = g & mask;

        // Exact integer tracking of non-finite count in trailing m-tick window
        bool any_bad = (bad_in_cur_win > 0);
        const double inv_m = 1.0 / static_cast<double>(m);

        auto exact_mean_var = [&](double& out_mean, double& out_var) {
#if defined(FMP_HAS_AVX2)
            __m256d s0 = _mm256_setzero_pd();
            __m256d s1 = _mm256_setzero_pd();
            int64_t k = 0;
            for (; k + 8 <= m; k += 8) {
                s0 = _mm256_add_pd(s0, _mm256_loadu_pd(x + k));
                s1 = _mm256_add_pd(s1, _mm256_loadu_pd(x + k + 4));
            }
            s0 = _mm256_add_pd(s0, s1);
            for (; k + 4 <= m; k += 4) {
                s0 = _mm256_add_pd(s0, _mm256_loadu_pd(x + k));
            }
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, s0);
            double sum = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
            for (; k < m; ++k) sum += x[k];
            out_mean = sum * inv_m;
            last_sum = sum;

            __m256d vmu = _mm256_set1_pd(out_mean);
            __m256d v0 = _mm256_setzero_pd();
            __m256d v1 = _mm256_setzero_pd();
            k = 0;
            for (; k + 8 <= m; k += 8) {
                __m256d d0 = _mm256_sub_pd(_mm256_loadu_pd(x + k), vmu);
                __m256d d1 = _mm256_sub_pd(_mm256_loadu_pd(x + k + 4), vmu);
                v0 = _mm256_fmadd_pd(d0, d0, v0);
                v1 = _mm256_fmadd_pd(d1, d1, v1);
            }
            v0 = _mm256_add_pd(v0, v1);
            for (; k + 4 <= m; k += 4) {
                __m256d d0 = _mm256_sub_pd(_mm256_loadu_pd(x + k), vmu);
                v0 = _mm256_fmadd_pd(d0, d0, v0);
            }
            _mm256_store_pd(tmp, v0);
            out_var = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
            for (; k < m; ++k) {
                double d = x[k] - out_mean;
                out_var += d * d;
            }
            last_var = out_var;
            last_peak_var = out_var;
#elif defined(FMP_HAS_NEON)
            float64x2_t s0 = vdupq_n_f64(0.0);
            float64x2_t s1 = vdupq_n_f64(0.0);
            int64_t k = 0;
            for (; k + 4 <= m; k += 4) {
                s0 = vaddq_f64(s0, vld1q_f64(x + k));
                s1 = vaddq_f64(s1, vld1q_f64(x + k + 2));
            }
            s0 = vaddq_f64(s0, s1);
            double sum = vgetq_lane_f64(s0, 0) + vgetq_lane_f64(s0, 1);
            for (; k < m; ++k) sum += x[k];
            out_mean = sum * inv_m;
            last_sum = sum;
            float64x2_t vmu = vdupq_n_f64(out_mean);
            float64x2_t v0 = vdupq_n_f64(0.0);
            float64x2_t v1 = vdupq_n_f64(0.0);
            k = 0;
            for (; k + 4 <= m; k += 4) {
                float64x2_t d0 = vsubq_f64(vld1q_f64(x + k), vmu);
                float64x2_t d1 = vsubq_f64(vld1q_f64(x + k + 2), vmu);
                v0 = vfmaq_f64(v0, d0, d0);
                v1 = vfmaq_f64(v1, d1, d1);
            }
            v0 = vaddq_f64(v0, v1);
            out_var = vgetq_lane_f64(v0, 0) + vgetq_lane_f64(v0, 1);
            for (; k < m; ++k) {
                double d = x[k] - out_mean;
                out_var += d * d;
            }
            last_var = out_var;
            last_peak_var = out_var;
#else
            double sum = 0.0;
            for (int64_t k = 0; k < m; ++k) sum += x[k];
            out_mean = sum * inv_m;
            last_sum = sum;
            out_var = 0.0;
            for (int64_t k = 0; k < m; ++k) {
                double d = x[k] - out_mean;
                out_var += d * d;
            }
            last_var = out_var;
            last_peak_var = out_var;
#endif
        };

        double mean_g = 0.0;
        double var_g = 0.0;
        if (m <= 256 || (g % 32) == 0 || !std::isfinite(last_sum) || !std::isfinite(last_var)) {
            exact_mean_var(mean_g, var_g);
        } else {
            double old_mu = mu[(g - 1) & mask];
            double x_out = u_ring[static_cast<size_t>((g - 1) & raw_mask)];
            double x_in = x[m - 1];
            last_sum += (x_in - x_out);
            mean_g = last_sum * inv_m;
            last_var += (x_in - mean_g) * (x_in - old_mu) - (x_out - mean_g) * (x_out - old_mu);
            var_g = last_var;
            if (var_g > last_peak_var) last_peak_var = var_g;
            if (var_g <= last_peak_var * 1e-4 || !std::isfinite(var_g)) {
                exact_mean_var(mean_g, var_g);
            }
        }

        mu[slot] = mean_g;
        if (any_bad) {
            valid[static_cast<size_t>(slot)] = 0;
            inv_norm[slot] = kQuietNaN;
            return;
        }
        double abs_level = std::abs(mean_g + base);
        double rel_floor = abs_level * 1e-13;
        double var_floor = std::max(1e-300, static_cast<double>(m) * rel_floor * rel_floor);
        if (var_g <= var_floor || !std::isfinite(var_g)) {
            valid[static_cast<size_t>(slot)] = 0;
            inv_norm[slot] = kQuietNaN;
        } else {
            valid[static_cast<size_t>(slot)] = 1;
            inv_norm[slot] = 1.0 / std::sqrt(var_g);
        }
    }

    inline double exact_cov_gids(int64_t g1, int64_t g2) const {
        return exact_cov_scalar(
            u_ring.data() + (g1 & raw_mask), mu[g1 & mask],
            u_ring.data() + (g2 & raw_mask), mu[g2 & mask],
            m);
    }

    // Push a strictly-increasing suffix-max record-holder onto left_stacks[slot]
    static inline void push_left_stack_entry(
        StackEntry* st, uint8_t& len, uint8_t& trunc, int64_t k, double c
    ) {
        if (len < kStackCap) {
            st[len++] = {k, c};
        } else {
            std::memmove(st, st + 1, sizeof(StackEntry) * (kStackCap - 1));
            st[kStackCap - 1] = {k, c};
            trunc = 1;
        }
    }

    // Fused right-to-left in-place covariance update + Left-MP / suffix-max stack + Right-MP
    // Processes k in [seg_k_lo, seg_k_hi] in STRICTLY DECREASING order (seg_k_hi down to seg_k_lo),
    // assuming [seg_k_lo & mask .. seg_k_hi & mask] is contiguous (no ring wrap inside segment).
    template <bool TrackFull, bool BuildStack, bool ValidG>
    void process_segment_rtl(
        int64_t seg_k_lo,
        int64_t seg_k_hi,
        int64_t g,
        double df_prev,
        double dg_prev,
        double inv_g,
        double& best_val,     // scaled (cn*inv_a) if !TrackFull, else true corr
        int64_t& best_idx,
        StackEntry* st,
        uint8_t& st_len,
        uint8_t& st_trunc
    ) {
        if (seg_k_lo > seg_k_hi) return;
        double* __restrict__ cov_p = cov;
        const double* __restrict__ df_a = df;
        const double* __restrict__ dg_a = dg;
        const double* __restrict__ inv_a = inv_norm;
        double* __restrict__ rc = right_corr;
        int64_t* __restrict__ ri = right_idx;

        const int64_t slot_lo = seg_k_lo & mask;
        const int64_t slot_hi = seg_k_hi & mask;
        int64_t sc = slot_hi;
        int64_t k_cur = seg_k_hi;

        (void)inv_g; (void)best_val; (void)best_idx; (void)st; (void)st_len; (void)st_trunc;
        (void)inv_a; (void)rc; (void)ri; (void)g;

#if defined(FMP_HAS_AVX2)
        const __m256d v_dfp = _mm256_set1_pd(df_prev);
        const __m256d v_dgp = _mm256_set1_pd(dg_prev);
        const int64_t simd_stop = std::max<int64_t>(slot_lo, 1);

        if constexpr (!ValidG) {
            // Subsequence g is NaN/flat: only advance in-place covariance recurrence!
            for (; sc - 7 >= simd_stop; sc -= 8, k_cur -= 8) {
                const int64_t base = sc - 7;
                // Load BOTH prior cov vectors BEFORE storing either (base+3 overlaps base..base+3)
                __m256d cp0 = _mm256_loadu_pd(cov_p + base - 1);
                __m256d cp1 = _mm256_loadu_pd(cov_p + base + 3);
                __m256d cn0 = _mm256_fmadd_pd(v_dfp, _mm256_loadu_pd(dg_a + base - 1),
                              _mm256_fmadd_pd(v_dgp, _mm256_loadu_pd(df_a + base - 1), cp0));
                __m256d cn1 = _mm256_fmadd_pd(v_dfp, _mm256_loadu_pd(dg_a + base + 3),
                              _mm256_fmadd_pd(v_dgp, _mm256_loadu_pd(df_a + base + 3), cp1));
                _mm256_storeu_pd(cov_p + base, cn0);
                _mm256_storeu_pd(cov_p + base + 4, cn1);
            }
        } else if constexpr (!TrackFull) {
            // Pure causal Left-MP mode: factor inv_g out of inner loop, zero right_corr/stack traffic!
            __m256d v_best = _mm256_set1_pd(best_val);
            for (; sc - 7 >= simd_stop; sc -= 8, k_cur -= 8) {
                const int64_t base = sc - 7;
                __m256d cp0 = _mm256_loadu_pd(cov_p + base - 1);
                __m256d cp1 = _mm256_loadu_pd(cov_p + base + 3);
                __m256d cn0 = _mm256_fmadd_pd(v_dfp, _mm256_loadu_pd(dg_a + base - 1),
                              _mm256_fmadd_pd(v_dgp, _mm256_loadu_pd(df_a + base - 1), cp0));
                __m256d cn1 = _mm256_fmadd_pd(v_dfp, _mm256_loadu_pd(dg_a + base + 3),
                              _mm256_fmadd_pd(v_dgp, _mm256_loadu_pd(df_a + base + 3), cp1));
                _mm256_storeu_pd(cov_p + base, cn0);
                _mm256_storeu_pd(cov_p + base + 4, cn1);

                __m256d s0 = _mm256_mul_pd(cn0, _mm256_loadu_pd(inv_a + base));
                __m256d s1 = _mm256_mul_pd(cn1, _mm256_loadu_pd(inv_a + base + 4));
                __m256d m_any = _mm256_or_pd(
                    _mm256_cmp_pd(s0, v_best, _CMP_GT_OQ),
                    _mm256_cmp_pd(s1, v_best, _CMP_GT_OQ));
                if (__builtin_expect(_mm256_movemask_pd(m_any) != 0, 0)) {
                    alignas(32) double tmp[8];
                    _mm256_store_pd(tmp, s0);
                    _mm256_store_pd(tmp + 4, s1);
                    const int64_t k_base = k_cur - 7;
                    for (int lane = 7; lane >= 0; --lane) {
                        if (tmp[lane] > best_val) {
                            best_val = tmp[lane];
                            best_idx = k_base + lane;
                        }
                    }
                    v_best = _mm256_set1_pd(best_val);
                }
            }
        } else {
            // Full-MP mode (with fused right-to-left suffix-max stack when BuildStack==true)
            const __m256d v_invg = _mm256_set1_pd(inv_g);
            __m256d v_best = _mm256_set1_pd(best_val);
            const __m256i v_g = _mm256_set1_epi64x(g);
            for (; sc - 7 >= simd_stop; sc -= 8, k_cur -= 8) {
                const int64_t base = sc - 7;
                __m256d cp0 = _mm256_loadu_pd(cov_p + base - 1);
                __m256d cp1 = _mm256_loadu_pd(cov_p + base + 3);
                __m256d cn0 = _mm256_fmadd_pd(v_dfp, _mm256_loadu_pd(dg_a + base - 1),
                              _mm256_fmadd_pd(v_dgp, _mm256_loadu_pd(df_a + base - 1), cp0));
                __m256d cn1 = _mm256_fmadd_pd(v_dfp, _mm256_loadu_pd(dg_a + base + 3),
                              _mm256_fmadd_pd(v_dgp, _mm256_loadu_pd(df_a + base + 3), cp1));
                _mm256_storeu_pd(cov_p + base, cn0);
                _mm256_storeu_pd(cov_p + base + 4, cn1);

                __m256d corr0 = _mm256_mul_pd(_mm256_mul_pd(cn0, _mm256_loadu_pd(inv_a + base)), v_invg);
                __m256d corr1 = _mm256_mul_pd(_mm256_mul_pd(cn1, _mm256_loadu_pd(inv_a + base + 4)), v_invg);

                __m256d ml_any = _mm256_or_pd(
                    _mm256_cmp_pd(corr0, v_best, _CMP_GT_OQ),
                    _mm256_cmp_pd(corr1, v_best, _CMP_GT_OQ));
                __m256d mr0 = _mm256_cmp_pd(corr0, _mm256_loadu_pd(rc + base), _CMP_GT_OQ);
                __m256d mr1 = _mm256_cmp_pd(corr1, _mm256_loadu_pd(rc + base + 4), _CMP_GT_OQ);
                __m256d hit_any = _mm256_or_pd(ml_any, _mm256_or_pd(mr0, mr1));

                if (__builtin_expect(_mm256_movemask_pd(hit_any) != 0, 0)) {
                    if (_mm256_movemask_pd(ml_any) != 0) {
                        alignas(32) double tmp[8];
                        _mm256_store_pd(tmp, corr0);
                        _mm256_store_pd(tmp + 4, corr1);
                        const int64_t k_base = k_cur - 7;
                        // Strictly right-to-left (lane 7 down to 0) preserves exact monotone suffix-max stack!
                        for (int lane = 7; lane >= 0; --lane) {
                            if (tmp[lane] > best_val) {
                                best_val = tmp[lane];
                                best_idx = k_base + lane;
                                if constexpr (BuildStack) {
                                    push_left_stack_entry(st, st_len, st_trunc, best_idx, best_val);
                                }
                            }
                        }
                        v_best = _mm256_set1_pd(best_val);
                    }
                    if (_mm256_movemask_pd(mr0) != 0) {
                        __m256d rcb0 = _mm256_loadu_pd(rc + base);
                        _mm256_storeu_pd(rc + base, _mm256_blendv_pd(rcb0, corr0, mr0));
                        __m256i rci0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ri + base));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ri + base),
                                            _mm256_blendv_epi8(rci0, v_g, _mm256_castpd_si256(mr0)));
                    }
                    if (_mm256_movemask_pd(mr1) != 0) {
                        __m256d rcb1 = _mm256_loadu_pd(rc + base + 4);
                        _mm256_storeu_pd(rc + base + 4, _mm256_blendv_pd(rcb1, corr1, mr1));
                        __m256i rci1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ri + base + 4));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ri + base + 4),
                                            _mm256_blendv_epi8(rci1, v_g, _mm256_castpd_si256(mr1)));
                    }
                }
            }
        }
#endif

        // Scalar right-to-left tail (also cleanly handles slot 0 wrapping to cap - 1)
        for (; sc >= slot_lo; --sc, --k_cur) {
            const int64_t sp = (sc - 1) & mask;
            const double cn = cov_p[sp] + df_prev * dg_a[sp] + dg_prev * df_a[sp];
            cov_p[sc] = cn;
            if constexpr (ValidG) {
                if constexpr (!TrackFull) {
                    const double s_val = cn * inv_a[sc];
                    if (s_val > best_val) {
                        best_val = s_val;
                        best_idx = k_cur;
                    }
                } else {
                    const double corr = (cn * inv_a[sc]) * inv_g;
                    if (corr > best_val) {
                        best_val = corr;
                        best_idx = k_cur;
                        if constexpr (BuildStack) {
                            push_left_stack_entry(st, st_len, st_trunc, k_cur, corr);
                        }
                    }
                    if (corr > rc[sc]) {
                        rc[sc] = corr;
                        ri[sc] = g;
                    }
                }
            }
        }
    }

    // Split arbitrary [r_lo, r_hi] into at most 2 contiguous right-to-left ring segments
    template <bool TrackFull, bool BuildStack, bool ValidG>
    void sweep_range_rtl(
        int64_t r_lo, int64_t r_hi,
        int64_t g, double df_prev, double dg_prev, double inv_g,
        double& best_val, int64_t& best_idx,
        StackEntry* st, uint8_t& st_len, uint8_t& st_trunc
    ) {
        if (r_lo > r_hi) return;
        const int64_t total = r_hi - r_lo + 1;
        const int64_t s_hi = r_hi & mask;
        const int64_t seg1 = std::min(total, s_hi + 1);
        process_segment_rtl<TrackFull, BuildStack, ValidG>(
            r_hi - seg1 + 1, r_hi, g, df_prev, dg_prev, inv_g,
            best_val, best_idx, st, st_len, st_trunc);
        if (seg1 < total) {
            process_segment_rtl<TrackFull, BuildStack, ValidG>(
                r_lo, r_hi - seg1, g, df_prev, dg_prev, inv_g,
                best_val, best_idx, st, st_len, st_trunc);
        }
    }

    // Fallback exact suffix-max rebuild for subsequence `k` over [gid_min, k - ez - 1]
    // (only called if a truncated stack completely empties on an adversarial monotone stream)
    void rebuild_left_stack_exact(int64_t k, int64_t gid_min) {
        int64_t slot = k & mask;
        StackEntry* st = left_stacks.data() + slot * kStackCap;
        uint8_t len = 0;
        uint8_t trunc = 0;
        double cur_max = kNegInf;
        double inv_k = inv_norm[slot];
        int64_t j_hi = k - ez - 1;
        if (valid[static_cast<size_t>(slot)] && j_hi >= gid_min) {
            for (int64_t j = j_hi; j >= gid_min; --j) {
                int64_t js = j & mask;
                if (!valid[static_cast<size_t>(js)]) continue;
                double c = exact_cov_gids(k, j) * inv_k * inv_norm[js];
                if (c > cur_max) {
                    cur_max = c;
                    push_left_stack_entry(st, len, trunc, j, c);
                }
            }
        }
        stack_len[static_cast<size_t>(slot)] = len;
        stack_trunc[static_cast<size_t>(slot)] = trunc;
        if (len > 0) {
            left_top_corr[slot] = st[len - 1].corr;
            left_top_idx[slot] = st[len - 1].idx;
        } else {
            left_top_corr[slot] = kNegInf;
            left_top_idx[slot] = -1;
        }
    }

    // Update causal conformal p-value and m-dependent Holder-tempered exchangeable e-CUSUM (AVX2).
    // 1) Excludes initial g < 2*m ramp-up and delays calibration by m ticks so every sample in
    //    cal_ring is strictly non-overlapping with the current window.
    // 2) Exact finite-sample distribution-free exchangeable e-value (E_{H0}[e_t] == 1 identically):
    //      e_t = (n + 1) * max(0, d - mean_all)^4 / sum_{j=1..n+1} max(0, X_j - mean_all)^4
    // 3) 1/m Holder-tempered 3-bet e-CUSUM mixture accounts for m-overlapping autocorrelation.
    void update_eprocess(int64_t g, double d, fmp_tick_result_t* out) {
        if (!std::isfinite(d) || g < 2 * m) {
            out->discord_z = 0.0;
            out->p_value = 1.0;
            out->e_value = e_wealth;
            out->is_alert = (e_wealth >= (1.0 / alpha)) ? 1 : 0;
            return;
        }
        auto push_cal = [&](double val) {
            if (cal_delay_cnt < m) {
                cal_delay[static_cast<size_t>(cal_delay_head)] = val;
                cal_delay_head = (cal_delay_head + 1 == m) ? 0 : (cal_delay_head + 1);
                ++cal_delay_cnt;
            } else {
                double oldest = cal_delay[static_cast<size_t>(cal_delay_head)];
                cal_delay[static_cast<size_t>(cal_delay_head)] = val;
                cal_delay_head = (cal_delay_head + 1 == m) ? 0 : (cal_delay_head + 1);
                cal_ring[cal_head] = oldest;
                cal_head = (cal_head + 1) & (kCalCap - 1);
                if (cal_count < kCalCap) ++cal_count;
            }
        };
        if (cal_count < 16) {
            push_cal(d);
            out->discord_z = 0.0;
            out->p_value = 1.0;
            out->e_value = e_wealth;
            out->is_alert = 0;
            return;
        }

        const int64_t ncal = cal_count;
        const double dncal = static_cast<double>(ncal);
        const double dn1 = static_cast<double>(ncal + 1);
        int64_t ge_cnt = 1;
        double sum = 0.0;
        double var = 0.0;
        double sum_w = 0.0;

#if defined(FMP_HAS_AVX2)
        __m256d vd = _mm256_set1_pd(d);
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        __m256i ge0 = _mm256_setzero_si256();
        __m256i ge1 = _mm256_setzero_si256();
        int64_t i = 0;
        for (; i + 8 <= ncal; i += 8) {
            __m256d v0 = _mm256_load_pd(cal_ring + i);
            __m256d v1 = _mm256_load_pd(cal_ring + i + 4);
            s0 = _mm256_add_pd(s0, v0);
            s1 = _mm256_add_pd(s1, v1);
            ge0 = _mm256_sub_epi64(ge0, _mm256_castpd_si256(_mm256_cmp_pd(v0, vd, _CMP_GE_OQ)));
            ge1 = _mm256_sub_epi64(ge1, _mm256_castpd_si256(_mm256_cmp_pd(v1, vd, _CMP_GE_OQ)));
        }
        s0 = _mm256_add_pd(s0, s1);
        ge0 = _mm256_add_epi64(ge0, ge1);
        for (; i + 4 <= ncal; i += 4) {
            __m256d v0 = _mm256_load_pd(cal_ring + i);
            s0 = _mm256_add_pd(s0, v0);
            ge0 = _mm256_sub_epi64(ge0, _mm256_castpd_si256(_mm256_cmp_pd(v0, vd, _CMP_GE_OQ)));
        }
        alignas(32) double stmp[4];
        alignas(32) int64_t gtmp[4];
        _mm256_store_pd(stmp, s0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(gtmp), ge0);
        sum = (stmp[0] + stmp[1]) + (stmp[2] + stmp[3]);
        ge_cnt += (gtmp[0] + gtmp[1]) + (gtmp[2] + gtmp[3]);
        for (; i < ncal; ++i) {
            double v = cal_ring[i];
            sum += v;
            if (v >= d) ++ge_cnt;
        }
        const double mean = sum / dncal;
        const double mean_all = (sum + d) / dn1;
        __m256d vmean = _mm256_set1_pd(mean);
        __m256d vmall = _mm256_set1_pd(mean_all);
        __m256d vzero = _mm256_setzero_pd();
        __m256d vr0 = _mm256_setzero_pd();
        __m256d vr1 = _mm256_setzero_pd();
        __m256d vw0 = _mm256_setzero_pd();
        __m256d vw1 = _mm256_setzero_pd();
        i = 0;
        for (; i + 8 <= ncal; i += 8) {
            __m256d x0 = _mm256_load_pd(cal_ring + i);
            __m256d x1 = _mm256_load_pd(cal_ring + i + 4);
            __m256d dv0 = _mm256_sub_pd(x0, vmean);
            __m256d dv1 = _mm256_sub_pd(x1, vmean);
            vr0 = _mm256_fmadd_pd(dv0, dv0, vr0);
            vr1 = _mm256_fmadd_pd(dv1, dv1, vr1);
            __m256d p0 = _mm256_max_pd(vzero, _mm256_sub_pd(x0, vmall));
            __m256d p1 = _mm256_max_pd(vzero, _mm256_sub_pd(x1, vmall));
            __m256d p0_2 = _mm256_mul_pd(p0, p0);
            __m256d p1_2 = _mm256_mul_pd(p1, p1);
            vw0 = _mm256_fmadd_pd(p0_2, p0_2, vw0);
            vw1 = _mm256_fmadd_pd(p1_2, p1_2, vw1);
        }
        vr0 = _mm256_add_pd(vr0, vr1);
        vw0 = _mm256_add_pd(vw0, vw1);
        for (; i + 4 <= ncal; i += 4) {
            __m256d x0 = _mm256_load_pd(cal_ring + i);
            __m256d dv0 = _mm256_sub_pd(x0, vmean);
            vr0 = _mm256_fmadd_pd(dv0, dv0, vr0);
            __m256d p0 = _mm256_max_pd(vzero, _mm256_sub_pd(x0, vmall));
            __m256d p0_2 = _mm256_mul_pd(p0, p0);
            vw0 = _mm256_fmadd_pd(p0_2, p0_2, vw0);
        }
        _mm256_store_pd(stmp, vr0);
        var = (stmp[0] + stmp[1]) + (stmp[2] + stmp[3]);
        _mm256_store_pd(stmp, vw0);
        sum_w = (stmp[0] + stmp[1]) + (stmp[2] + stmp[3]);
        for (; i < ncal; ++i) {
            double x = cal_ring[i];
            double dv = x - mean;
            var += dv * dv;
            double pos = x - mean_all;
            if (pos > 0.0) {
                double p2 = pos * pos;
                sum_w += p2 * p2;
            }
        }
#else
        for (int64_t i = 0; i < ncal; ++i) {
            double v = cal_ring[i];
            sum += v;
            if (v >= d) ++ge_cnt;
        }
        const double mean = sum / dncal;
        const double mean_all = (sum + d) / dn1;
        for (int64_t i = 0; i < ncal; ++i) {
            double x = cal_ring[i];
            double dv = x - mean;
            var += dv * dv;
            double pos = x - mean_all;
            if (pos > 0.0) {
                double p2 = pos * pos;
                sum_w += p2 * p2;
            }
        }
#endif

        double sd = std::sqrt(var / dncal);
        double z = (sd > 1e-12) ? ((d - mean) / sd) : 0.0;
        double pval = static_cast<double>(ge_cnt) / dn1;

        double pos_d = (d > mean_all) ? (d - mean_all) : 0.0;
        double wd2 = pos_d * pos_d;
        double w_d = wd2 * wd2;
        sum_w += w_d;
        double e_factor = (sum_w > 1e-30) ? (dn1 * w_d / sum_w) : 1.0;

        static constexpr double kLams[3] = {0.10, 0.30, 0.60};
        const double inv_m = 1.0 / static_cast<double>(m);
        double e_sum = 0.0;
        const double e_cap = std::min(1e100, std::max(1e12, 100.0 / alpha));
        if (pos_d == 0.0 && sum_w > 1e-30) {
            for (int b = 0; b < 3; ++b) {
                double base_w = (e_cusum[b] > 1.0) ? e_cusum[b] : 1.0;
                double nxt = base_w * e_decay[b];
                e_cusum[b] = nxt;
                e_sum += nxt;
            }
        } else {
            for (int b = 0; b < 3; ++b) {
                double mult = 1.0 + kLams[b] * (e_factor - 1.0);
                if (mult < 1e-6) mult = 1e-6;
                double base_w = (e_cusum[b] > 1.0) ? e_cusum[b] : 1.0;
                double nxt = base_w * std::exp(inv_m * std::log(mult));
                if (nxt > e_cap) nxt = e_cap;
                e_cusum[b] = nxt;
                e_sum += nxt;
            }
        }
        e_wealth = e_sum * (1.0 / 3.0);

        // Append current d to m-tick causal delay buffer AFTER scoring
        push_cal(d);

        out->discord_z = z;
        out->p_value = pval;
        out->e_value = e_wealth;
        out->is_alert = (e_wealth >= (1.0 / alpha)) ? 1 : 0;
    }
};

// ---------------------------------------------------------------------------
// Public C API Implementation
// ---------------------------------------------------------------------------
extern "C" {

const char* fmp_version(void) {
    return "0.1.0";
}

const char* fmp_simd_isa(void) {
#if defined(FMP_HAS_AVX2)
    return "AVX2+FMA";
#elif defined(FMP_HAS_NEON)
    return "ARM64-NEON";
#else
    return "SCALAR";
#endif
}

int fmp_self_join(
    const double* T,
    int64_t N,
    int64_t m,
    int64_t ez,
    int n_threads,
    double* out_mp,
    int64_t* out_idx,
    double* out_lmp,
    int64_t* out_lidx,
    double* out_rmp,
    int64_t* out_ridx
) {
    if (!T || !out_mp || !out_idx || m < 3 || N < m) return -1;
    int64_t excl = (ez < 0) ? fmp_default_excl_zone(m) : ez;
    int64_t n = N - m + 1;
    int nt = resolve_threads(n_threads);

    SeriesPrep A = precompute_series(T, N, m);

    // Stagger r_best (+2304B) & l_best (+2816B) mod 4096B away from A.slab (+256,+768,+1280,+1792B).
    // Both left & right kSeriesPad sentinel zones around l_best are +inf (and l_idx is -1).
    const int64_t b_stride = ((n + 2 * kSeriesPad + 511) & ~511LL) + 64;
    const int64_t slab_len = 256 + kSeriesPad + 2 * b_stride;
    std::vector<double> best_slab(static_cast<size_t>(slab_len), kPosInf);
    double* r_best = best_slab.data() + 256 + kSeriesPad;
    double* l_best = r_best + b_stride;
    std::fill_n(r_best, n, kNegInf);
    std::fill_n(l_best, n, kNegInf);
    std::vector<int64_t> r_idx(static_cast<size_t>(n), -1);
    std::vector<int64_t> l_idx_buf(static_cast<size_t>(n + 2 * kSeriesPad), -1);
    int64_t* l_idx = l_idx_buf.data() + kSeriesPad;

    int64_t d_min = excl + 1;
    if (d_min < n) {
        int actual_nt = std::min<int>(nt, static_cast<int>(std::max<int64_t>(1, (n - d_min + kTileW - 1) / kTileW)));
        std::atomic<int64_t> next_db(d_min);
        if (actual_nt <= 1 || n < 2048) {
            run_diag_dynamic<true, false>(
                A, A, next_db, n,
                r_best, r_idx.data(),
                l_best, l_idx);
        } else {
            std::vector<std::vector<double>> t_slab(
                static_cast<size_t>(actual_nt),
                std::vector<double>(static_cast<size_t>(slab_len), kPosInf));
            std::vector<std::vector<int64_t>> t_ri(
                static_cast<size_t>(actual_nt), std::vector<int64_t>(static_cast<size_t>(n), -1));
            std::vector<std::vector<int64_t>> t_li(
                static_cast<size_t>(actual_nt), std::vector<int64_t>(static_cast<size_t>(n + 2 * kSeriesPad), -1));

            std::vector<std::thread> workers;
            workers.reserve(static_cast<size_t>(actual_nt - 1));
            for (int t = 1; t < actual_nt; ++t) {
                double* rb = t_slab[static_cast<size_t>(t)].data() + 256 + kSeriesPad;
                double* lb = rb + b_stride;
                std::fill_n(rb, n, kNegInf);
                std::fill_n(lb, n, kNegInf);
                workers.emplace_back([&, t, rb, lb]() {
                    run_diag_dynamic<true, false>(
                        A, A, next_db, n,
                        rb, t_ri[static_cast<size_t>(t)].data(),
                        lb, t_li[static_cast<size_t>(t)].data() + kSeriesPad);
                });
            }
            run_diag_dynamic<true, false>(
                A, A, next_db, n,
                r_best, r_idx.data(),
                l_best, l_idx);
            for (auto& w : workers) w.join();

            for (int t = 1; t < actual_nt; ++t) {
                const double* rb = t_slab[static_cast<size_t>(t)].data() + 256 + kSeriesPad;
                const double* lb = rb + b_stride;
                const auto& ri = t_ri[static_cast<size_t>(t)];
                const int64_t* li = t_li[static_cast<size_t>(t)].data() + kSeriesPad;
                for (int64_t i = 0; i < n; ++i) {
                    size_t ui = static_cast<size_t>(i);
                    if (rb[i] > r_best[i] ||
                        (rb[i] == r_best[i] && ri[ui] >= 0 && (r_idx[ui] < 0 || ri[ui] < r_idx[ui]))) {
                        r_best[i] = rb[i];
                        r_idx[ui] = ri[ui];
                    }
                    if (lb[i] > l_best[i] ||
                        (lb[i] == l_best[i] && li[i] >= 0 && (l_idx[i] < 0 || li[i] < l_idx[i]))) {
                        l_best[i] = lb[i];
                        l_idx[i] = li[i];
                    }
                }
            }
        }
    }

    const double* uA = A.u_ptr();
    for (int64_t i = 0; i < n; ++i) {
        size_t ui = static_cast<size_t>(i);
        double lc = l_best[i];
        double rc = r_best[i];
        int64_t li = l_idx[i];
        int64_t ri = r_idx[ui];
        double ld = (__builtin_expect(lc > 1.0 - 1e-10 && li >= 0, 0))
            ? exact_z_dist_scalar(uA + i, uA + li, m)
            : corr_to_dist(lc, m);
        double rd = (__builtin_expect(rc > 1.0 - 1e-10 && ri >= 0, 0))
            ? exact_z_dist_scalar(uA + i, uA + ri, m)
            : corr_to_dist(rc, m);
        if (ld <= rd && li >= 0) {
            out_mp[i] = ld;
            out_idx[i] = li;
        } else if (ri >= 0) {
            out_mp[i] = rd;
            out_idx[i] = ri;
        } else {
            out_mp[i] = kPosInf;
            out_idx[i] = -1;
        }
        if (out_lmp) out_lmp[i] = ld;
        if (out_lidx) out_lidx[i] = li;
        if (out_rmp) out_rmp[i] = rd;
        if (out_ridx) out_ridx[i] = ri;
    }
    return 0;
}

int fmp_ab_join(
    const double* A_raw,
    int64_t NA,
    const double* B_raw,
    int64_t NB,
    int64_t m,
    int n_threads,
    double* out_mp_a,
    int64_t* out_idx_a,
    double* out_mp_b,
    int64_t* out_idx_b
) {
    if (!A_raw || !B_raw || !out_mp_a || !out_idx_a || m < 3 || NA < m || NB < m) return -1;
    if ((out_mp_b != nullptr) != (out_idx_b != nullptr)) return -1;
    int64_t nA = NA - m + 1;
    int64_t nB = NB - m + 1;
    int nt = resolve_threads(n_threads);
    const bool need_b = (out_mp_b != nullptr && out_idx_b != nullptr);

    SeriesPrep A = precompute_series(A_raw, NA, m);
    SeriesPrep B = precompute_series(B_raw, NB, m);

    auto init_a_best = [&](double* dst) {
        if (need_b) {
            std::fill_n(dst, nA, kNegInf);
        } else {
            for (int64_t i = 0; i < nA; ++i) {
                dst[i] = A.valid[static_cast<size_t>(i)] ? -kPosInf : kQuietNaN;
            }
        }
    };

    const int64_t a_stride = ((nA + 2 * kSeriesPad + 511) & ~511LL) + 64;
    const int64_t b_buf_len = need_b ? (nB + 2 * kSeriesPad) : 1;
    const int64_t slab_len = 256 + kSeriesPad + a_stride + b_buf_len;
    std::vector<double> best_slab(static_cast<size_t>(slab_len), kPosInf);
    double* a_best = best_slab.data() + 256 + kSeriesPad;
    double* b_best = a_best + a_stride;
    init_a_best(a_best);
    if (need_b) std::fill_n(b_best, nB, kNegInf);
    std::vector<int64_t> a_idx(static_cast<size_t>(nA), -1);
    std::vector<int64_t> b_idx_buf(static_cast<size_t>(b_buf_len), -1);
    int64_t* b_idx = need_b ? (b_idx_buf.data() + kSeriesPad) : b_idx_buf.data();

    int64_t d_min = -(nA - 1);
    int64_t d_max = nB;
    int64_t total_diags = d_max - d_min;
    int actual_nt = std::min<int>(nt, static_cast<int>(std::max<int64_t>(1, (total_diags + kTileW - 1) / kTileW)));

    auto run_range = [&](int64_t lo, int64_t hi, double* rb, int64_t* ri, double* cb, int64_t* ci) {
        if (need_b) {
            run_diag_static<true, false>(A, B, lo, hi, rb, ri, cb, ci);
        } else {
            run_diag_static<false, false>(A, B, lo, hi, rb, ri, nullptr, nullptr);
        }
    };

    if (actual_nt <= 1 || (nA * nB) < 4000000LL) {
        run_range(d_min, d_max, a_best, a_idx.data(), b_best, b_idx);
    } else {
        auto cuts = compute_balanced_diag_cuts(nA, nB, d_min, d_max, actual_nt);
        std::vector<std::vector<double>> t_slab(
            static_cast<size_t>(actual_nt),
            std::vector<double>(static_cast<size_t>(slab_len), kPosInf));
        std::vector<std::vector<int64_t>> t_ai(
            static_cast<size_t>(actual_nt), std::vector<int64_t>(static_cast<size_t>(nA), -1));
        std::vector<std::vector<int64_t>> t_bi(
            static_cast<size_t>(actual_nt), std::vector<int64_t>(static_cast<size_t>(b_buf_len), -1));

        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(actual_nt - 1));
        for (int t = 1; t < actual_nt; ++t) {
            double* rb = t_slab[static_cast<size_t>(t)].data() + 256 + kSeriesPad;
            double* cb = rb + a_stride;
            init_a_best(rb);
            if (need_b) std::fill_n(cb, nB, kNegInf);
            int64_t* ci = need_b ? (t_bi[static_cast<size_t>(t)].data() + kSeriesPad)
                                 : t_bi[static_cast<size_t>(t)].data();
            int64_t lo = cuts[static_cast<size_t>(t)];
            int64_t hi = cuts[static_cast<size_t>(t + 1)];
            workers.emplace_back([&, t, lo, hi, rb, cb, ci]() {
                run_range(lo, hi, rb, t_ai[static_cast<size_t>(t)].data(), cb, ci);
            });
        }
        run_range(cuts[0], cuts[1], a_best, a_idx.data(), b_best, b_idx);
        for (auto& w : workers) w.join();

        for (int t = 1; t < actual_nt; ++t) {
            const double* rb = t_slab[static_cast<size_t>(t)].data() + 256 + kSeriesPad;
            const auto& ri = t_ai[static_cast<size_t>(t)];
            for (int64_t i = 0; i < nA; ++i) {
                size_t ui = static_cast<size_t>(i);
                if (rb[i] > a_best[i] ||
                    (rb[i] == a_best[i] && ri[ui] >= 0 && (a_idx[ui] < 0 || ri[ui] < a_idx[ui]))) {
                    a_best[i] = rb[i];
                    a_idx[ui] = ri[ui];
                }
            }
            if (need_b) {
                const double* cb = rb + a_stride;
                const int64_t* ci = t_bi[static_cast<size_t>(t)].data() + kSeriesPad;
                for (int64_t j = 0; j < nB; ++j) {
                    if (cb[j] > b_best[j] ||
                        (cb[j] == b_best[j] && ci[j] >= 0 && (b_idx[j] < 0 || ci[j] < b_idx[j]))) {
                        b_best[j] = cb[j];
                        b_idx[j] = ci[j];
                    }
                }
            }
        }
    }

    const double* invA = A.inv_ptr();
    const double* uA = A.u_ptr();
    const double* uB = B.u_ptr();
    for (int64_t i = 0; i < nA; ++i) {
        size_t ui = static_cast<size_t>(i);
        int64_t j = a_idx[ui];
        double corr = need_b ? a_best[i] : ((j >= 0) ? (a_best[i] * invA[i]) : kNegInf);
        out_mp_a[i] = (__builtin_expect(corr > 1.0 - 1e-10 && j >= 0, 0))
            ? exact_z_dist_scalar(uA + i, uB + j, m)
            : corr_to_dist(corr, m);
        out_idx_a[i] = j;
    }
    if (need_b) {
        for (int64_t j = 0; j < nB; ++j) {
            int64_t i = b_idx[j];
            double corr = b_best[j];
            out_mp_b[j] = (__builtin_expect(corr > 1.0 - 1e-10 && i >= 0, 0))
                ? exact_z_dist_scalar(uA + i, uB + j, m)
                : corr_to_dist(corr, m);
            out_idx_b[j] = i;
        }
    }
    return 0;
}

int fmp_leak_scan(
    const double* ref,
    int64_t N_ref,
    const double* qry,
    int64_t N_qry,
    int64_t m,
    double corr_thresh,
    int check_neg,
    int n_threads,
    double* out_best_corr,
    int64_t* out_best_ref_idx,
    int64_t* out_hit_count
) {
    if (!ref || !qry || !out_best_corr || !out_best_ref_idx || m < 3 || N_ref < m || N_qry < m) return -1;
    int64_t nQ = N_qry - m + 1;
    int64_t nR = N_ref - m + 1;
    int nt = resolve_threads(n_threads);

    SeriesPrep Q = precompute_series(qry, N_qry, m);
    SeriesPrep R = precompute_series(ref, N_ref, m);

    auto init_q_best = [&](double* dst) {
        for (int64_t i = 0; i < nQ; ++i) {
            dst[i] = Q.valid[static_cast<size_t>(i)] ? -kPosInf : kQuietNaN;
        }
    };

    std::vector<double> q_slab(static_cast<size_t>(256 + nQ));
    double* q_best = q_slab.data() + 256;
    init_q_best(q_best);
    std::vector<int64_t> q_idx(static_cast<size_t>(nQ), -1);

    int64_t d_min = -(nQ - 1);
    int64_t d_max = nR;
    int64_t total_diags = d_max - d_min;
    int actual_nt = std::min<int>(nt, static_cast<int>(std::max<int64_t>(1, (total_diags + kTileW - 1) / kTileW)));

    auto run_range = [&](int64_t lo, int64_t hi, double* rb, int64_t* ri) {
        if (check_neg) {
            run_diag_static<false, true>(Q, R, lo, hi, rb, ri, nullptr, nullptr);
        } else {
            run_diag_static<false, false>(Q, R, lo, hi, rb, ri, nullptr, nullptr);
        }
    };

    if (actual_nt <= 1 || (nQ * nR) < 4000000LL) {
        run_range(d_min, d_max, q_best, q_idx.data());
    } else {
        auto cuts = compute_balanced_diag_cuts(nQ, nR, d_min, d_max, actual_nt);
        std::vector<std::vector<double>> t_slab(
            static_cast<size_t>(actual_nt), std::vector<double>(static_cast<size_t>(256 + nQ)));
        std::vector<std::vector<int64_t>> t_qi(
            static_cast<size_t>(actual_nt), std::vector<int64_t>(static_cast<size_t>(nQ), -1));
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(actual_nt - 1));
        for (int t = 1; t < actual_nt; ++t) {
            double* rb = t_slab[static_cast<size_t>(t)].data() + 256;
            init_q_best(rb);
            int64_t lo = cuts[static_cast<size_t>(t)];
            int64_t hi = cuts[static_cast<size_t>(t + 1)];
            workers.emplace_back([&, t, lo, hi, rb]() {
                run_range(lo, hi, rb, t_qi[static_cast<size_t>(t)].data());
            });
        }
        run_range(cuts[0], cuts[1], q_best, q_idx.data());
        for (auto& w : workers) w.join();

        for (int t = 1; t < actual_nt; ++t) {
            const double* rb = t_slab[static_cast<size_t>(t)].data() + 256;
            const auto& ri = t_qi[static_cast<size_t>(t)];
            for (int64_t i = 0; i < nQ; ++i) {
                size_t ui = static_cast<size_t>(i);
                if (rb[i] > q_best[i] ||
                    (rb[i] == q_best[i] && ri[ui] >= 0 && (q_idx[ui] < 0 || ri[ui] < q_idx[ui]))) {
                    q_best[i] = rb[i];
                    q_idx[ui] = ri[ui];
                }
            }
        }
    }

    const double* invQ = Q.inv_ptr();
    const double* uQ = Q.u_ptr();
    const double* muQ = Q.mu_ptr();
    const double* uR = R.u_ptr();
    const double* muR = R.mu_ptr();
    const double* invR = R.inv_ptr();
    int64_t hits = 0;
    for (int64_t i = 0; i < nQ; ++i) {
        size_t ui = static_cast<size_t>(i);
        int64_t j = q_idx[ui];
        double c = (j >= 0) ? (q_best[i] * invQ[i]) : kNegInf;
        if (j >= 0 && std::isfinite(c) && (c >= corr_thresh - 1e-6 || c > 1.0)) {
            double c_ex = exact_cov_scalar(uQ + i, muQ[i], uR + j, muR[j], m) * invQ[i] * invR[j];
            if (check_neg) c_ex = std::abs(c_ex);
            c = c_ex;
        }
        if (c > 1.0) c = 1.0;
        if (c < -1.5 || !std::isfinite(c)) c = kQuietNaN;
        out_best_corr[i] = c;
        out_best_ref_idx[i] = j;
        if (std::isfinite(c) && c >= corr_thresh) ++hits;
    }
    if (out_hit_count) *out_hit_count = hits;
    return 0;
}

int fmp_mstump(
    const double* T,
    int64_t D,
    int64_t N,
    int64_t m,
    int64_t ez,
    int n_threads,
    double* out_mp,
    int64_t* out_idx
) {
    if (!T || !out_mp || !out_idx || D < 1 || m < 3 || N < m) return -1;
    int64_t excl = (ez < 0) ? fmp_default_excl_zone(m) : ez;
    int64_t n = N - m + 1;
    int nt = resolve_threads(n_threads);

    std::vector<SeriesPrep> ch(static_cast<size_t>(D));
    bool any_flat = false;
    bool any_drops = false;
    for (int64_t d = 0; d < D; ++d) {
        ch[static_cast<size_t>(d)] = precompute_series(T + d * N, N, m);
        if (ch[static_cast<size_t>(d)].has_any_flat) any_flat = true;
        if (ch[static_cast<size_t>(d)].has_drops) any_drops = true;
    }

    int64_t d_min = excl + 1;
    if (d_min >= n) {
        for (int64_t k = 0; k < D * n; ++k) {
            out_mp[k] = kPosInf;
            out_idx[k] = -1;
        }
        return 0;
    }

    // Prepare per-channel pointers (and flat-handling arrays only when any channel has flat segments)
    std::vector<const double*> u_ptrs(static_cast<size_t>(D));
    std::vector<const double*> mu_ptrs(static_cast<size_t>(D));
    std::vector<const double*> inv_ptrs(static_cast<size_t>(D));
    std::vector<const double*> df_ptrs(static_cast<size_t>(D));
    std::vector<const double*> dg_ptrs(static_cast<size_t>(D));
    std::vector<const double*> flat_ptrs(static_cast<size_t>(D), nullptr);
    std::vector<const uint8_t*> drop_ptrs(static_cast<size_t>(D), nullptr);
    std::vector<int64_t> any_drop_prefix;

    std::vector<std::vector<double>> inv_ms;
    std::vector<std::vector<double>> flat_half;
    if (any_flat) {
        inv_ms.resize(static_cast<size_t>(D));
        flat_half.resize(static_cast<size_t>(D));
        for (int64_t c = 0; c < D; ++c) {
            const auto& S = ch[static_cast<size_t>(c)];
            const double* s_inv = S.inv_ptr();
            inv_ms[static_cast<size_t>(c)].resize(static_cast<size_t>(n));
            flat_half[static_cast<size_t>(c)].resize(static_cast<size_t>(n), 0.0);
            for (int64_t i = 0; i < n; ++i) {
                size_t ui = static_cast<size_t>(i);
                if (S.is_nan[ui]) {
                    inv_ms[static_cast<size_t>(c)][ui] = kQuietNaN;
                    flat_half[static_cast<size_t>(c)][ui] = 0.0;
                } else if (S.is_flat[ui]) {
                    inv_ms[static_cast<size_t>(c)][ui] = 0.0;
                    flat_half[static_cast<size_t>(c)][ui] = 0.5;
                } else {
                    inv_ms[static_cast<size_t>(c)][ui] = s_inv[i];
                    flat_half[static_cast<size_t>(c)][ui] = 0.0;
                }
            }
        }
    }

    for (int64_t c = 0; c < D; ++c) {
        size_t uc = static_cast<size_t>(c);
        u_ptrs[uc] = ch[uc].u_ptr();
        mu_ptrs[uc] = ch[uc].mu_ptr();
        inv_ptrs[uc] = any_flat ? inv_ms[uc].data() : ch[uc].inv_ptr();
        df_ptrs[uc] = ch[uc].df_ptr();
        dg_ptrs[uc] = ch[uc].dg_ptr();
        flat_ptrs[uc] = any_flat ? flat_half[uc].data() : nullptr;
        drop_ptrs[uc] = ch[uc].has_drops ? ch[uc].is_drop.data() : nullptr;
    }

    if (any_drops) {
        any_drop_prefix.assign(static_cast<size_t>(n + 1), 0);
        for (int64_t i = 0; i < n; ++i) {
            int64_t dflag = 0;
            for (int64_t c = 0; c < D; ++c) {
                if (drop_ptrs[static_cast<size_t>(c)] &&
                    drop_ptrs[static_cast<size_t>(c)][static_cast<size_t>(i)]) {
                    dflag = 1;
                    break;
                }
            }
            any_drop_prefix[static_cast<size_t>(i + 1)] =
                any_drop_prefix[static_cast<size_t>(i)] + dflag;
        }
    }

    bool any_invalid = false;
    for (int64_t c = 0; c < D; ++c) {
        if (ch[static_cast<size_t>(c)].has_any_invalid) {
            any_invalid = true;
            break;
        }
    }
    bool clean_f32 = (!any_flat && !any_invalid);
    for (int64_t c = 0; c < D && clean_f32; ++c) {
        const double* iv = ch[static_cast<size_t>(c)].inv_ptr();
        for (int64_t i = 0; i < n; ++i) {
            double v = iv[i];
            if (!(v >= 1e-15 && v <= 1e15)) {
                clean_f32 = false;
                break;
            }
        }
    }

    MstumpChannels mch;
    mch.D = D;
    mch.n = n;
    mch.m = m;
    mch.clean_f32 = clean_f32;
    mch.u = u_ptrs.data();
    mch.mu = mu_ptrs.data();
    mch.inv = inv_ptrs.data();
    mch.df = df_ptrs.data();
    mch.dg = dg_ptrs.data();
    mch.flat = any_flat ? flat_ptrs.data() : nullptr;
    mch.is_drop = any_drops ? drop_ptrs.data() : nullptr;
    mch.any_drop_prefix = any_drops ? any_drop_prefix.data() : nullptr;

    int actual_nt = std::min<int>(nt, static_cast<int>(std::max<int64_t>(1, (n - d_min + kMsTileW - 1) / kMsTileW)));
    std::atomic<int64_t> next_db(d_min);

    auto run_worker = [&](double* mp_buf, int64_t* idx_buf) {
        if (any_flat) {
            dispatch_mstump_worker<true>(mch, next_db, n, mp_buf, idx_buf);
        } else {
            dispatch_mstump_worker<false>(mch, next_db, n, mp_buf, idx_buf);
        }
    };

    for (int64_t k = 0; k < D * n; ++k) {
        out_mp[k] = kPosInf;
        out_idx[k] = -1;
    }

    if (actual_nt <= 1 || (n < 512 && D * n * n < 2000000LL)) {
        run_worker(out_mp, out_idx);
    } else {
        std::vector<std::vector<double>> t_mp(static_cast<size_t>(actual_nt), std::vector<double>(static_cast<size_t>(D * n), kPosInf));
        std::vector<std::vector<int64_t>> t_ix(static_cast<size_t>(actual_nt), std::vector<int64_t>(static_cast<size_t>(D * n), -1));
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(actual_nt - 1));
        for (int t = 1; t < actual_nt; ++t) {
            workers.emplace_back([&, t]() {
                run_worker(t_mp[static_cast<size_t>(t)].data(), t_ix[static_cast<size_t>(t)].data());
            });
        }
        run_worker(out_mp, out_idx);
        for (auto& w : workers) w.join();
        for (int t = 1; t < actual_nt; ++t) {
            const auto& mp = t_mp[static_cast<size_t>(t)];
            const auto& ix = t_ix[static_cast<size_t>(t)];
            for (int64_t k = 0; k < D * n; ++k) {
                size_t uk = static_cast<size_t>(k);
                if (mp[uk] < out_mp[k] ||
                    (mp[uk] == out_mp[k] && ix[uk] >= 0 && (out_idx[k] < 0 || ix[uk] < out_idx[k]))) {
                    out_mp[k] = mp[uk];
                    out_idx[k] = ix[uk];
                }
            }
        }
    }

    // Scale accumulated sum_{p=0..k} sqrt(u_(p)) by sqrt(2*m) / (k + 1)
    const double sqrt_2m = std::sqrt(2.0 * static_cast<double>(m));
    for (int64_t k = 0; k < D; ++k) {
        const double scale = sqrt_2m / static_cast<double>(k + 1);
        double* row_mp = out_mp + k * n;
        for (int64_t i = 0; i < n; ++i) {
            double v = row_mp[i];
            row_mp[i] = std::isfinite(v) ? (v * scale) : kPosInf;
        }
    }
    return 0;
}

int fmp_floss_cac(
    const int64_t* idx,
    int64_t n,
    int64_t L,
    double* out_cac
) {
    if (!idx || !out_cac || n <= 0) return -1;
    std::vector<int64_t> diff(static_cast<size_t>(n + 1), 0);
    int64_t valid_arcs = 0;
    for (int64_t i = 0; i < n; ++i) {
        int64_t j = idx[i];
        if (j < 0 || j >= n || j == i) continue;
        int64_t lo = std::min(i, j);
        int64_t hi = std::max(i, j);
        diff[static_cast<size_t>(lo)] += 1;
        diff[static_cast<size_t>(hi)] -= 1;
        ++valid_arcs;
    }
    if (valid_arcs == 0) {
        std::fill_n(out_cac, n, 1.0);
        return 0;
    }
    int64_t acc = 0;
    const double dn = static_cast<double>(n);
    const int64_t half_n = n / 2;
    const int64_t excl = (L <= 0) ? std::min<int64_t>(half_n, 1)
                                  : ((L > half_n / 5) ? half_n : std::max<int64_t>(1, 5 * L));
    for (int64_t i = 0; i < n; ++i) {
        acc += diff[static_cast<size_t>(i)];
        if (i < excl || i >= n - excl) {
            out_cac[i] = 1.0;
        } else {
            double iac = 2.0 * static_cast<double>(i) * (dn - static_cast<double>(i)) / dn;
            double v = (iac > 1e-12) ? (static_cast<double>(acc) / iac) : 1.0;
            out_cac[i] = std::min(1.0, std::max(0.0, v));
        }
    }
    return 0;
}

int64_t fmp_auto_window(
    const double* T,
    int64_t N,
    int64_t min_m,
    int64_t max_m
) {
    if (!T || N < 3) return 3;
    int64_t hi = std::clamp<int64_t>(max_m, 3, N);
    int64_t lo = std::clamp<int64_t>(min_m, 3, hi);
    if (N < 16) {
        return std::clamp<int64_t>(std::max<int64_t>(3, N / 2), lo, hi);
    }
    int64_t cap_m = std::min<int64_t>(hi, std::max<int64_t>(lo, N / 4));
    if (cap_m <= lo) return lo;

    // Sample up to 4096 trailing points for fast O(L*M) ACF peak finding
    int64_t len = std::min<int64_t>(N, 4096);
    const double* x = T + (N - len);
    double sum = 0.0;
    int64_t fcnt = 0;
    for (int64_t i = 0; i < len; ++i) {
        if (std::isfinite(x[i])) { sum += x[i]; ++fcnt; }
    }
    if (fcnt < 16) return lo;
    double mean = sum / static_cast<double>(fcnt);
    std::vector<double> z(static_cast<size_t>(len), 0.0);
    double var = 0.0;
    for (int64_t i = 0; i < len; ++i) {
        if (std::isfinite(x[i])) {
            double d = x[i] - mean;
            z[static_cast<size_t>(i)] = d;
            var += d * d;
        }
    }
    if (var <= 1e-30) return std::min<int64_t>(cap_m, std::max<int64_t>(lo, 32));

    int64_t max_lag = std::min<int64_t>(cap_m, len / 3);
    if (max_lag <= lo) return lo;
    std::vector<double> acf(static_cast<size_t>(max_lag + 1), 0.0);
    for (int64_t lag = 1; lag <= max_lag; ++lag) {
        double dot = 0.0;
        int64_t lim = len - lag;
        const double* p0 = z.data();
        const double* p1 = z.data() + lag;
        for (int64_t i = 0; i < lim; ++i) dot += p0[i] * p1[i];
        acf[static_cast<size_t>(lag)] = dot / var;
    }

    // Find first strong local maximum past the first zero/trough crossing
    bool passed_trough = false;
    int64_t best_lag = -1;
    double best_val = 0.15;
    for (int64_t lag = std::max<int64_t>(2, lo); lag < max_lag; ++lag) {
        double prev = acf[static_cast<size_t>(lag - 1)];
        double cur = acf[static_cast<size_t>(lag)];
        double next = acf[static_cast<size_t>(lag + 1)];
        if (cur < 0.2 || cur < prev) passed_trough = true;
        if (passed_trough && cur > prev && cur >= next && cur > best_val) {
            best_val = cur;
            best_lag = lag;
            if (cur > 0.5) break; // Strong fundamental period found
        }
    }
    if (best_lag >= lo) return best_lag;
    // Fallback heuristic: sensible scale proportional to sqrt(N)
    int64_t fb = static_cast<int64_t>(std::sqrt(static_cast<double>(N)));
    return std::clamp(fb, lo, cap_m);
}

// ---------------------------------------------------------------------------
// RingMP Streaming C API
// ---------------------------------------------------------------------------
fmp_stream_t* fmp_stream_create(
    int64_t m,
    int64_t window_subseqs,
    int64_t ez,
    int egress,
    double alpha,
    int track_full_mp
) {
    constexpr int64_t kMaxStreamDim = (1LL << 26); // 67M guard against signed shift overflow & OOM
    if (m < 3 || m > kMaxStreamDim || window_subseqs < 4 || window_subseqs > kMaxStreamDim) {
        return nullptr;
    }
    auto* s = new fmp_stream();
    s->m = m;
    s->W = window_subseqs;
    s->ez = (ez < 0) ? fmp_default_excl_zone(m) : ez;
    s->egress = egress ? 1 : 0;
    s->alpha = (alpha > 0.0 && alpha < 1.0) ? alpha : 0.05;
    s->track_full_mp = track_full_mp ? 1 : 0;

    int64_t c = 1;
    while (c < window_subseqs + 4) c <<= 1;

    int64_t rc = 1;
    while (rc < window_subseqs + m + 4) rc <<= 1;
    s->raw_cap = rc;
    s->raw_mask = rc - 1;

    s->u_ring.assign(static_cast<size_t>(2 * rc), 0.0);
    s->bad_ring.assign(static_cast<size_t>(2 * rc), 0);
    s->cal_delay.assign(static_cast<size_t>(m), 0.0);
    static constexpr double kLams[3] = {0.10, 0.30, 0.60};
    for (int b = 0; b < 3; ++b) {
        s->e_decay[b] = std::pow(1.0 - kLams[b], 1.0 / static_cast<double>(m));
    }
    s->init_slabs(c);
    return s;
}

void fmp_stream_destroy(fmp_stream_t* s) {
    delete s;
}

int fmp_stream_update(fmp_stream_t* s, double x_new, fmp_tick_result_t* out) {
    if (!s) return -1;
    fmp_tick_result_t local_out{};
    fmp_tick_result_t* res = out ? out : &local_out;
    res->gid = -1;
    res->left_dist = kPosInf;
    res->left_idx = -1;
    res->discord_z = 0.0;
    res->p_value = 1.0;
    res->e_value = s->e_wealth;
    res->is_alert = 0;
    res->n_repaired = 0;

    // Grow raw ring BEFORE writing if egress==0 and approaching capacity
    if (!s->egress && s->n_raw + 2 >= s->raw_cap) {
        s->grow_raw_capacity();
    }

    // 1. Ingest raw tick into mirrored ring buffer
    const int64_t ridx = s->n_raw;
    const int64_t rpos = ridx & s->raw_mask;
    uint8_t is_bad = (std::isfinite(x_new) && std::abs(x_new) <= 1e150) ? 0 : 1;
    if (!is_bad && !s->has_base) {
        s->base = x_new;
        s->has_base = 1;
    }
    double u_val = 0.0;
    if (!is_bad) {
        u_val = x_new - s->base;
        if (!std::isfinite(u_val) || std::abs(u_val) > 1e150) {
            is_bad = 1;
            u_val = 0.0;
        }
    }

    // Maintain exact integer bad count in trailing m-tick window
    if (ridx >= s->m) {
        s->bad_in_cur_win -= s->bad_ring[static_cast<size_t>((ridx - s->m) & s->raw_mask)];
    }
    s->bad_in_cur_win += is_bad;

    s->u_ring[static_cast<size_t>(rpos)] = u_val;
    s->u_ring[static_cast<size_t>(rpos + s->raw_cap)] = u_val;
    s->bad_ring[static_cast<size_t>(rpos)] = is_bad;
    s->bad_ring[static_cast<size_t>(rpos + s->raw_cap)] = is_bad;
    s->n_raw += 1;

    if (s->n_raw < s->m) {
        return 0; // Still warming up first window
    }

    // 2. New subsequence gid = g
    const int64_t g = s->n_sub;
    s->n_sub += 1;

    // Grow subsequence slabs BEFORE accessing slot_g if egress==0 and g approaches capacity
    if (!s->egress && s->n_sub + 2 >= s->cap) {
        s->grow_subseq_capacity(g);
    }

    const int64_t slot_g = g & s->mask;
    s->compute_subseq_stats(g);

    // Reset right-MP & left-MP state for new slot
    s->right_corr[slot_g] = kNegInf;
    s->right_idx[slot_g] = -1;
    s->left_top_corr[slot_g] = kNegInf;
    s->left_top_idx[slot_g] = -1;
    s->stack_len[static_cast<size_t>(slot_g)] = 0;
    s->stack_trunc[static_cast<size_t>(slot_g)] = 0;

    // Compute df[g-1], dg[g-1] now that mu[g] is available
    if (g >= 1) {
        int64_t pslot = (g - 1) & s->mask;
        double x_out = s->u_ring[static_cast<size_t>((g - 1) & s->raw_mask)];
        double x_in = s->u_ring[static_cast<size_t>((g - 1 + s->m) & s->raw_mask)];
        s->df[pslot] = 0.5 * (x_in - x_out);
        s->dg[pslot] = (x_in - s->mu[slot_g]) + (x_out - s->mu[pslot]);
    }

    const int64_t gid_min = (s->egress && g >= s->W) ? (g - s->W + 1) : 0;
    const int64_t k_hi = g - s->ez - 1;
    const double inv_g = s->inv_norm[slot_g];
    const bool valid_g = (s->valid[static_cast<size_t>(slot_g)] != 0);

    bool g_dropped = false;
    if (valid_g) {
        double norm_g = 1.0 / inv_g;
        if (s->norm_hwm > 32.0 * norm_g || (g > 0 && !s->prev_valid)) {
            g_dropped = true;
            s->norm_hwm = norm_g;
            s->active_drops.push_back(g);
        } else if (norm_g > s->norm_hwm) {
            s->norm_hwm = norm_g;
        }
    }
    s->prev_valid = valid_g ? 1 : 0;
    if (__builtin_expect(!s->active_drops.empty(), 0)) {
        size_t keep = 0;
        while (keep < s->active_drops.size() && s->active_drops[keep] < gid_min) {
            ++keep;
        }
        if (keep > 0) {
            s->active_drops.erase(s->active_drops.begin(), s->active_drops.begin() + static_cast<ptrdiff_t>(keep));
        }
    }

    double best_left_corr = kNegInf;
    int64_t best_left_idx = -1;

    if (k_hi >= gid_min) {
        const double df_prev = (g >= 1) ? s->df[(g - 1) & s->mask] : 0.0;
        const double dg_prev = (g >= 1) ? s->dg[(g - 1) & s->mask] : 0.0;
        const int64_t k_loop_lo = (gid_min == 0) ? 1 : gid_min;

        auto dispatch_sweep = [&]<bool TrackFull, bool BuildStack, bool ValidG>() {
            double best_val = TrackFull ? kNegInf : -kPosInf;
            int64_t best_idx = -1;
            StackEntry* st = BuildStack ? (s->left_stacks.data() + slot_g * kStackCap) : nullptr;
            uint8_t st_len = 0;
            uint8_t st_trunc = 0;

            auto eval_exact_point = [&](int64_t k) {
                int64_t ks = k & s->mask;
                double c_ex = s->exact_cov_gids(g, k);
                s->cov[ks] = c_ex;
                if constexpr (ValidG) {
                    if constexpr (!TrackFull) {
                        double sv = c_ex * s->inv_norm[ks];
                        if (sv > best_val) {
                            best_val = sv;
                            best_idx = k;
                        }
                    } else {
                        double corr = (c_ex * s->inv_norm[ks]) * inv_g;
                        if (corr > best_val) {
                            best_val = corr;
                            best_idx = k;
                            if constexpr (BuildStack) {
                                fmp_stream::push_left_stack_entry(st, st_len, st_trunc, k, corr);
                            }
                        }
                        if (corr > s->right_corr[ks]) {
                            s->right_corr[ks] = corr;
                            s->right_idx[ks] = g;
                        }
                    }
                }
            };

            if (__builtin_expect(g_dropped, 0)) {
                // Spike exited current subsequence g: exact right-to-left covariance re-anchor
                // eliminates all O(spike^2 * eps) cancellation garbage across all active lags.
                for (int64_t k = k_hi; k >= gid_min; --k) {
                    eval_exact_point(k);
                }
            } else {
                // Strictly right-to-left order [k_hi .. gid_min] with in-place cov update & round-robin re-anchor:
                //   1) [re_k + 1 .. k_hi] (reads cov_old[re_k] before re_k is overwritten!)
                //   2) exact re_k (plus any active spike-drop k in [k_loop_lo, k_hi])
                //   3) [k_loop_lo .. re_k - 1]
                //   4) exact k == 0 (if gid_min == 0)
                if (k_loop_lo <= k_hi) {
                    int64_t total_len = k_hi - k_loop_lo + 1;
                    int64_t re_k = k_loop_lo + (g % total_len);
                    if (__builtin_expect(s->active_drops.empty(), 1)) {
                        if (re_k < k_hi) {
                            s->sweep_range_rtl<TrackFull, BuildStack, ValidG>(
                                re_k + 1, k_hi, g, df_prev, dg_prev, inv_g,
                                best_val, best_idx, st, st_len, st_trunc);
                        }
                        eval_exact_point(re_k);
                        if (k_loop_lo < re_k) {
                            s->sweep_range_rtl<TrackFull, BuildStack, ValidG>(
                                k_loop_lo, re_k - 1, g, df_prev, dg_prev, inv_g,
                                best_val, best_idx, st, st_len, st_trunc);
                        }
                    } else {
                        auto& splits = s->split_scratch;
                        splits.clear();
                        bool inserted_re = false;
                        for (int64_t dk : s->active_drops) {
                            if (dk < k_loop_lo) continue;
                            if (dk > k_hi) break;
                            if (!inserted_re && re_k < dk) {
                                splits.push_back(re_k);
                                inserted_re = true;
                            } else if (!inserted_re && re_k == dk) {
                                inserted_re = true;
                            }
                            splits.push_back(dk);
                        }
                        if (!inserted_re) splits.push_back(re_k);

                        int64_t hi = k_hi;
                        for (int64_t si = static_cast<int64_t>(splits.size()) - 1; si >= 0; --si) {
                            int64_t sp = splits[static_cast<size_t>(si)];
                            if (sp < hi) {
                                s->sweep_range_rtl<TrackFull, BuildStack, ValidG>(
                                    sp + 1, hi, g, df_prev, dg_prev, inv_g,
                                    best_val, best_idx, st, st_len, st_trunc);
                            }
                            eval_exact_point(sp);
                            hi = sp - 1;
                        }
                        if (k_loop_lo <= hi) {
                            s->sweep_range_rtl<TrackFull, BuildStack, ValidG>(
                                k_loop_lo, hi, g, df_prev, dg_prev, inv_g,
                                best_val, best_idx, st, st_len, st_trunc);
                        }
                    }
                }
                if (gid_min == 0) {
                    eval_exact_point(0);
                }
            }

            if constexpr (BuildStack) {
                s->stack_len[static_cast<size_t>(slot_g)] = st_len;
                s->stack_trunc[static_cast<size_t>(slot_g)] = st_trunc;
                if (st_len > 0) {
                    best_left_corr = st[st_len - 1].corr;
                    best_left_idx = st[st_len - 1].idx;
                }
            } else if constexpr (ValidG) {
                if constexpr (TrackFull) {
                    best_left_corr = best_val;
                    best_left_idx = best_idx;
                } else {
                    best_left_corr = (best_idx >= 0) ? (best_val * inv_g) : kNegInf;
                    best_left_idx = best_idx;
                }
            }
            s->left_top_corr[slot_g] = best_left_corr;
            s->left_top_idx[slot_g] = best_left_idx;
        };

        if (!valid_g) {
            dispatch_sweep.template operator()<false, false, false>();
        } else if (!s->track_full_mp) {
            dispatch_sweep.template operator()<false, false, true>();
        } else if (s->egress) {
            dispatch_sweep.template operator()<true, true, true>();
        } else {
            dispatch_sweep.template operator()<true, false, true>();
        }
    }

    // 3. Exact O(1)-amortized Egress Pop (when gid_exp = g - W leaves the sliding window)
    int32_t n_repaired = 0;
    if (s->egress && s->track_full_mp && g >= s->W) {
        const int64_t gid_exp = g - s->W;
        int64_t* lidx = s->left_top_idx;
        double* lcorr = s->left_top_corr;

        auto pop_slot = [&](int64_t k) {
            int64_t sl = k & s->mask;
            uint8_t len = s->stack_len[static_cast<size_t>(sl)];
            StackEntry* st = s->left_stacks.data() + sl * kStackCap;
            while (len > 0 && st[len - 1].idx <= gid_exp) {
                --len;
            }
            s->stack_len[static_cast<size_t>(sl)] = len;
            if (len > 0) {
                lcorr[sl] = st[len - 1].corr;
                lidx[sl] = st[len - 1].idx;
            } else if (s->stack_trunc[static_cast<size_t>(sl)]) {
                s->rebuild_left_stack_exact(k, gid_min);
                ++n_repaired;
            } else {
                lcorr[sl] = kNegInf;
                lidx[sl] = -1;
            }
        };

        // Fast 8-wide AVX2 scan over active left_top_idx comparing == gid_exp
        auto scan_egress_seg = [&](int64_t k_start, int64_t seg_len) {
            int64_t s0 = k_start & s->mask;
            int64_t pos = 0;
#if defined(FMP_HAS_AVX2)
            __m256i v_exp = _mm256_set1_epi64x(gid_exp);
            for (; pos + 8 <= seg_len; pos += 8) {
                __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lidx + s0 + pos));
                __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lidx + s0 + pos + 4));
                __m256i eq0 = _mm256_cmpeq_epi64(v0, v_exp);
                __m256i eq1 = _mm256_cmpeq_epi64(v1, v_exp);
                __m256i eq_any = _mm256_or_si256(eq0, eq1);
                if (__builtin_expect(_mm256_movemask_pd(_mm256_castsi256_pd(eq_any)) != 0, 0)) {
                    int m0 = _mm256_movemask_pd(_mm256_castsi256_pd(eq0));
                    while (m0) {
                        int lane = __builtin_ctz(static_cast<unsigned>(m0));
                        m0 &= (m0 - 1);
                        pop_slot(k_start + pos + lane);
                    }
                    int m1 = _mm256_movemask_pd(_mm256_castsi256_pd(eq1));
                    while (m1) {
                        int lane = __builtin_ctz(static_cast<unsigned>(m1));
                        m1 &= (m1 - 1);
                        pop_slot(k_start + pos + 4 + lane);
                    }
                }
            }
            for (; pos + 4 <= seg_len; pos += 4) {
                __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lidx + s0 + pos));
                __m256i eq0 = _mm256_cmpeq_epi64(v0, v_exp);
                int m0 = _mm256_movemask_pd(_mm256_castsi256_pd(eq0));
                if (__builtin_expect(m0 != 0, 0)) {
                    while (m0) {
                        int lane = __builtin_ctz(static_cast<unsigned>(m0));
                        m0 &= (m0 - 1);
                        pop_slot(k_start + pos + lane);
                    }
                }
            }
#endif
            for (; pos < seg_len; ++pos) {
                if (lidx[s0 + pos] == gid_exp) {
                    pop_slot(k_start + pos);
                }
            }
        };

        // Active k whose left_top_idx could equal gid_exp satisfy k >= gid_exp + ez + 1
        int64_t ek_lo = gid_exp + s->ez + 1;
        if (ek_lo <= g) {
            int64_t elen = g - ek_lo + 1;
            int64_t es0 = ek_lo & s->mask;
            int64_t eseg1 = std::min(elen, s->cap - es0);
            scan_egress_seg(ek_lo, eseg1);
            if (eseg1 < elen) {
                scan_egress_seg(ek_lo + eseg1, elen - eseg1);
            }
        }
    }

    // 4. Emit exact causal tick result + Ville E-process update
    res->gid = g;
    res->left_dist = (__builtin_expect(best_left_corr > 1.0 - 1e-10 && best_left_idx >= 0, 0))
        ? exact_z_dist_scalar(
            s->u_ring.data() + (g & s->raw_mask),
            s->u_ring.data() + (best_left_idx & s->raw_mask),
            s->m)
        : corr_to_dist(best_left_corr, s->m);
    res->left_idx = best_left_idx;
    res->n_repaired = n_repaired;
    s->update_eprocess(g, res->left_dist, res);
    return 0;
}

int fmp_stream_init(fmp_stream_t* s, const double* initial_data, int64_t n_init) {
    if (!s || !initial_data || n_init < 0) return -1;
    // If initial_data fits cleanly and is long, ingest via fmp_stream_update.
    // Even for W=4096, 4096 streaming updates take ~10 ms total and build exact suffix-max stacks!
    fmp_tick_result_t tmp{};
    for (int64_t i = 0; i < n_init; ++i) {
        int rc = fmp_stream_update(s, initial_data[i], &tmp);
        if (rc != 0) return rc;
    }
    return 0;
}

int fmp_stream_update_batch(
    fmp_stream_t* s,
    const double* xs,
    int64_t count,
    double* out_left_dist,
    int64_t* out_left_idx,
    double* out_e_value,
    int32_t* out_is_alert
) {
    if (!s || !xs || count < 0) return -1;
    fmp_tick_result_t r{};
    for (int64_t i = 0; i < count; ++i) {
        int rc = fmp_stream_update(s, xs[i], &r);
        if (rc != 0) return rc;
        if (out_left_dist) out_left_dist[i] = r.left_dist;
        if (out_left_idx) out_left_idx[i] = r.left_idx;
        if (out_e_value) out_e_value[i] = r.e_value;
        if (out_is_alert) out_is_alert[i] = r.is_alert;
    }
    return 0;
}

int64_t fmp_stream_active_len(const fmp_stream_t* s) {
    if (!s || s->n_sub <= 0) return 0;
    if (s->egress) return std::min(s->n_sub, s->W);
    return s->n_sub;
}

int64_t fmp_stream_gid_min(const fmp_stream_t* s) {
    if (!s || s->n_sub <= 0) return 0;
    if (s->egress && s->n_sub > s->W) return s->n_sub - s->W;
    return 0;
}

int fmp_stream_snapshot(
    const fmp_stream_t* s,
    int relative_indices,
    double* out_mp,
    int64_t* out_idx,
    double* out_lmp,
    int64_t* out_lidx
) {
    if (!s || !out_mp || !out_idx) return -1;
    int64_t active = fmp_stream_active_len(s);
    int64_t gmin = fmp_stream_gid_min(s);
    for (int64_t i = 0; i < active; ++i) {
        int64_t gid = gmin + i;
        int64_t sl = gid & s->mask;
        double lc = s->left_top_corr[static_cast<size_t>(sl)];
        int64_t li = s->left_top_idx[static_cast<size_t>(sl)];
        if (li >= 0 && li < gmin) {
            li = -1;
            lc = kNegInf;
        }
        double rc = s->track_full_mp ? s->right_corr[static_cast<size_t>(sl)] : kNegInf;
        int64_t ri = s->track_full_mp ? s->right_idx[static_cast<size_t>(sl)] : -1;

        double ld = (__builtin_expect(lc > 1.0 - 1e-10 && li >= 0, 0))
            ? exact_z_dist_scalar(
                s->u_ring.data() + (gid & s->raw_mask),
                s->u_ring.data() + (li & s->raw_mask),
                s->m)
            : corr_to_dist(lc, s->m);
        double rd = (__builtin_expect(rc > 1.0 - 1e-10 && ri >= 0, 0))
            ? exact_z_dist_scalar(
                s->u_ring.data() + (gid & s->raw_mask),
                s->u_ring.data() + (ri & s->raw_mask),
                s->m)
            : corr_to_dist(rc, s->m);

        double bd = kPosInf;
        int64_t bi = -1;
        if (ld <= rd && li >= 0) {
            bd = ld;
            bi = li;
        } else if (ri >= 0) {
            bd = rd;
            bi = ri;
        }
        out_mp[i] = bd;
        out_idx[i] = (bi >= 0 && relative_indices) ? (bi - gmin) : bi;
        if (out_lmp) out_lmp[i] = ld;
        if (out_lidx) out_lidx[i] = (li >= 0 && relative_indices) ? (li - gmin) : li;
    }
    return 0;
}

} // extern "C"
