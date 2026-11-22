#ifndef FLASHMP_HPP
#define FLASHMP_HPP

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Library metadata
const char* fmp_version(void);
const char* fmp_simd_isa(void);

// Default exclusion zone helper matching STUMPY: ceil(m / 4)
// Valid neighbor pairs satisfy |i - j| > excl_zone (i.e. |i - j| >= excl_zone + 1)
static inline int64_t fmp_default_excl_zone(int64_t m) {
    return (m + 3) / 4;
}

// ---------------------------------------------------------------------------
// 1. Exact Batch Self-Join (Full MP + optional Causal Left-MP + Right-MP)
//    All computed in a single L1-tiled AVX2/FMA or NEON SCAMP pass.
//    - T: length N double array
//    - m: subsequence length (m >= 3, m <= N)
//    - ez: exclusion zone (if ez < 0, defaults to ceil(m / 4));
//          pairs (i, j) are valid iff |i - j| > ez.
//    - n_threads: <= 0 uses hardware_concurrency()
//    - out_mp, out_idx: length n = N - m + 1 (required, non-null)
//    - out_lmp, out_lidx: optional causal Left-MP (pass NULL to skip write)
//    - out_rmp, out_ridx: optional Right-MP (pass NULL to skip write)
//    Returns 0 on success, non-zero on invalid arguments.
// ---------------------------------------------------------------------------
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
);

// ---------------------------------------------------------------------------
// 2. Exact Batch AB-Join (A->B and optional B->A in a single tiled pass)
//    - A: length NA, B: length NB, m >= 3
//    - out_mp_a, out_idx_a: length nA = NA - m + 1 (required)
//    - out_mp_b, out_idx_b: length nB = NB - m + 1 (optional, may be NULL)
//    Returns 0 on success.
// ---------------------------------------------------------------------------
int fmp_ab_join(
    const double* A,
    int64_t NA,
    const double* B,
    int64_t NB,
    int64_t m,
    int n_threads,
    double* out_mp_a,
    int64_t* out_idx_a,
    double* out_mp_b,
    int64_t* out_idx_b
);

// ---------------------------------------------------------------------------
// 3. Affine-Invariant Subsequence Contamination Scanner (LeakScan)
//    Scans every query subsequence in qry (len N_qry) against ref (len N_ref).
//    - check_neg: if non-zero, matches |corr| >= corr_thresh (catches sign flips)
//    - out_best_corr: length n_qry = N_qry - m + 1 (best signed or abs corr)
//    - out_best_ref_idx: length n_qry (matching index in ref, or -1)
//    - out_hit_count: number of query subsequences with score >= corr_thresh
//    Returns 0 on success.
// ---------------------------------------------------------------------------
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
);

// ---------------------------------------------------------------------------
// 4. Exact Multivariate Matrix Profile (mSTUMP / Yeh et al. ICDM 2017)
//    - T: row-major D x N double matrix
//    - out_mp:  row-major D x n (where n = N - m + 1), k-th row = (k+1)-dim MP
//    - out_idx: row-major D x n, nearest neighbor index for each subspace k
//    Returns 0 on success.
// ---------------------------------------------------------------------------
int fmp_mstump(
    const double* T,
    int64_t D,
    int64_t N,
    int64_t m,
    int64_t ez,
    int n_threads,
    double* out_mp,
    int64_t* out_idx
);

// ---------------------------------------------------------------------------
// 5. Exact O(n) FLOSS/FLUSS Corrected Arc Curve (Semantic Regime Segmentation)
//    - idx: length n matrix profile neighbor indices
//    - L: expected subsequence/regime scale (typically m)
//    - out_cac: length n double array in [0, 1]; deep valleys = regime changes
//    Returns 0 on success.
// ---------------------------------------------------------------------------
int fmp_floss_cac(
    const int64_t* idx,
    int64_t n,
    int64_t L,
    double* out_cac
);

// ---------------------------------------------------------------------------
// 6. Fast Causal Autocorrelation Period Selector
//    Returns recommended window length m in [min_m, max_m].
// ---------------------------------------------------------------------------
int64_t fmp_auto_window(
    const double* T,
    int64_t N,
    int64_t min_m,
    int64_t max_m
);

// ---------------------------------------------------------------------------
// 7. RingMP: Exact Bounded-Window Streaming Matrix Profile + Ville E-Process
//    Uses Monotone Suffix-Max Left-Stack for O(1)-amortized exact egress,
//    ping-pong contiguous AVX2/FMA df/dg updates, and round-robin re-anchoring.
// ---------------------------------------------------------------------------
typedef struct fmp_stream fmp_stream_t;

typedef struct fmp_tick_result {
    int64_t gid;         // Global subsequence index (0-based), -1 if warming up
    double  left_dist;   // Exact causal Left-MP distance within active window
    int64_t left_idx;    // Global index of exact causal nearest neighbor, or -1
    double  discord_z;   // Causal robust z-score of left_dist
    double  p_value;     // Causal empirical conformal p-value in (0, 1]
    double  e_value;     // Anytime-valid Ville supermartingale wealth W_t
    int32_t is_alert;    // 1 iff e_value >= 1/alpha
    int32_t n_repaired;  // Fallback repairs triggered (0 in normal operation)
} fmp_tick_result_t;

fmp_stream_t* fmp_stream_create(
    int64_t m,
    int64_t window_subseqs,
    int64_t ez,
    int egress,
    double alpha,
    int track_full_mp
);

void fmp_stream_destroy(fmp_stream_t* s);

int fmp_stream_init(
    fmp_stream_t* s,
    const double* initial_data,
    int64_t n_init
);

int fmp_stream_update(
    fmp_stream_t* s,
    double x_new,
    fmp_tick_result_t* out
);

int fmp_stream_update_batch(
    fmp_stream_t* s,
    const double* xs,
    int64_t count,
    double* out_left_dist,
    int64_t* out_left_idx,
    double* out_e_value,
    int32_t* out_is_alert
);

int64_t fmp_stream_active_len(const fmp_stream_t* s);
int64_t fmp_stream_gid_min(const fmp_stream_t* s);

// Copies current active window's exact Full-MP (and optional Left-MP).
// Arrays must have capacity >= fmp_stream_active_len(s).
// Indices are returned relative to the active window [0 .. active_len-1]
// if relative_indices != 0, else global subsequence IDs.
int fmp_stream_snapshot(
    const fmp_stream_t* s,
    int relative_indices,
    double* out_mp,
    int64_t* out_idx,
    double* out_lmp,
    int64_t* out_lidx
);

#ifdef __cplusplus
}
#endif

#endif // FLASHMP_HPP
