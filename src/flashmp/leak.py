"""Affine-invariant subsequence contamination / data-leakage scanner."""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import Any

from ._native import (
    _alloc_double,
    _alloc_int64,
    _as_c_double_array,
    _use_numpy,
    get_fast,
    get_lib,
)


@dataclass(slots=True)
class LeakExample:
    """Detailed affine fit for a single detected subsequence leak between ref and qry."""

    qry_idx: int
    ref_idx: int
    corr: float
    abs_corr: float
    alpha: float
    beta: float
    max_abs_err: float
    match_type: str  # "verbatim" | "affine" | "neg_affine"

    def to_dict(self) -> dict[str, Any]:
        return {
            "qry_idx": self.qry_idx,
            "ref_idx": self.ref_idx,
            "corr": self.corr,
            "abs_corr": self.abs_corr,
            "alpha": self.alpha,
            "beta": self.beta,
            "max_abs_err": self.max_abs_err,
            "match_type": self.match_type,
        }


@dataclass
class LeakReport:
    """Report returned by `scan_leakage(ref, qry, ...)`."""

    m: int
    corr_thresh: float
    check_neg: bool
    contaminated_windows: int
    total_windows: int
    contamination_rate: float
    examples: list[LeakExample]
    best_corr: Any = None
    best_ref_idx: Any = None

    @property
    def is_contaminated(self) -> bool:
        return self.contaminated_windows > 0

    def to_dict(self) -> dict[str, Any]:
        counts = {"verbatim": 0, "affine": 0, "neg_affine": 0}
        for ex in self.examples:
            if ex.match_type in counts:
                counts[ex.match_type] += 1
        return {
            "m": self.m,
            "corr_thresh": self.corr_thresh,
            "check_neg": self.check_neg,
            "is_contaminated": self.is_contaminated,
            "contaminated_windows": self.contaminated_windows,
            "total_windows": self.total_windows,
            "contamination_rate": self.contamination_rate,
            "example_type_counts": counts,
            "examples": [ex.to_dict() for ex in self.examples],
        }

    def summary(self) -> str:
        status = "CONTAMINATED" if self.is_contaminated else "CLEAN"
        pct = 100.0 * self.contamination_rate
        lines = [
            f"LEAK_SCAN {status}: {self.contaminated_windows}/{self.total_windows} "
            f"windows ({pct:.2f}%) matched at |r|>={self.corr_thresh:.4f} (m={self.m}, check_neg={self.check_neg})"
        ]
        for k, ex in enumerate(self.examples[:10]):
            sign = "+" if ex.beta >= 0 else "-"
            lines.append(
                f"  #{k} [{ex.match_type}] qry[{ex.qry_idx}:{ex.qry_idx + self.m}] "
                f"~= {ex.alpha:.6g} * ref[{ex.ref_idx}:{ex.ref_idx + self.m}] "
                f"{sign} {abs(ex.beta):.6g} (r={ex.corr:.6f}, max_err={ex.max_abs_err:.3e})"
            )
        if len(self.examples) > 10:
            lines.append(f"  ... +{len(self.examples) - 10} more examples")
        return "\n".join(lines)


def _fit_ols_example(
    ref_seq: Any,
    qry_seq: Any,
    ref_idx: int,
    qry_idx: int,
    m: int,
    fallback_score: float,
) -> LeakExample:
    """Compute exact OLS affine fit `qry ≈ alpha * ref + beta` and classify leak type."""
    sum_x = 0.0
    sum_y = 0.0
    for k in range(m):
        sum_x += float(ref_seq[ref_idx + k])
        sum_y += float(qry_seq[qry_idx + k])
    mx = sum_x / m
    my = sum_y / m

    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    for k in range(m):
        dx = float(ref_seq[ref_idx + k]) - mx
        dy = float(qry_seq[qry_idx + k]) - my
        sxx += dx * dx
        sxy += dx * dy
        syy += dy * dy

    if sxx > 0.0:
        alpha = sxy / sxx
        beta = my - alpha * mx
    else:
        alpha = 0.0
        beta = my

    max_abs_err = 0.0
    for k in range(m):
        pred = alpha * float(ref_seq[ref_idx + k]) + beta
        err = abs(float(qry_seq[qry_idx + k]) - pred)
        if err > max_abs_err:
            max_abs_err = err

    denom = math.sqrt(sxx * syy)
    if denom > 0.0:
        signed_r = max(-1.0, min(1.0, sxy / denom))
    else:
        signed_r = fallback_score

    if abs(alpha - 1.0) < 1e-6 and abs(beta) < 1e-6 and max_abs_err < 1e-6:
        match_type = "verbatim"
    elif alpha < 0.0:
        match_type = "neg_affine"
    else:
        match_type = "affine"

    return LeakExample(
        qry_idx=int(qry_idx),
        ref_idx=int(ref_idx),
        corr=float(signed_r),
        abs_corr=float(abs(signed_r)),
        alpha=float(alpha),
        beta=float(beta),
        max_abs_err=float(max_abs_err),
        match_type=match_type,
    )


def scan_leakage(
    ref: Any,
    qry: Any,
    m: int = 64,
    corr_thresh: float = 0.999,
    check_neg: bool = True,
    max_examples: int = 20,
    threads: int = 0,
) -> LeakReport:
    """Detect verbatim, affine-rescaled/shifted, and sign-flipped subsequence leakage.

    Scans every length-`m` window of `qry` (e.g. test/eval set) against `ref`
    (e.g. train/pretraining corpus) using SIMD z-normalized correlation.
    """
    ref_ptr, N_ref, ref_owner, is_np_ref = _as_c_double_array(ref)
    qry_ptr, N_qry, qry_owner, is_np_qry = _as_c_double_array(qry)

    m_int = int(m)
    if m_int < 3:
        raise ValueError(f"m must be >= 3, got {m_int}")
    if N_ref < m_int:
        raise ValueError(f"ref length ({N_ref}) must be >= m ({m_int})")
    if N_qry < m_int:
        raise ValueError(f"qry length ({N_qry}) must be >= m ({m_int})")

    n_qry = N_qry - m_int + 1
    thresh = float(corr_thresh)
    use_np = _use_numpy(is_np_ref or is_np_qry)

    best_corr, bc_ptr = _alloc_double(n_qry, use_np)
    best_ref_idx, bri_ptr = _alloc_int64(n_qry, use_np)

    fast = get_fast()
    if fast is not None:
        contaminated = int(
            fast.fast_leak_scan(
                ref_owner,
                qry_owner,
                m_int,
                thresh,
                bool(check_neg),
                int(threads),
                best_corr,
                best_ref_idx,
            )
        )
    else:
        hit_count = ctypes.c_int64(0)
        rc = get_lib().fmp_leak_scan(
            ref_ptr,
            N_ref,
            qry_ptr,
            N_qry,
            m_int,
            thresh,
            1 if check_neg else 0,
            int(threads),
            bc_ptr,
            bri_ptr,
            ctypes.byref(hit_count),
        )
        if rc != 0:
            del ref_owner, qry_owner
            raise RuntimeError(f"fmp_leak_scan failed with error code {rc}")
        contaminated = int(hit_count.value)
    rate = contaminated / n_qry if n_qry > 0 else 0.0

    # Extract top non-overlapping contaminated examples for OLS affine classification.
    examples: list[LeakExample] = []
    max_ex = max(0, int(max_examples))
    if contaminated > 0 and max_ex > 0:
        candidates: list[tuple[float, int, int]] = []
        for q in range(n_qry):
            c = float(best_corr[q])
            r_idx = int(best_ref_idx[q])
            if math.isfinite(c) and c >= thresh and r_idx >= 0:
                candidates.append((c, q, r_idx))
        # Highest score first, lowest qry_idx tie-break for determinism.
        candidates.sort(key=lambda t: (-t[0], t[1]))

        ez = max(1, m_int // 2)
        picked_q: list[int] = []
        for score, q_idx, r_idx in candidates:
            if any(abs(q_idx - prev_q) <= ez for prev_q in picked_q):
                continue
            picked_q.append(q_idx)
            examples.append(_fit_ols_example(ref_owner, qry_owner, r_idx, q_idx, m_int, score))
            if len(examples) >= max_ex:
                break

    del ref_owner, qry_owner
    return LeakReport(
        m=m_int,
        corr_thresh=thresh,
        check_neg=bool(check_neg),
        contaminated_windows=contaminated,
        total_windows=n_qry,
        contamination_rate=rate,
        examples=examples,
        best_corr=best_corr,
        best_ref_idx=best_ref_idx,
    )
