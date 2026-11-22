"""Core batch Matrix Profile, STUMPY drop-in API, mSTUMP, FLOSS, motifs & discords."""

from __future__ import annotations

import array
import math
from typing import Any, NamedTuple

from ._native import (
    NULL_DOUBLE,
    NULL_INT64,
    _INT64_CODE,
    _alloc_double,
    _alloc_double_2d,
    _alloc_int64,
    _alloc_int64_2d,
    _as_c_double_2d,
    _as_c_double_array,
    _as_c_int64_array,
    _default_excl_zone,
    _get_numpy,
    _use_numpy,
    get_lib,
)


class Motif(NamedTuple):
    """Exact recurring subsequence pair discovered from a Matrix Profile."""

    index: int
    neighbor: int
    distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "neighbor": int(self.neighbor),
            "distance": float(self.distance),
        }


class Discord(NamedTuple):
    """Exact anomalous subsequence discovered from a Matrix Profile."""

    index: int
    distance: float
    z_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "distance": float(self.distance),
            "z_score": float(self.z_score),
        }


class _Matrix2D:
    """Zero-dependency 2D (rows, cols) matrix backed by a flat stdlib array.array."""

    __slots__ = ("_buf", "_rows", "_cols", "_is_float")

    def __init__(self, buf: array.array, rows: int, cols: int, is_float: bool) -> None:
        self._buf = buf
        self._rows = int(rows)
        self._cols = int(cols)
        self._is_float = is_float

    @property
    def shape(self) -> tuple[int, int]:
        return (self._rows, self._cols)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def size(self) -> int:
        return self._rows * self._cols

    def __len__(self) -> int:
        return self._rows

    def _norm_idx(self, idx: int, bound: int) -> int:
        if idx < 0:
            idx += bound
        if idx < 0 or idx >= bound:
            raise IndexError(f"index {idx} out of bounds for axis with size {bound}")
        return idx

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            r = self._norm_idx(key, self._rows)
            start = r * self._cols
            return self._buf[start : start + self._cols]
        if isinstance(key, slice):
            row_indices = range(*key.indices(self._rows))
            out = array.array(self._buf.typecode)
            for r in row_indices:
                start = r * self._cols
                out.extend(self._buf[start : start + self._cols])
            return _Matrix2D(out, len(row_indices), self._cols, self._is_float)
        if isinstance(key, tuple) and len(key) == 2:
            rkey, ckey = key
            if isinstance(rkey, int) and isinstance(ckey, int):
                r = self._norm_idx(rkey, self._rows)
                c = self._norm_idx(ckey, self._cols)
                val = self._buf[r * self._cols + c]
                return float(val) if self._is_float else int(val)
            if isinstance(rkey, int) and isinstance(ckey, slice):
                r = self._norm_idx(rkey, self._rows)
                start = r * self._cols
                return self._buf[start : start + self._cols][ckey]
            if isinstance(rkey, slice) and isinstance(ckey, int):
                c = self._norm_idx(ckey, self._cols)
                row_indices = range(*rkey.indices(self._rows))
                return array.array(
                    self._buf.typecode, (self._buf[r * self._cols + c] for r in row_indices)
                )
            if isinstance(rkey, slice) and isinstance(ckey, slice):
                row_indices = range(*rkey.indices(self._rows))
                col_indices = range(*ckey.indices(self._cols))
                out = array.array(self._buf.typecode)
                for r in row_indices:
                    base = r * self._cols
                    for c in col_indices:
                        out.append(self._buf[base + c])
                return _Matrix2D(out, len(row_indices), len(col_indices), self._is_float)
        raise IndexError(f"Unsupported index {key!r} for _Matrix2D")

    def tolist(self) -> list[list[Any]]:
        cast = float if self._is_float else int
        res: list[list[Any]] = []
        for r in range(self._rows):
            start = r * self._cols
            res.append([cast(x) for x in self._buf[start : start + self._cols]])
        return res

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        import numpy as np  # lazy only if user explicitly passes to np.asarray()

        arr = np.frombuffer(self._buf, dtype=np.float64 if self._is_float else np.int64).reshape(
            self._rows, self._cols
        )
        if dtype is not None:
            return arr.astype(dtype, copy=True if copy is None else bool(copy))
        return arr.copy() if copy else arr

    def __repr__(self) -> str:
        kind = "float64" if self._is_float else "int64"
        return f"_Matrix2D(shape=({self._rows}, {self._cols}), dtype={kind})"


class MPArray:
    """Zero-copy (n, 4) STUMPY-compatible Matrix Profile view [P, I, left_I, right_I].

    Supports zero-copy column access (`mp[:, 0]`, `mp[:, 1]`, `mp[:, 2]`, `mp[:, 3]`),
    STUMPY properties (`.P_`, `.I_`, `.left_I_`, `.right_I_`), scalar indexing
    (`mp[i, j]`), slicing, and seamless conversion via `np.asarray(mp)`.
    """

    __slots__ = ("_P", "_I", "_left_I", "_right_I", "_n", "_cols")

    def __init__(self, P: Any, I: Any, left_I: Any, right_I: Any) -> None:
        self._P = P
        self._I = I
        self._left_I = left_I
        self._right_I = right_I
        self._n = len(P)
        self._cols = (P, I, left_I, right_I)

    @property
    def shape(self) -> tuple[int, int]:
        return (self._n, 4)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def size(self) -> int:
        return self._n * 4

    @property
    def dtype(self) -> Any:
        np = _get_numpy()
        return np.dtype("O") if np is not None else object

    @property
    def P_(self) -> Any:
        return self._P

    @property
    def I_(self) -> Any:
        return self._I

    @property
    def left_I_(self) -> Any:
        return self._left_I

    @property
    def right_I_(self) -> Any:
        return self._right_I

    def __len__(self) -> int:
        return self._n

    def _norm_col(self, c: int) -> int:
        if c < 0:
            c += 4
        if c < 0 or c >= 4:
            raise IndexError(f"column index {c} out of bounds for axis 1 with size 4")
        return c

    def _norm_row(self, r: int) -> int:
        if r < 0:
            r += self._n
        if r < 0 or r >= self._n:
            raise IndexError(f"row index {r} out of bounds for axis 0 with size {self._n}")
        return r

    def _is_int(self, x: Any) -> bool:
        if isinstance(x, bool):
            return False
        if isinstance(x, int):
            return True
        np = _get_numpy()
        return np is not None and isinstance(x, np.integer)

    def __getitem__(self, key: Any) -> Any:
        if self._is_int(key):
            r = self._norm_row(int(key))
            row = [float(self._P[r]), int(self._I[r]), int(self._left_I[r]), int(self._right_I[r])]
            np = _get_numpy()
            if np is not None:
                return np.array(row, dtype=object)
            return row

        if isinstance(key, slice):
            return MPArray(
                self._P[key],
                self._I[key],
                self._left_I[key],
                self._right_I[key],
            )

        if isinstance(key, tuple) and len(key) == 2:
            rkey, ckey = key
            if self._is_int(ckey):
                c = self._norm_col(int(ckey))
                col = self._cols[c]
                if isinstance(rkey, slice) and rkey == slice(None):
                    return col
                if self._is_int(rkey):
                    r = self._norm_row(int(rkey))
                    return float(col[r]) if c == 0 else int(col[r])
                return col[rkey]

            if self._is_int(rkey) and isinstance(ckey, slice):
                r = self._norm_row(int(rkey))
                full_row = [
                    float(self._P[r]),
                    int(self._I[r]),
                    int(self._left_I[r]),
                    int(self._right_I[r]),
                ]
                sliced = full_row[ckey]
                np = _get_numpy()
                if np is not None:
                    return np.array(sliced, dtype=object)
                return sliced

        np = _get_numpy()
        if np is not None:
            return self.__array__()[key]

        if isinstance(key, tuple) and len(key) == 2:
            rkey, ckey = key
            rows = range(*rkey.indices(self._n)) if isinstance(rkey, slice) else [self._norm_row(int(rkey))]
            cols = range(*ckey.indices(4)) if isinstance(ckey, slice) else [self._norm_col(int(ckey))]
            return [
                [float(self._cols[c][r]) if c == 0 else int(self._cols[c][r]) for c in cols]
                for r in rows
            ]
        raise IndexError(f"Unsupported index {key!r} for MPArray")

    def __iter__(self):
        P, I, lI, rI = self._P, self._I, self._left_I, self._right_I
        for r in range(self._n):
            yield [float(P[r]), int(I[r]), int(lI[r]), int(rI[r])]

    def tolist(self) -> list[list[Any]]:
        P, I, lI, rI = self._P, self._I, self._left_I, self._right_I
        return [
            [float(P[r]), int(I[r]), int(lI[r]), int(rI[r])]
            for r in range(self._n)
        ]

    def copy(self) -> MPArray:
        np = _get_numpy()
        if np is not None and isinstance(self._P, np.ndarray):
            return MPArray(
                self._P.copy(),
                self._I.copy(),
                self._left_I.copy(),
                self._right_I.copy(),
            )
        return MPArray(
            array.array("d", self._P),
            array.array(_INT64_CODE, self._I),
            array.array(_INT64_CODE, self._left_I),
            array.array(_INT64_CODE, self._right_I),
        )

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        import numpy as np  # lazy only when coerced to NumPy array

        target_dtype = object if dtype is None else dtype
        out = np.empty((self._n, 4), dtype=target_dtype)
        out[:, 0] = self._P
        out[:, 1] = self._I
        out[:, 2] = self._left_I
        out[:, 3] = self._right_I
        return out

    def __repr__(self) -> str:
        if self._n == 0:
            return "MPArray(shape=(0, 4))"
        preview = min(3, self._n)
        rows = [
            f"[{float(self._P[i]):.4f}, {int(self._I[i])}, {int(self._left_I[i])}, {int(self._right_I[i])}]"
            for i in range(preview)
        ]
        ellipsis = ", ..." if self._n > preview else ""
        return f"MPArray(shape=({self._n}, 4), rows=[{', '.join(rows)}{ellipsis}])"


class MatrixProfileResult:
    """Rich result returned by `flashmp.profile()` containing full, left, and right MPs."""

    __slots__ = (
        "mp",
        "idx",
        "left_mp",
        "left_idx",
        "right_mp",
        "right_idx",
        "m",
        "n",
        "excl_zone",
    )

    def __init__(
        self,
        mp: Any,
        idx: Any,
        left_mp: Any,
        left_idx: Any,
        right_mp: Any,
        right_idx: Any,
        m: int,
        n: int,
        excl_zone: int,
    ) -> None:
        self.mp = mp
        self.idx = idx
        self.left_mp = left_mp
        self.left_idx = left_idx
        self.right_mp = right_mp
        self.right_idx = right_idx
        self.m = int(m)
        self.n = int(n)
        self.excl_zone = int(excl_zone)

    def motifs(self, k: int = 3, excl_zone: int | None = None) -> list[Motif]:
        return motifs(self.mp, self.idx, self.m, k=k, excl_zone=excl_zone)

    def discords(self, k: int = 5, excl_zone: int | None = None) -> list[Discord]:
        return discords(self.mp, self.m, k=k, excl_zone=excl_zone)

    def regimes(
        self,
        L: int | None = None,
        max_regimes: int = 5,
        threshold: float = 0.65,
    ) -> list[int]:
        if self.n <= 0:
            return []
        scale = self.m if L is None else int(L)
        if L is None and self.n <= 12 * scale:
            scale = max(4, self.n // 12)
        cac = floss(self.idx, scale)
        return find_regimes(cac, scale, max_regimes=max_regimes, threshold=threshold)

    def to_stump(self) -> MPArray:
        return MPArray(self.mp, self.idx, self.left_idx, self.right_idx)

    def __repr__(self) -> str:
        return f"MatrixProfileResult(n={self.n}, m={self.m}, excl_zone={self.excl_zone})"


def _validate_window(N: int, m: int) -> int:
    m_int = int(m)
    if m_int < 3:
        raise ValueError(f"Subsequence length m must be >= 3, got {m_int}")
    if N < m_int:
        raise ValueError(f"Time series length N={N} must be >= m={m_int}")
    return m_int


def _resolve_excl_zone(m_int: int, excl_zone: int | None) -> int:
    if excl_zone is None:
        return _default_excl_zone(m_int)
    if isinstance(excl_zone, bool):
        raise TypeError("excl_zone must be an integer or None, not bool")
    ez = int(excl_zone)
    if ez < 0:
        raise ValueError(f"excl_zone must be >= 0, got {ez}")
    return ez


def profile(
    T: Any,
    m: int,
    excl_zone: int | None = None,
    threads: int = 0,
) -> MatrixProfileResult:
    """Compute exact Full, Causal Left, and Right Matrix Profiles in a single SIMD pass."""
    ptr, N, owner, is_np = _as_c_double_array(T)
    m_int = _validate_window(N, m)
    ez = _resolve_excl_zone(m_int, excl_zone)
    n = N - m_int + 1
    use_np = _use_numpy(is_np)

    mp, mp_ptr = _alloc_double(n, use_np)
    idx, idx_ptr = _alloc_int64(n, use_np)
    lmp, lmp_ptr = _alloc_double(n, use_np)
    lidx, lidx_ptr = _alloc_int64(n, use_np)
    rmp, rmp_ptr = _alloc_double(n, use_np)
    ridx, ridx_ptr = _alloc_int64(n, use_np)

    rc = get_lib().fmp_self_join(
        ptr,
        N,
        m_int,
        ez,
        int(threads),
        mp_ptr,
        idx_ptr,
        lmp_ptr,
        lidx_ptr,
        rmp_ptr,
        ridx_ptr,
    )
    del owner
    if rc != 0:
        raise RuntimeError(f"fmp_self_join failed with error code {rc}")
    return MatrixProfileResult(
        mp=mp,
        idx=idx,
        left_mp=lmp,
        left_idx=lidx,
        right_mp=rmp,
        right_idx=ridx,
        m=m_int,
        n=n,
        excl_zone=ez,
    )


def stump(
    T_A: Any,
    m: int,
    T_B: Any = None,
    ignore_trivial: bool = True,
    excl_zone: int | None = None,
    threads: int = 0,
) -> MPArray:
    """Drop-in STUMPY-compatible Matrix Profile (`[P, I, left_I, right_I]` shape `(n, 4)`)."""
    ptr_a, NA, owner_a, is_np_a = _as_c_double_array(T_A)
    m_int = _validate_window(NA, m)

    if T_B is None and ignore_trivial:
        ez = _resolve_excl_zone(m_int, excl_zone)
        n = NA - m_int + 1
        use_np = _use_numpy(is_np_a)
        mp, mp_ptr = _alloc_double(n, use_np)
        idx, idx_ptr = _alloc_int64(n, use_np)
        lidx, lidx_ptr = _alloc_int64(n, use_np)
        ridx, ridx_ptr = _alloc_int64(n, use_np)

        rc = get_lib().fmp_self_join(
            ptr_a,
            NA,
            m_int,
            ez,
            int(threads),
            mp_ptr,
            idx_ptr,
            NULL_DOUBLE,
            lidx_ptr,
            NULL_DOUBLE,
            ridx_ptr,
        )
        del owner_a
        if rc != 0:
            raise RuntimeError(f"fmp_self_join failed with error code {rc}")
        return MPArray(mp, idx, lidx, ridx)

    # AB-join (or self-join with ignore_trivial=False)
    if T_B is None:
        ptr_b, NB, owner_b, is_np_b = ptr_a, NA, owner_a, is_np_a
    else:
        ptr_b, NB, owner_b, is_np_b = _as_c_double_array(T_B)
        _validate_window(NB, m_int)

    nA = NA - m_int + 1
    use_np = _use_numpy(is_np_a or is_np_b)
    mp_a, mp_a_ptr = _alloc_double(nA, use_np)
    idx_a, idx_a_ptr = _alloc_int64(nA, use_np)

    rc = get_lib().fmp_ab_join(
        ptr_a,
        NA,
        ptr_b,
        NB,
        m_int,
        int(threads),
        mp_a_ptr,
        idx_a_ptr,
        NULL_DOUBLE,
        NULL_INT64,
    )
    del owner_a, owner_b
    if rc != 0:
        raise RuntimeError(f"fmp_ab_join failed with error code {rc}")

    minus_left, _ = _alloc_int64(nA, use_np, fill=-1)
    minus_right, _ = _alloc_int64(nA, use_np, fill=-1)
    return MPArray(mp_a, idx_a, minus_left, minus_right)


def abjoin(
    T_A: Any,
    T_B: Any,
    m: int,
    threads: int = 0,
) -> tuple[Any, Any, Any, Any]:
    """Exact bidirectional AB-Join returning `(mp_a, idx_a, mp_b, idx_b)` in a single pass."""
    ptr_a, NA, owner_a, is_np_a = _as_c_double_array(T_A)
    ptr_b, NB, owner_b, is_np_b = _as_c_double_array(T_B)
    m_int = _validate_window(NA, m)
    _validate_window(NB, m_int)

    nA = NA - m_int + 1
    nB = NB - m_int + 1
    use_np = _use_numpy(is_np_a or is_np_b)

    mp_a, mp_a_ptr = _alloc_double(nA, use_np)
    idx_a, idx_a_ptr = _alloc_int64(nA, use_np)
    mp_b, mp_b_ptr = _alloc_double(nB, use_np)
    idx_b, idx_b_ptr = _alloc_int64(nB, use_np)

    rc = get_lib().fmp_ab_join(
        ptr_a,
        NA,
        ptr_b,
        NB,
        m_int,
        int(threads),
        mp_a_ptr,
        idx_a_ptr,
        mp_b_ptr,
        idx_b_ptr,
    )
    del owner_a, owner_b
    if rc != 0:
        raise RuntimeError(f"fmp_ab_join failed with error code {rc}")
    return mp_a, idx_a, mp_b, idx_b


def mstump(
    T: Any,
    m: int,
    excl_zone: int | None = None,
    threads: int = 0,
) -> tuple[Any, Any]:
    """Exact k-dimensional Multivariate Matrix Profile matching `stumpy.mstump` shape `(D, n)`."""
    ptr, D, N, owner, is_np = _as_c_double_2d(T)
    if D < 1:
        raise ValueError(f"Number of dimensions D must be >= 1, got {D}")
    m_int = _validate_window(N, m)
    ez = _resolve_excl_zone(m_int, excl_zone)
    n = N - m_int + 1
    use_np = _use_numpy(is_np)

    mp_buf, mp_ptr = _alloc_double_2d(D, n, use_np)
    idx_buf, idx_ptr = _alloc_int64_2d(D, n, use_np)

    rc = get_lib().fmp_mstump(
        ptr,
        D,
        N,
        m_int,
        ez,
        int(threads),
        mp_ptr,
        idx_ptr,
    )
    del owner
    if rc != 0:
        raise RuntimeError(f"fmp_mstump failed with error code {rc}")

    if use_np:
        return mp_buf, idx_buf
    return _Matrix2D(mp_buf, D, n, True), _Matrix2D(idx_buf, D, n, False)


def floss(idx: Any, L: int) -> Any:
    """Compute exact O(n) FLOSS/FLUSS Corrected Arc Curve (CAC) in [0, 1]."""
    ptr, n, owner, is_np = _as_c_int64_array(idx)
    if n <= 0:
        raise ValueError("idx must be non-empty")
    L_int = max(1, int(L))
    use_np = _use_numpy(is_np)

    cac, cac_ptr = _alloc_double(n, use_np)
    rc = get_lib().fmp_floss_cac(ptr, n, L_int, cac_ptr)
    del owner
    if rc != 0:
        raise RuntimeError(f"fmp_floss_cac failed with error code {rc}")
    return cac


def find_regimes(
    cac: Any,
    L: int,
    max_regimes: int = 8,
    threshold: float = 0.65,
) -> list[int]:
    """Extract sorted regime changepoint indices from a FLOSS Corrected Arc Curve."""
    n = len(cac)
    max_splits = max(0, int(max_regimes) - 1)
    if n == 0 or max_splits == 0:
        return []

    L_int = max(1, int(L))
    ez = max(1, min(5 * L_int, max(L_int, n // (max_regimes + 1))))
    thresh = float(threshold)

    np = _get_numpy()
    if np is not None and isinstance(cac, np.ndarray):
        work = np.where(np.isfinite(cac), cac.astype(np.float64, copy=True), np.inf)
        found: list[int] = []
        for _ in range(max_splits):
            pos = int(np.argmin(work))
            val = float(work[pos])
            if not math.isfinite(val) or val >= thresh:
                break
            found.append(pos)
            lo = max(0, pos - ez)
            hi = min(n, pos + ez + 1)
            work[lo:hi] = np.inf
        found.sort()
        return found

    work_py = [float(v) if math.isfinite(float(v)) else math.inf for v in cac]
    found_py: list[int] = []
    for _ in range(max_splits):
        best_val = thresh
        best_pos = -1
        for i, v in enumerate(work_py):
            if v < best_val:
                best_val = v
                best_pos = i
        if best_pos < 0:
            break
        found_py.append(best_pos)
        lo = max(0, best_pos - ez)
        hi = min(n, best_pos + ez + 1)
        for j in range(lo, hi):
            work_py[j] = math.inf
    found_py.sort()
    return found_py


def auto_window(T: Any, min_m: int = 8, max_m: int = 512) -> int:
    """Fast causal autocorrelation period selector returning recommended window length `m`."""
    ptr, N, owner, _ = _as_c_double_array(T)
    if N <= 0:
        raise ValueError("Time series T must be non-empty")
    res = int(get_lib().fmp_auto_window(ptr, N, int(min_m), int(max_m)))
    del owner
    return res


def motifs(
    mp: Any,
    idx: Any,
    m: int,
    k: int = 3,
    excl_zone: int | None = None,
) -> list[Motif]:
    """Greedy non-overlapping top-k motif extraction with symmetric exclusion-zone masking."""
    n = len(mp)
    if len(idx) != n:
        raise ValueError(f"mp length ({n}) != idx length ({len(idx)})")
    k_int = int(k)
    if n == 0 or k_int <= 0:
        return []

    ez = max(1, int(m) // 2) if excl_zone is None else max(0, int(excl_zone))
    np = _get_numpy()

    if np is not None and isinstance(mp, np.ndarray):
        idx_arr = idx if isinstance(idx, np.ndarray) else np.asarray(idx, dtype=np.int64)
        valid = np.isfinite(mp) & (idx_arr >= 0) & (idx_arr < n) & (idx_arr != np.arange(n, dtype=np.int64))
        work = np.where(valid, mp.astype(np.float64, copy=True), np.inf)
        out: list[Motif] = []
        for _ in range(k_int):
            pos = int(np.argmin(work))
            dist = float(work[pos])
            if not math.isfinite(dist):
                break
            nb = int(idx_arr[pos])
            a, b = (pos, nb) if pos < nb else (nb, pos)
            out.append(Motif(index=a, neighbor=b, distance=dist))
            work[max(0, a - ez) : min(n, a + ez + 1)] = np.inf
            work[max(0, b - ez) : min(n, b + ez + 1)] = np.inf
        return out

    work_py = [
        float(mp[i]) if (math.isfinite(float(mp[i])) and 0 <= int(idx[i]) < n and int(idx[i]) != i) else math.inf
        for i in range(n)
    ]
    out_py: list[Motif] = []
    for _ in range(k_int):
        best_val = math.inf
        best_pos = -1
        for i, v in enumerate(work_py):
            if v < best_val:
                best_val = v
                best_pos = i
        if best_pos < 0:
            break
        nb = int(idx[best_pos])
        a, b = (best_pos, nb) if best_pos < nb else (nb, best_pos)
        out_py.append(Motif(index=a, neighbor=b, distance=best_val))
        for center in (a, b):
            lo = max(0, center - ez)
            hi = min(n, center + ez + 1)
            for j in range(lo, hi):
                work_py[j] = math.inf
    return out_py


def _robust_loc_scale(values: list[float]) -> tuple[float, float]:
    """Return (median, robust_sigma) over finite values, falling back to std if MAD == 0."""
    cnt = len(values)
    if cnt == 0:
        return 0.0, 0.0
    sv = sorted(values)
    mid = cnt // 2
    med = sv[mid] if (cnt & 1) else 0.5 * (sv[mid - 1] + sv[mid])
    abs_devs = sorted(abs(v - med) for v in sv)
    mad = abs_devs[mid] if (cnt & 1) else 0.5 * (abs_devs[mid - 1] + abs_devs[mid])
    sigma = 1.4826 * mad
    if sigma > 1e-12:
        return med, sigma
    mean = sum(sv) / cnt
    var = sum((v - mean) * (v - mean) for v in sv) / cnt
    return med, math.sqrt(var)


def discords(
    mp: Any,
    m: int,
    k: int = 5,
    excl_zone: int | None = None,
) -> list[Discord]:
    """Greedy non-overlapping top-k discord extraction with robust z-scores."""
    n = len(mp)
    k_int = int(k)
    if n == 0 or k_int <= 0:
        return []

    ez = max(1, int(m) // 2) if excl_zone is None else max(0, int(excl_zone))
    np = _get_numpy()

    if np is not None and isinstance(mp, np.ndarray):
        finite_mask = np.isfinite(mp)
        if not np.any(finite_mask):
            return []
        finite_vals = mp[finite_mask]
        med = float(np.median(finite_vals))
        mad = float(np.median(np.abs(finite_vals - med)))
        sigma = 1.4826 * mad
        if sigma <= 1e-12:
            sigma = float(np.std(finite_vals))

        work = np.where(finite_mask, mp.astype(np.float64, copy=True), -np.inf)
        out: list[Discord] = []
        for _ in range(k_int):
            pos = int(np.argmax(work))
            dist = float(work[pos])
            if not math.isfinite(dist):
                break
            z = (dist - med) / sigma if sigma > 1e-12 else 0.0
            out.append(Discord(index=pos, distance=dist, z_score=float(z)))
            work[max(0, pos - ez) : min(n, pos + ez + 1)] = -np.inf
        return out

    finite_py = [float(v) for v in mp if math.isfinite(float(v))]
    if not finite_py:
        return []
    med, sigma = _robust_loc_scale(finite_py)
    work_py = [float(v) if math.isfinite(float(v)) else -math.inf for v in mp]
    out_py: list[Discord] = []
    for _ in range(k_int):
        best_val = -math.inf
        best_pos = -1
        for i, v in enumerate(work_py):
            if v > best_val:
                best_val = v
                best_pos = i
        if best_pos < 0:
            break
        z = (best_val - med) / sigma if sigma > 1e-12 else 0.0
        out_py.append(Discord(index=best_pos, distance=best_val, z_score=z))
        lo = max(0, best_pos - ez)
        hi = min(n, best_pos + ez + 1)
        for j in range(lo, hi):
            work_py[j] = -math.inf
    return out_py
