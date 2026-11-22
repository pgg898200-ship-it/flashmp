"""Comprehensive tests for zero-copy C-API extension (_fast) and AI/DataFrame integrations.

Runs 100% under BOTH:
  - pytest (e.g. `.bench_venv/bin/pytest tests/test_integrations_and_fast.py`)
  - bare stdlib `python3 tests/test_integrations_and_fast.py` (zero third-party packages)
"""

from __future__ import annotations

import array
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import tracemalloc
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import numpy as np
except ImportError:  # pragma: no cover - bare stdlib mode
    np = None  # type: ignore[assignment]

import flashmp
import flashmp._fast as _fast
from flashmp.integrations import (
    AgentToolkit,
    anthropic_tools,
    extract_series,
    install_skill,
    mcp_client_config,
    openai_tools,
)
from flashmp.stream import StreamMP


_INT64_CODE = "q" if array.array("q").itemsize == 8 else "l"
_INT32_CODE = "i" if array.array("i").itemsize == 4 else "l"


def _make_wave(n: int, period: float = 24.0, phase: float = 0.0) -> array.array:
    return array.array(
        "d",
        (
            math.sin(2.0 * math.pi * i / period + phase)
            + 0.15 * math.cos(2.0 * math.pi * i / (period * 0.37) + 0.5)
            for i in range(n)
        ),
    )


def _assert_close_seq(tc: unittest.TestCase, a: Any, b: Any, tol: float = 1e-12) -> None:
    tc.assertEqual(len(a), len(b))
    for i in range(len(a)):
        va = float(a[i])
        vb = float(b[i])
        if math.isnan(va) and math.isnan(vb):
            continue
        if math.isinf(va) and math.isinf(vb) and (va > 0) == (vb > 0):
            continue
        tc.assertLessEqual(abs(va - vb), tol, f"Mismatch at index {i}: {va} vs {vb}")


# ---------------------------------------------------------------------------
# Duck-Typed Pandas / Polars Mocks (zero dependency on pandas/polars)
# ---------------------------------------------------------------------------


class _FakeDtype:
    def __init__(self, kind: str | None = None, name: str = "", polars_numeric: bool | None = None) -> None:
        if kind is not None:
            self.kind = kind
        self._name = name
        self._polars_numeric = polars_numeric

    def is_numeric(self) -> bool:
        if self._polars_numeric is None:
            raise AttributeError("not a polars dtype")
        return self._polars_numeric

    def __str__(self) -> str:
        return self._name


class _MockPandasSeries:
    def __init__(self, data: list[Any], name: str | None = None, dtype: Any = None, with_numpy: bool = False) -> None:
        self._data = list(data)
        self.name = name
        self.dtype = dtype
        self._with_numpy = with_numpy

    @property
    def values(self) -> Any:
        if self._with_numpy and np is not None:
            return np.asarray(self._data)
        return list(self._data)

    def to_numpy(self) -> Any:
        if self._with_numpy and np is not None:
            return np.asarray(self._data)
        return array.array("d", self._data)

    def tolist(self) -> list[Any]:
        return list(self._data)

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class _MockPandasDataFrame:
    def __init__(self, cols: dict[str, _MockPandasSeries]) -> None:
        self._cols = dict(cols)
        self.columns = list(cols.keys())

    def __getitem__(self, key: Any) -> _MockPandasSeries:
        return self._cols[key]


class _MockPolarsSeries:
    def __init__(self, name: str, data: list[Any], dtype_name: str = "Float64", is_num: bool = True) -> None:
        self.name = name
        self._data = list(data)
        self.dtype = _FakeDtype(kind=None, name=dtype_name, polars_numeric=is_num)

    def to_numpy(self) -> Any:
        if np is None:
            raise ModuleNotFoundError("No module named 'numpy'")
        return np.asarray(self._data)

    def to_list(self) -> list[Any]:
        return list(self._data)

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class _MockPolarsDataFrame:
    def __init__(self, series_list: list[_MockPolarsSeries]) -> None:
        self._map = {s.name: s for s in series_list}
        self.columns = [s.name for s in series_list]

    def get_column(self, name: str) -> _MockPolarsSeries:
        return self._map[name]

    def __getitem__(self, name: str) -> _MockPolarsSeries:
        return self._map[name]


# ---------------------------------------------------------------------------
# 1. Zero-Copy CPython C-API Extension Tests (_fast)
# ---------------------------------------------------------------------------


class TestFastCExtension(unittest.TestCase):
    def test_metadata_matches_native(self) -> None:
        self.assertEqual(_fast.version(), flashmp.version())
        self.assertEqual(_fast.simd_isa(), flashmp.simd_isa())

    def test_fast_self_join_stdlib_array_matches_profile_and_stump(self) -> None:
        x = _make_wave(220, period=19.0)
        x[100:116] = array.array("d", (0.9 * ((-1.0) ** k) for k in range(16)))
        m = 16
        n = len(x) - m + 1

        mp = array.array("d", [0.0]) * n
        idx = array.array(_INT64_CODE, [0]) * n
        lmp = array.array("d", [0.0]) * n
        lidx = array.array(_INT64_CODE, [0]) * n
        rmp = array.array("d", [0.0]) * n
        ridx = array.array(_INT64_CODE, [0]) * n

        _fast.fast_self_join(x, m, -1, 1, mp, idx, lmp, lidx, rmp, ridx)

        ref = flashmp.profile(x, m, threads=1)
        _assert_close_seq(self, mp, ref.mp, tol=0.0)
        self.assertEqual(list(idx), list(ref.idx))
        _assert_close_seq(self, lmp, ref.left_mp, tol=0.0)
        self.assertEqual(list(lidx), list(ref.left_idx))
        _assert_close_seq(self, rmp, ref.right_mp, tol=0.0)
        self.assertEqual(list(ridx), list(ref.right_idx))

        # Optional outputs omitted / None (stump layout: mp, idx, None, lidx, None, ridx)
        mp2 = array.array("d", [0.0]) * n
        idx2 = array.array(_INT64_CODE, [0]) * n
        lidx2 = array.array(_INT64_CODE, [0]) * n
        ridx2 = array.array(_INT64_CODE, [0]) * n
        _fast.fast_self_join(x, m, None, 0, mp2, idx2, None, lidx2, None, ridx2)
        _assert_close_seq(self, mp2, ref.mp, tol=1e-14)
        self.assertEqual(list(idx2), list(ref.idx))
        self.assertEqual(list(lidx2), list(ref.left_idx))
        self.assertEqual(list(ridx2), list(ref.right_idx))

    def test_fast_ab_join_matches_abjoin(self) -> None:
        a = _make_wave(140, period=17.0, phase=0.1)
        b = _make_wave(180, period=17.0, phase=0.4)
        m = 14
        nA = len(a) - m + 1
        nB = len(b) - m + 1

        mp_a = array.array("d", [0.0]) * nA
        idx_a = array.array(_INT64_CODE, [0]) * nA
        mp_b = array.array("d", [0.0]) * nB
        idx_b = array.array(_INT64_CODE, [0]) * nB

        _fast.fast_ab_join(a, b, m, 1, mp_a, idx_a, mp_b, idx_b)
        ref_ma, ref_ia, ref_mb, ref_ib = flashmp.abjoin(a, b, m, threads=1)
        _assert_close_seq(self, mp_a, ref_ma, tol=0.0)
        self.assertEqual(list(idx_a), list(ref_ia))
        _assert_close_seq(self, mp_b, ref_mb, tol=0.0)
        self.assertEqual(list(idx_b), list(ref_ib))

        # Unidirectional (out_mp_b=None, out_idx_b=None) + aliased a is b
        mp_self = array.array("d", [0.0]) * nA
        idx_self = array.array(_INT64_CODE, [0]) * nA
        _fast.fast_ab_join(a, a, m, 1, mp_self, idx_self)
        self.assertLess(max(mp_self), 1e-12)

        # Providing only one of out_mp_b / out_idx_b must raise ValueError
        with self.assertRaises(ValueError):
            _fast.fast_ab_join(a, b, m, 1, mp_a, idx_a, mp_b, None)

    def test_fast_leak_scan_detects_verbatim_and_neg_affine(self) -> None:
        ref = _make_wave(240, period=21.0)
        qry = array.array("d", (math.sin(i * 0.43) + 0.3 * math.sin(i * 1.17) for i in range(200)))
        m = 20
        # Plant negative affine leak: qry[60:80] = -2.5 * ref[90:110] + 13.0
        for k in range(m):
            qry[60 + k] = -2.5 * ref[90 + k] + 13.0

        n_qry = len(qry) - m + 1
        out_corr = array.array("d", [0.0]) * n_qry
        out_idx = array.array(_INT64_CODE, [0]) * n_qry

        hits = _fast.fast_leak_scan(ref, qry, m, 0.999, True, 1, out_corr, out_idx)
        rep = flashmp.scan_leakage(ref, qry, m=m, corr_thresh=0.999, check_neg=True, threads=1)
        self.assertEqual(hits, rep.contaminated_windows)
        self.assertGreaterEqual(hits, 1)
        self.assertGreaterEqual(out_corr[60], 0.999999)
        self.assertEqual(out_idx[60], 90)

        # With check_neg=False, the negated segment should not match index 90 at +0.999
        hits_pos = _fast.fast_leak_scan(ref, qry, m, 0.999, False, 1, out_corr, out_idx)
        rep_pos = flashmp.scan_leakage(ref, qry, m=m, corr_thresh=0.999, check_neg=False, threads=1)
        self.assertEqual(hits_pos, rep_pos.contaminated_windows)

    def test_fast_mstump_flat_array_and_bonus_floss_auto_window(self) -> None:
        D, N, m = 3, 120, 12
        n = N - m + 1
        rows = [_make_wave(N, period=16.0 + 3.0 * d, phase=0.2 * d) for d in range(D)]
        flat = array.array("d")
        for r in rows:
            flat.extend(r)

        out_mp = array.array("d", [0.0]) * (D * n)
        out_idx = array.array(_INT64_CODE, [0]) * (D * n)
        _fast.fast_mstump(flat, D, N, m, -1, 1, out_mp, out_idx)

        ref_mp, ref_idx = flashmp.mstump(rows, m, threads=1)
        for d in range(D):
            _assert_close_seq(self, out_mp[d * n : (d + 1) * n], ref_mp[d], tol=0.0)
            self.assertEqual(list(out_idx[d * n : (d + 1) * n]), list(ref_idx[d]))

        # Bonus fast_floss_cac and fast_auto_window
        idx_1d = array.array(_INT64_CODE, ref_idx[0])
        cac_fast = array.array("d", [0.0]) * n
        _fast.fast_floss_cac(idx_1d, m, cac_fast)
        cac_ref = flashmp.floss(idx_1d, m)
        _assert_close_seq(self, cac_fast, cac_ref, tol=0.0)

        w_fast = _fast.fast_auto_window(rows[0], 8, 64)
        w_ref = flashmp.auto_window(rows[0], min_m=8, max_m=64)
        self.assertEqual(w_fast, w_ref)

    def test_buffer_validation_stdlib_and_unaligned_memoryview(self) -> None:
        m = 8
        N = 40
        n = N - m + 1
        ok_x = _make_wave(N)
        ok_mp = array.array("d", [0.0]) * n
        ok_idx = array.array(_INT64_CODE, [0]) * n

        # Unaligned memoryview (offset by 1 byte!) MUST be rejected cleanly
        raw = bytearray(N * 8 + 1)
        unaligned_mv = memoryview(raw)[1:].cast("d")
        with self.assertRaises(ValueError):
            _fast.fast_self_join(unaligned_mv, m, -1, 1, ok_mp, ok_idx)

        # Wrong typecode (float32 / int32)
        f32 = array.array("f", [1.0] * N)
        with self.assertRaises(TypeError):
            _fast.fast_self_join(f32, m, -1, 1, ok_mp, ok_idx)
        i32_out = array.array(_INT32_CODE, [0]) * n
        with self.assertRaises(TypeError):
            _fast.fast_self_join(ok_x, m, -1, 1, ok_mp, i32_out)

        # Non-buffer / string / bytes / bool / list inputs
        for bad in ("12345678", b"12345678" * 10, [1.0] * N, True, None):
            with self.assertRaises(TypeError):
                _fast.fast_self_join(bad, m, -1, 1, ok_mp, ok_idx)

        # Read-only buffer passed as writable output
        ro_mp = memoryview(ok_mp).toreadonly()
        with self.assertRaises((TypeError, ValueError, BufferError)):
            _fast.fast_self_join(ok_x, m, -1, 1, ro_mp, ok_idx)

        # Wrong output length
        short_mp = array.array("d", [0.0]) * (n - 1)
        with self.assertRaises(ValueError):
            _fast.fast_self_join(ok_x, m, -1, 1, short_mp, ok_idx)

        # Bool for integer params rejected
        with self.assertRaises(TypeError):
            _fast.fast_self_join(ok_x, True, -1, 1, ok_mp, ok_idx)
        with self.assertRaises(TypeError):
            _fast.fast_self_join(ok_x, m, True, 1, ok_mp, ok_idx)

    @unittest.skipIf(np is None, "requires numpy")
    def test_buffer_validation_numpy_unaligned_and_dtypes(self) -> None:
        m = 12
        N = 80
        n = N - m + 1
        x = np.sin(np.linspace(0, 12.0, N)).astype(np.float64)
        out_mp = np.empty(n, dtype=np.float64)
        out_idx = np.empty(n, dtype=np.int64)

        # Valid aligned float64 NumPy zero-copy call
        _fast.fast_self_join(x, m, -1, 1, out_mp, out_idx)
        ref = flashmp.profile(x, m, threads=1)
        self.assertTrue(np.array_equal(out_mp, ref.mp))
        self.assertTrue(np.array_equal(out_idx, ref.idx))

        # Unaligned NumPy view rejected with ValueError
        raw = np.zeros(N * 8 + 3, dtype=np.uint8)
        unaligned = np.ndarray((N,), dtype=np.float64, buffer=raw[1 : 1 + N * 8])
        unaligned[:] = x
        self.assertFalse(unaligned.flags.aligned)
        with self.assertRaises(ValueError):
            _fast.fast_self_join(unaligned, m, -1, 1, out_mp, out_idx)

        # Non-contiguous stride / reversed rejected
        with self.assertRaises((ValueError, BufferError)):
            _fast.fast_self_join(x[::2], m, -1, 1, np.empty(len(x[::2]) - m + 1, dtype=np.float64), np.empty(len(x[::2]) - m + 1, dtype=np.int64))
        with self.assertRaises((ValueError, BufferError)):
            _fast.fast_self_join(x[::-1], m, -1, 1, out_mp, out_idx)

        # Complex, bool, float32, string, object, uint64, wrong-endian rejected
        for bad_arr in (
            x.astype(np.complex128),
            np.ones(N, dtype=np.bool_),
            x.astype(np.float32),
            np.array(["1.0"] * N),
            np.array([1.0] * N, dtype=object),
            x.astype(">f8"),
        ):
            with self.assertRaises((TypeError, ValueError)):
                _fast.fast_self_join(bad_arr, m, -1, 1, out_mp, out_idx)

        # Unsigned int64 rejected for out_idx
        with self.assertRaises(TypeError):
            _fast.fast_self_join(x, m, -1, 1, out_mp, np.empty(n, dtype=np.uint64))

        # Read-only output rejected
        ro = np.empty(n, dtype=np.float64)
        ro.setflags(write=False)
        with self.assertRaises((ValueError, BufferError, TypeError)):
            _fast.fast_self_join(x, m, -1, 1, ro, out_idx)

        # 2D NumPy array for fast_mstump + Fortran order rejected
        M = np.vstack([x, np.cos(np.linspace(0, 12.0, N))])
        mp_2d = np.empty((2, n), dtype=np.float64)
        idx_2d = np.empty((2, n), dtype=np.int64)
        _fast.fast_mstump(M, 2, N, m, -1, 1, mp_2d, idx_2d)
        ref_p, ref_i = flashmp.mstump(M, m, threads=1)
        self.assertTrue(np.array_equal(mp_2d, ref_p))
        self.assertTrue(np.array_equal(idx_2d, ref_i))

        M_f = np.asfortranarray(M)
        with self.assertRaises((ValueError, BufferError)):
            _fast.fast_mstump(M_f, 2, N, m, -1, 1, mp_2d, idx_2d)

    def test_fast_stream_exact_match_vs_stream_mp_and_lifecycle(self) -> None:
        m = 16
        W = 64
        x = _make_wave(260, period=18.0)
        x[170:186] = array.array("d", (0.85 * ((-1.0) ** k) for k in range(16)))

        with (
            _fast.FastStream(m=m, window=W, ez=-1, egress=True, alpha=0.05, track_full_mp=True) as fs,
            StreamMP(m=m, window=W, egress=True, alpha=0.05, track_full_mp=True) as sm,
        ):
            self.assertEqual(fs.m, m)
            self.assertEqual(fs.window, W)
            self.assertEqual(fs.excl_zone, sm.excl_zone)
            self.assertTrue(fs.egress)
            self.assertTrue(fs.track_full_mp)
            self.assertFalse(fs.closed)

            fs.init(x[:100])
            sm.init(x[:100])
            self.assertEqual(fs.active_len(), sm.active_len)
            self.assertEqual(fs.gid_min(), sm.gid_min)

            # Single-tick updates through egress & anomaly
            for i in range(100, 200):
                t_fast = fs.update(float(x[i]))
                t_ref = sm.update(float(x[i]))
                self.assertEqual(len(t_fast), 8)
                self.assertEqual(t_fast[0], t_ref.gid)
                self.assertEqual(t_fast[1], t_ref.left_dist)
                self.assertEqual(t_fast[2], t_ref.left_idx)
                self.assertEqual(t_fast[3], t_ref.discord_z)
                self.assertEqual(t_fast[4], t_ref.p_value)
                self.assertEqual(t_fast[5], t_ref.e_value)
                self.assertIs(type(t_fast[6]), bool)
                self.assertEqual(t_fast[6], t_ref.is_alert)
                self.assertEqual(t_fast[7], t_ref.n_repaired)

            # Batch updates match bit-for-bit
            tail = x[200:]
            cnt = len(tail)
            ld = array.array("d", [0.0]) * cnt
            li = array.array(_INT64_CODE, [0]) * cnt
            ev = array.array("d", [0.0]) * cnt
            al = array.array(_INT32_CODE, [0]) * cnt
            fs.update_batch(tail, ld, li, ev, al)
            ref_batch = sm.update_batch(tail)
            _assert_close_seq(self, ld, ref_batch.left_dist, tol=0.0)
            self.assertEqual(list(li), list(ref_batch.left_idx))
            _assert_close_seq(self, ev, ref_batch.e_value, tol=0.0)
            self.assertEqual(list(al), list(ref_batch.is_alert))

            # Snapshots (relative and global) match bit-for-bit
            active = fs.active_len()
            self.assertEqual(active, sm.active_len)
            for rel in (True, False):
                s_mp = array.array("d", [0.0]) * active
                s_idx = array.array(_INT64_CODE, [0]) * active
                s_lmp = array.array("d", [0.0]) * active
                s_lidx = array.array(_INT64_CODE, [0]) * active
                written = fs.snapshot(rel, s_mp, s_idx, s_lmp, s_lidx)
                self.assertEqual(written, active)
                ref_snap = sm.snapshot(relative_indices=rel)
                _assert_close_seq(self, s_mp, ref_snap.mp, tol=0.0)
                self.assertEqual(list(s_idx), list(ref_snap.idx))
                _assert_close_seq(self, s_lmp, ref_snap.left_mp, tol=0.0)
                self.assertEqual(list(s_lidx), list(ref_snap.left_idx))

            # Edge inputs: int -1, nan, bool rejected
            t_neg1 = fs.update(-1)
            self.assertEqual(len(t_neg1), 8)
            t_nan = fs.update(float("nan"))
            self.assertEqual(len(t_nan), 8)
            with self.assertRaises(TypeError):
                fs.update(True)

        # Closed stream safety & idempotent close
        self.assertTrue(fs.closed)
        fs.close()
        with self.assertRaises(RuntimeError):
            fs.update(1.0)
        with self.assertRaises(RuntimeError):
            fs.active_len()

        # Uninitialized __new__ instance fails cleanly without crashing
        uninit = _fast.FastStream.__new__(_fast.FastStream)
        with self.assertRaises(RuntimeError):
            uninit.update(1.0)
        uninit.close()

    def test_fast_stream_per_tick_speedup_over_ctypes(self) -> None:
        m = 32
        W = 256
        x = _make_wave(10_500, period=28.0)
        warmup = x[:500]
        ticks = [float(v) for v in x[500:]]

        # Force the comparator StreamMP onto the ctypes fallback path (since StreamMP now uses _fast by default).
        prev_env = os.environ.get("FLASHMP_LIB")
        os.environ["FLASHMP_LIB"] = "__force_ctypes_fallback__"
        try:
            sm = StreamMP(m=m, window=W, egress=True, alpha=0.05, track_full_mp=False)
        finally:
            if prev_env is None:
                os.environ.pop("FLASHMP_LIB", None)
            else:
                os.environ["FLASHMP_LIB"] = prev_env

        with (
            _fast.FastStream(m=m, window=W, ez=-1, egress=True, alpha=0.05, track_full_mp=False) as fs,
            sm,
        ):
            self.assertIsNone(sm._fast)
            fs.init(warmup)
            sm.init(warmup)

            fs_upd = fs.update
            t0 = time.perf_counter()
            for v in ticks:
                fs_upd(v)
            t1 = time.perf_counter()

            sm_upd = sm.update
            t2 = time.perf_counter()
            for v in ticks:
                sm_upd(v)
            t3 = time.perf_counter()

            fast_us = (t1 - t0) * 1e6 / len(ticks)
            ctypes_us = (t3 - t2) * 1e6 / len(ticks)
            speedup = ctypes_us / fast_us
            # FastStream eliminates ~4.5 us/tick of ctypes overhead (typically 3x-6x faster)
            self.assertGreater(
                speedup,
                1.8,
                f"Expected FastStream to be >1.8x faster than ctypes ({fast_us:.3f} vs {ctypes_us:.3f} us/tick)",
            )

    def test_gil_release_multithreaded_safety(self) -> None:
        # 4 threads concurrently running fast_self_join + FastStream init/update_batch
        N = 1200
        m = 24
        n = N - m + 1
        x = _make_wave(N, period=25.0)

        ref_mp = array.array("d", [0.0]) * n
        ref_idx = array.array(_INT64_CODE, [0]) * n
        _fast.fast_self_join(x, m, -1, 1, ref_mp, ref_idx)

        ref_ld = array.array("d", [0.0]) * 600
        ref_li = array.array(_INT64_CODE, [0]) * 600
        ref_ev = array.array("d", [0.0]) * 600
        ref_al = array.array(_INT32_CODE, [0]) * 600
        with _fast.FastStream(m=m, window=256, ez=-1, egress=True, alpha=0.05, track_full_mp=False) as fs_ref:
            fs_ref.init(x[:600])
            fs_ref.update_batch(x[600:], ref_ld, ref_li, ref_ev, ref_al)

        errors: list[BaseException] = []
        barrier = threading.Barrier(4)

        def worker() -> None:
            try:
                barrier.wait(timeout=5.0)
                for _ in range(3):
                    mp = array.array("d", [0.0]) * n
                    idx = array.array(_INT64_CODE, [0]) * n
                    _fast.fast_self_join(x, m, -1, 1, mp, idx)
                    if list(idx) != list(ref_idx):
                        raise AssertionError("Thread idx mismatch in fast_self_join")

                    ld = array.array("d", [0.0]) * 600
                    li = array.array(_INT64_CODE, [0]) * 600
                    ev = array.array("d", [0.0]) * 600
                    al = array.array(_INT32_CODE, [0]) * 600
                    with _fast.FastStream(m=m, window=256, ez=-1, egress=True, alpha=0.05, track_full_mp=False) as fs:
                        fs.init(x[:600])
                        fs.update_batch(x[600:], ld, li, ev, al)
                    if list(li) != list(ref_li) or list(al) != list(ref_al):
                        raise AssertionError("Thread mismatch in FastStream.update_batch")
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=30.0)
            self.assertFalse(th.is_alive())
        self.assertEqual(errors, [])

    def test_no_refcount_or_memory_leaks_across_10000_updates(self) -> None:
        val = 1.23456789
        fs = _fast.FastStream(m=16, window=128, ez=-1, egress=True, alpha=0.05, track_full_mp=False)
        try:
            # Warmup internal buffers & small caches
            for i in range(500):
                fs.update(math.sin(i * 0.1))

            gc.collect()
            rc_val_before = sys.getrefcount(val)
            rc_fs_before = sys.getrefcount(fs)

            tracemalloc.start()
            snap_before = tracemalloc.take_snapshot()
            for i in range(10_000):
                t = fs.update(val if (i & 1) else math.sin(i * 0.05))
                del t
            gc.collect()
            snap_after = tracemalloc.take_snapshot()
            tracemalloc.stop()

            self.assertEqual(sys.getrefcount(val), rc_val_before)
            self.assertEqual(sys.getrefcount(fs), rc_fs_before)

            # Total Python heap growth across 10,000 updates must be tiny (< 64 KiB)
            diffs = snap_after.compare_to(snap_before, "lineno")
            net_bytes = sum(d.size_diff for d in diffs)
            self.assertLess(net_bytes, 65_536, f"Unexpected Python memory growth: {net_bytes} bytes")
        finally:
            fs.close()

        # Repeated create/close/dealloc cycles must not leak Python objects
        for _ in range(50):
            s = _fast.FastStream(m=8, window=32, ez=-1, egress=True, alpha=0.05, track_full_mp=False)
            s.update(1.0)
            del s
        gc.collect()


# ---------------------------------------------------------------------------
# 2. Zero-Dependency AI Agent & DataFrame Ecosystem Tests (integrations.py)
# ---------------------------------------------------------------------------


class TestIntegrationsZeroDep(unittest.TestCase):
    EXPECTED_TOOLS = {
        "ts_digest",
        "ts_drilldown",
        "ts_anomalies",
        "ts_regimes",
        "ts_leak_scan",
        "ts_stream_create",
        "ts_stream_push",
        "ts_swarm_sift",
    }

    def test_zero_third_party_imports_in_subprocess(self) -> None:
        code = (
            "import sys; "
            "import flashmp.integrations as fi; "
            "assert 'numpy' not in sys.modules; "
            "assert 'pandas' not in sys.modules; "
            "assert 'polars' not in sys.modules; "
            "assert len(fi.openai_tools()) == 8; "
            "assert len(fi.anthropic_tools()) == 8"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_DIR)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_openai_and_anthropic_tool_schemas(self) -> None:
        oa_chat = openai_tools()
        self.assertEqual(len(oa_chat), 8)
        chat_names = {t["function"]["name"] for t in oa_chat}
        self.assertEqual(chat_names, self.EXPECTED_TOOLS)
        for t in oa_chat:
            self.assertEqual(t["type"], "function")
            fn = t["function"]
            self.assertIsInstance(fn["description"], str)
            self.assertEqual(fn["parameters"]["type"], "object")
            self.assertIn("properties", fn["parameters"])

        # OpenAI Responses API format + strict mode
        oa_resp = openai_tools(format="responses", strict=True)
        self.assertEqual({t["name"] for t in oa_resp}, self.EXPECTED_TOOLS)
        for t in oa_resp:
            self.assertEqual(t["type"], "function")
            self.assertTrue(t["strict"])
            self.assertIs(t["parameters"]["additionalProperties"], False)

        with self.assertRaises(ValueError):
            openai_tools(format="invalid")

        # Anthropic format
        ant = anthropic_tools()
        self.assertEqual(len(ant), 8)
        self.assertEqual({t["name"] for t in ant}, self.EXPECTED_TOOLS)
        for t in ant:
            self.assertIn("description", t)
            self.assertEqual(t["input_schema"]["type"], "object")
            self.assertIn("properties", t["input_schema"])

        # Deep-copy isolation: mutating returned tool does not pollute future calls
        ant[0]["input_schema"]["properties"]["__mutated__"] = {"type": "string"}
        self.assertNotIn("__mutated__", anthropic_tools()[0]["input_schema"]["properties"])

    def test_agent_toolkit_end_to_end_and_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            allowed_dir = tmp_path / "workspace"
            secret_dir = tmp_path / "secret"
            allowed_dir.mkdir()
            secret_dir.mkdir()

            ok_csv = allowed_dir / "sensor.csv"
            ok_csv.write_text(
                "t,val\n" + "\n".join(f"{i},{math.sin(i * 0.25):.6f}" for i in range(120)) + "\n",
                encoding="utf-8",
            )
            secret_json = secret_dir / "creds.json"
            secret_json.write_text(json.dumps({"api_key_secret": [1.0, 2.0, 3.0]}), encoding="utf-8")
            evil_symlink = allowed_dir / "escape.json"
            evil_symlink.symlink_to(secret_json)

            with AgentToolkit(allowed_dirs=[allowed_dir]) as tk:
                self.assertEqual(len(tk.openai_tools()), 8)
                self.assertEqual(len(tk.anthropic_tools()), 8)
                self.assertEqual(set(tk.tool_names), self.EXPECTED_TOOLS)

                # 1. ts_digest -> ts_drilldown (including OpenAI JSON-string arguments!)
                series = [math.sin(2.0 * math.pi * i / 20.0) for i in range(600)]
                series[300:320] = [0.9 * ((-1.0) ** k) for k in range(20)]

                dig = tk.execute("ts_digest", json.dumps({"values": series, "m": 20}))
                self.assertIn("handle", dig)
                self.assertGreater(dig["token_savings_pct"], 85.0)

                dd_slice = tk.execute(
                    "ts_drilldown",
                    {"handle": dig["handle"], "start": 300, "end": 320},
                )
                self.assertEqual(dd_slice["values"], series[300:320])

                dd_disc = tk.execute(
                    "ts_drilldown",
                    {"handle": dig["handle"], "discord_rank": 0},
                )
                self.assertEqual(len(dd_disc["values"]), 20)

                # 1b. ts_anomalies & ts_regimes
                anom = tk.execute("ts_anomalies", {"values": series, "m": 20, "k": 2})
                self.assertGreaterEqual(len(anom["anomalies"]), 1)
                self.assertLessEqual(abs(anom["anomalies"][0]["index"] - 300), 20)

                reg_series = [math.sin(2.0 * math.pi * i / 16.0) for i in range(250)] + [
                    math.sin(2.0 * math.pi * i / 40.0) for i in range(250)
                ]
                reg = tk.execute("ts_regimes", {"values": reg_series, "m": 16, "max_regimes": 4})
                self.assertIn("changepoints", reg)
                self.assertIn("regimes", reg)

                # 2. ts_leak_scan (detects negative affine leak)
                ref = [math.sin(i * 0.21) + 0.2 * math.cos(i * 0.77) for i in range(160)]
                qry = [math.cos(i * 0.31) + 0.2 * math.sin(i * 1.13) for i in range(140)]
                for k in range(20):
                    qry[50 + k] = -3.0 * ref[70 + k] + 12.5
                leak = tk.execute(
                    "ts_leak_scan",
                    {"ref_values": ref, "qry_values": qry, "m": 20, "corr_thresh": 0.999},
                )
                self.assertTrue(leak["is_contaminated"])
                self.assertGreaterEqual(leak["contaminated_windows"], 1)

                # 3. ts_stream_create -> ts_stream_push
                st = tk.execute(
                    "ts_stream_create",
                    {"m": 12, "window": 64, "initial_values": series[:80]},
                )
                self.assertIn("handle", st)
                pushed = tk.execute(
                    "ts_stream_push",
                    {"handle": st["handle"], "values": series[80:100]},
                )
                self.assertEqual(pushed["count"], 20)

                # 4. ts_swarm_sift -> ts_drilldown on swarm handle
                channels = {
                    f"ch{c}": [math.sin(2.0 * math.pi * i / 16.0 + c * 0.1) for i in range(180)]
                    for c in range(6)
                }
                channels["ch3"][130:146] = [0.95 * ((-1.0) ** k) for k in range(16)]
                sw = tk.execute(
                    "ts_swarm_sift",
                    {"channels": channels, "m": 16, "window": 100, "max_tokens": 250},
                )
                self.assertIn("handle", sw)
                self.assertIn("brief", sw)
                sw_dd = tk.execute("ts_drilldown", {"handle": sw["handle"], "channel": "ch3", "length": 16})
                self.assertEqual(len(sw_dd["values"]), 16)

                # 5. Sandbox enforcement: allowed file succeeds, outside & symlink blocked!
                ok_res = tk.execute("ts_digest", {"path": str(ok_csv), "m": 12})
                self.assertEqual(ok_res["n_points"], 120)

                with self.assertRaises(PermissionError):
                    tk.execute("ts_digest", {"path": str(secret_json), "column": "__probe__"})
                with self.assertRaises(PermissionError):
                    tk.execute("ts_digest", {"path": str(evil_symlink)})

                # Non-raising execute_json safely encodes sandbox error without secret leak
                err_json = json.loads(tk.execute_json("ts_digest", {"path": str(secret_json)}))
                self.assertTrue(err_json["is_error"])
                self.assertIn("Access denied", err_json["error"])
                self.assertNotIn("api_key_secret", err_json["error"])

    def test_mcp_client_config_json_and_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d1 = Path(tmp) / "ws1"
            d2 = Path(tmp) / "ws2"
            d1.mkdir()
            d2.mkdir()

            for client in ("claude-code", "cursor", "gemini"):
                raw_no_env = mcp_client_config(client)
                parsed_no_env = json.loads(raw_no_env)
                srv_no_env = parsed_no_env["mcpServers"]["flashmp"]
                self.assertEqual(srv_no_env["command"], "flashmp-mcp")
                self.assertNotIn("env", srv_no_env)

                raw_with_env = mcp_client_config(client, allowed_dirs=[d1, d2])
                parsed = json.loads(raw_with_env)
                env_val = parsed["mcpServers"]["flashmp"]["env"]["FLASHMP_MCP_ALLOWED_DIRS"]
                self.assertEqual(env_val, os.pathsep.join([str(d1.resolve()), str(d2.resolve())]))

            # Codex TOML config must parse cleanly via stdlib tomllib!
            toml_no_env = mcp_client_config("codex")
            parsed_toml_no_env = tomllib.loads(toml_no_env)
            self.assertEqual(parsed_toml_no_env["mcp_servers"]["flashmp"]["command"], "flashmp-mcp")
            self.assertNotIn("env", parsed_toml_no_env["mcp_servers"]["flashmp"])

            toml_with_env = mcp_client_config("codex", allowed_dirs=d1)
            parsed_toml = tomllib.loads(toml_with_env)
            self.assertEqual(
                parsed_toml["mcp_servers"]["flashmp"]["env"]["FLASHMP_MCP_ALLOWED_DIRS"],
                str(d1.resolve()),
            )

            with self.assertRaises(ValueError):
                mcp_client_config("unknown-client")
            with self.assertRaises(ValueError):
                mcp_client_config("claude-code", allowed_dirs=[])

    def test_install_skill_single_and_all(self) -> None:
        expected_bytes = (ROOT / "SKILL.md").read_bytes()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            for target in ("claude", "cursor", "codex"):
                sub = tmp_path / f"single_{target}"
                paths = install_skill(target=target, dest_dir=sub)
                self.assertEqual(len(paths), 1)
                self.assertTrue(paths[0].is_file())
                self.assertEqual(paths[0].name, "SKILL.md")
                self.assertEqual(paths[0].read_bytes(), expected_bytes)

            all_root = tmp_path / "all_targets"
            all_paths = install_skill(target="all", dest_dir=all_root)
            self.assertEqual(len(all_paths), 3)
            self.assertEqual(len(set(all_paths)), 3)
            for p in all_paths:
                self.assertTrue(p.is_file())
                self.assertEqual(p.read_bytes(), expected_bytes)

            with self.assertRaises(ValueError):
                install_skill(target="invalid", dest_dir=tmp_path)

    def test_extract_series_duck_typed_pandas_polars_and_builtins(self) -> None:
        sig = [math.sin(i * 0.2) for i in range(60)]
        ts = list(range(60))

        # 1. Mock Pandas DataFrame: skips string col and 'timestamp', picks 'latency'
        pd_df = _MockPandasDataFrame(
            {
                "host": ["web-1"] * 60,
                "timestamp": _MockPandasSeries(ts, name="timestamp", dtype=_FakeDtype(kind="i", name="int64")),
                "latency": _MockPandasSeries(sig, name="latency", dtype=_FakeDtype(kind="f", name="float64")),
                "flag": _MockPandasSeries([True, False] * 30, name="flag", dtype=_FakeDtype(kind="b", name="bool")),
            }
        )
        auto_pd = extract_series(pd_df)
        _assert_close_seq(self, auto_pd, sig, tol=1e-15)
        by_name_pd = extract_series(pd_df, column="timestamp")
        _assert_close_seq(self, by_name_pd, ts, tol=1e-15)
        by_idx_pd = extract_series(pd_df, column=2)
        _assert_close_seq(self, by_idx_pd, sig, tol=1e-15)
        with self.assertRaises(TypeError):
            extract_series(pd_df, column="flag")
        with self.assertRaises(KeyError):
            extract_series(pd_df, column="nonexistent")
        with self.assertRaises(IndexError):
            extract_series(pd_df, column=99)

        # 2. Mock Polars DataFrame (to_numpy raises ModuleNotFoundError when numpy absent -> uses to_list!)
        pl_df = _MockPolarsDataFrame(
            [
                _MockPolarsSeries("label", ["ok"] * 60, dtype_name="String", is_num=False),
                _MockPolarsSeries("t", ts, dtype_name="Int64", is_num=True),
                _MockPolarsSeries("temp_c", sig, dtype_name="Float64", is_num=True),
                _MockPolarsSeries("is_err", [False] * 60, dtype_name="Boolean", is_num=False),
            ]
        )
        auto_pl = extract_series(pl_df)
        _assert_close_seq(self, auto_pl, sig, tol=1e-15)
        named_pl = extract_series(pl_df, column="temp_c")
        _assert_close_seq(self, named_pl, sig, tol=1e-15)
        # Single Polars Series directly
        pl_s = _MockPolarsSeries("temp_c", sig, dtype_name="Float64", is_num=True)
        _assert_close_seq(self, extract_series(pl_s), sig, tol=1e-15)
        _assert_close_seq(self, extract_series(pl_s, column="temp_c"), sig, tol=1e-15)
        with self.assertRaises(TypeError):
            extract_series(_MockPolarsSeries("b", [True, False], dtype_name="Boolean", is_num=False))

        # 3. dict, list, tuple, array.array (zero-copy for 'd')
        d_map = {"time": ts, "metric": sig}
        _assert_close_seq(self, extract_series(d_map), sig, tol=1e-15)
        _assert_close_seq(self, extract_series(d_map, column="time"), ts, tol=1e-15)
        with self.assertRaises(KeyError):
            extract_series(d_map, column="missing")

        arr_d = array.array("d", sig)
        self.assertIs(extract_series(arr_d), arr_d)
        _assert_close_seq(self, extract_series(array.array("f", [1.5, 2.5, 3.5])), [1.5, 2.5, 3.5], tol=1e-6)
        _assert_close_seq(self, extract_series(sig), sig, tol=1e-15)
        _assert_close_seq(self, extract_series(tuple(sig)), sig, tol=1e-15)

        for bad in ("abc", b"123", True, [True, False], [1 + 2j, 3 + 4j], ["x", "y"]):
            with self.assertRaises(TypeError):
                extract_series(bad)

        # Feed extracted series straight into flashmp.profile and _fast.fast_self_join!
        res = flashmp.profile(auto_pl, 12)
        self.assertEqual(res.n, 60 - 12 + 1)

    @unittest.skipIf(np is None, "requires numpy")
    def test_extract_series_numpy_zero_copy_and_2d(self) -> None:
        arr = np.linspace(0.0, 5.0, 50, dtype=np.float64)
        out = extract_series(arr)
        self.assertIs(out, arr)  # zero-copy for aligned C-contiguous float64!

        # Unaligned NumPy view safely copied to aligned float64
        raw = np.zeros(50 * 8 + 3, dtype=np.uint8)
        u = np.ndarray((50,), dtype=np.float64, buffer=raw[1 : 1 + 50 * 8])
        u[:] = arr
        out_u = extract_series(u)
        self.assertTrue(out_u.flags.aligned)
        self.assertTrue(np.array_equal(out_u, arr))

        # 2D NumPy column extraction
        mat = np.column_stack([np.arange(50, dtype=np.float64), arr])
        self.assertTrue(np.array_equal(extract_series(mat, column=1), arr))
        with self.assertRaises(ValueError):
            extract_series(mat)

        # Mock Pandas Series backed by NumPy preserves zero-copy NumPy output
        ps = _MockPandasSeries(arr.tolist(), name="val", dtype=arr.dtype, with_numpy=True)
        out_ps = extract_series(ps)
        self.assertIsInstance(out_ps, np.ndarray)
        self.assertEqual(out_ps.dtype, np.float64)
        self.assertTrue(np.array_equal(out_ps, arr))

        # Reject bool/complex/string ndarrays
        for bad in (np.array([True, False]), np.array([1 + 1j, 2 + 2j]), np.array(["a", "b"])):
            with self.assertRaises(TypeError):
                extract_series(bad)

    def test_direct_dataframe_and_series_in_stump_and_digest(self) -> None:
        n = 80
        ts = [float(i) for i in range(n)]
        s1 = [math.sin(2.0 * math.pi * i / 16.0) for i in range(n)]
        s2 = [math.cos(2.0 * math.pi * i / 16.0) for i in range(n)]

        # 1. Direct Mock Pandas Series -> stump() & digest()
        pd_ser = _MockPandasSeries(s1, name="cpu_pct", dtype=_FakeDtype(kind="f", name="float64"))
        mp_ser = flashmp.stump(pd_ser, 12)
        mp_ref = flashmp.stump(s1, 12)
        _assert_close_seq(self, mp_ser.P_, mp_ref.P_, tol=1e-14)

        d_ser = flashmp.digest(pd_ser, m=12)
        self.assertEqual(d_ser.channel_names, ["cpu_pct"])
        self.assertEqual(d_ser.n_points, n)
        self.assertEqual(d_ser.drilldown(10, 22)["values"], s1[10:22])

        # 2. Direct Mock Polars DataFrame with timestamp + 2 numeric metric cols -> multivariate digest()
        pl_df = _MockPolarsDataFrame(
            [
                _MockPolarsSeries("timestamp", ts, dtype_name="Float64", is_num=True),
                _MockPolarsSeries("latency_ms", s1, dtype_name="Float64", is_num=True),
                _MockPolarsSeries("qps", s2, dtype_name="Float64", is_num=True),
            ]
        )
        d_df = flashmp.digest(pl_df, m=12)
        self.assertEqual(d_df.n_channels, 2)
        self.assertEqual(d_df.channel_names, ["latency_ms", "qps"])
        self.assertEqual(d_df.drilldown(5, 15, channel="qps")["values"], s2[5:15])

        # 3. Direct Mock Polars DataFrame -> stump() skips timestamp and extracts latency_ms
        mp_df = flashmp.stump(pl_df, 12)
        _assert_close_seq(self, mp_df.P_, mp_ref.P_, tol=1e-14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
