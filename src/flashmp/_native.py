"""Low-level ctypes loader, C-ABI bindings, and zero-copy buffer helpers for flashmp."""

from __future__ import annotations

import array
import ctypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_INT64_CODE = "q" if array.array("q").itemsize == 8 else "l"
_INT32_CODE = "i" if array.array("i").itemsize == 4 else "l"

_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)
_INT64_PTR = ctypes.POINTER(ctypes.c_int64)
_INT32_PTR = ctypes.POINTER(ctypes.c_int32)

NULL_DOUBLE = _DOUBLE_PTR()
NULL_INT64 = _INT64_PTR()
NULL_INT32 = _INT32_PTR()


class CTickResult(ctypes.Structure):
    """Exact C-ABI mirror of fmp_tick_result_t in flashmp.hpp."""

    _fields_ = [
        ("gid", ctypes.c_int64),
        ("left_dist", ctypes.c_double),
        ("left_idx", ctypes.c_int64),
        ("discord_z", ctypes.c_double),
        ("p_value", ctypes.c_double),
        ("e_value", ctypes.c_double),
        ("is_alert", ctypes.c_int32),
        ("n_repaired", ctypes.c_int32),
    ]


_LIB_CACHE: ctypes.CDLL | None = None


def _lib_filename() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "libflashmp.dylib"
    if system == "windows":
        return "flashmp.dll"
    return "libflashmp.so"


def _candidate_lib_paths() -> list[Path]:
    libname = _lib_filename()
    pkg_dir = Path(__file__).resolve().parent
    csrc_dir = pkg_dir / "_csrc"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    cache_base = Path(xdg_cache) if xdg_cache else (Path.home() / ".cache")
    paths: list[Path] = []
    env_override = os.environ.get("FLASHMP_LIB")
    if env_override:
        paths.append(Path(env_override))
    paths.append(csrc_dir / libname)
    paths.append(pkg_dir / libname)
    paths.append(cache_base / "flashmp" / libname)
    return paths


def _build_native_library() -> Path:
    pkg_dir = Path(__file__).resolve().parent
    csrc_dir = pkg_dir / "_csrc"
    src_cpp = csrc_dir / "flashmp.cpp"
    if not src_cpp.is_file():
        raise RuntimeError(
            f"Native library {_lib_filename()} not found and C++ source missing at {src_cpp}"
        )

    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError(
            "Native library not found and neither g++ nor clang++ is available to build it."
        )

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    cache_dir = (Path(xdg_cache) if xdg_cache else (Path.home() / ".cache")) / "flashmp"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / _lib_filename()
    except OSError:
        out_path = Path(tempfile.gettempdir()) / f"flashmp_{os.getuid()}_{_lib_filename()}"

    tmp_out = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    system = platform.system().lower()
    shared_flag = "-dynamiclib" if system == "darwin" else "-shared"
    base_cmd = [
        compiler,
        "-O3",
        "-std=c++20",
        "-fno-math-errno",
        "-funroll-loops",
        shared_flag,
        "-fPIC",
        "-pthread",
        str(src_cpp),
        "-o",
        str(tmp_out),
    ]

    # Try with -march=native first, fallback to portable optimization flags.
    attempts = [base_cmd[:2] + ["-march=native"] + base_cmd[2:], base_cmd]
    last_err = ""
    for cmd in attempts:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and tmp_out.is_file():
            os.replace(tmp_out, out_path)
            return out_path
        last_err = proc.stderr or proc.stdout
        if tmp_out.exists():
            try:
                tmp_out.unlink()
            except OSError:
                pass
    raise RuntimeError(f"Failed to compile flashmp native library: {last_err.strip()}")


def _configure_bindings(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.fmp_version.argtypes = []
    lib.fmp_version.restype = ctypes.c_char_p

    lib.fmp_simd_isa.argtypes = []
    lib.fmp_simd_isa.restype = ctypes.c_char_p

    lib.fmp_self_join.argtypes = [
        _DOUBLE_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
        _DOUBLE_PTR,
        _INT64_PTR,
        _DOUBLE_PTR,
        _INT64_PTR,
        _DOUBLE_PTR,
        _INT64_PTR,
    ]
    lib.fmp_self_join.restype = ctypes.c_int

    lib.fmp_ab_join.argtypes = [
        _DOUBLE_PTR,
        ctypes.c_int64,
        _DOUBLE_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
        _DOUBLE_PTR,
        _INT64_PTR,
        _DOUBLE_PTR,
        _INT64_PTR,
    ]
    lib.fmp_ab_join.restype = ctypes.c_int

    lib.fmp_leak_scan.argtypes = [
        _DOUBLE_PTR,
        ctypes.c_int64,
        _DOUBLE_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        _DOUBLE_PTR,
        _INT64_PTR,
        _INT64_PTR,
    ]
    lib.fmp_leak_scan.restype = ctypes.c_int

    lib.fmp_mstump.argtypes = [
        _DOUBLE_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
        _DOUBLE_PTR,
        _INT64_PTR,
    ]
    lib.fmp_mstump.restype = ctypes.c_int

    lib.fmp_floss_cac.argtypes = [
        _INT64_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        _DOUBLE_PTR,
    ]
    lib.fmp_floss_cac.restype = ctypes.c_int

    lib.fmp_auto_window.argtypes = [
        _DOUBLE_PTR,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.fmp_auto_window.restype = ctypes.c_int64

    lib.fmp_stream_create.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
    ]
    lib.fmp_stream_create.restype = ctypes.c_void_p

    lib.fmp_stream_destroy.argtypes = [ctypes.c_void_p]
    lib.fmp_stream_destroy.restype = None

    lib.fmp_stream_init.argtypes = [
        ctypes.c_void_p,
        _DOUBLE_PTR,
        ctypes.c_int64,
    ]
    lib.fmp_stream_init.restype = ctypes.c_int

    lib.fmp_stream_update.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.POINTER(CTickResult),
    ]
    lib.fmp_stream_update.restype = ctypes.c_int

    lib.fmp_stream_update_batch.argtypes = [
        ctypes.c_void_p,
        _DOUBLE_PTR,
        ctypes.c_int64,
        _DOUBLE_PTR,
        _INT64_PTR,
        _DOUBLE_PTR,
        _INT32_PTR,
    ]
    lib.fmp_stream_update_batch.restype = ctypes.c_int

    lib.fmp_stream_active_len.argtypes = [ctypes.c_void_p]
    lib.fmp_stream_active_len.restype = ctypes.c_int64

    lib.fmp_stream_gid_min.argtypes = [ctypes.c_void_p]
    lib.fmp_stream_gid_min.restype = ctypes.c_int64

    lib.fmp_stream_snapshot.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        _DOUBLE_PTR,
        _INT64_PTR,
        _DOUBLE_PTR,
        _INT64_PTR,
    ]
    lib.fmp_stream_snapshot.restype = ctypes.c_int

    return lib


def get_lib() -> ctypes.CDLL:
    """Return the loaded native flashmp shared library, building it if missing."""
    global _LIB_CACHE
    if _LIB_CACHE is not None:
        return _LIB_CACHE

    lib_path: Path | None = None
    for cand in _candidate_lib_paths():
        if cand.is_file():
            lib_path = cand
            break

    if lib_path is None:
        lib_path = _build_native_library()

    lib = ctypes.CDLL(str(lib_path))
    _LIB_CACHE = _configure_bindings(lib)
    return _LIB_CACHE


def version() -> str:
    """Return the native flashmp library version string."""
    raw = get_lib().fmp_version()
    return raw.decode("utf-8") if raw else "0.1.0"


def simd_isa() -> str:
    """Return the compiled SIMD instruction set identifier (e.g. AVX2+FMA, ARM64-NEON)."""
    raw = get_lib().fmp_simd_isa()
    return raw.decode("utf-8") if raw else "SCALAR"


def _get_numpy() -> Any | None:
    """Return already-imported numpy module without triggering eager import."""
    return sys.modules.get("numpy")


def _use_numpy(is_numpy_input: bool = False) -> bool:
    return is_numpy_input or (_get_numpy() is not None)


def _array_ptr(buf: array.array, ptr_type: Any) -> Any:
    if len(buf) == 0:
        return ptr_type()
    addr, _ = buf.buffer_info()
    return ctypes.cast(addr, ptr_type)


def _check_real_numpy_dtype(arr: Any, allow_int_only: bool = False) -> None:
    kind = arr.dtype.kind
    if kind in ("c", "S", "U", "V"):
        raise TypeError(f"Unsupported non-real dtype {arr.dtype}; expected real numeric values")
    if kind == "b":
        raise TypeError("Boolean arrays are not supported as numeric time series; cast explicitly if intended")
    if kind == "O":
        for v in arr.flat:
            if isinstance(v, (bool, complex, str, bytes)) or v is None:
                raise TypeError(f"Object array contains non-real element {v!r}")
    if allow_int_only and kind not in ("i", "u"):
        raise TypeError(f"Expected integer dtype, got {arr.dtype}")


def _coerce_real_scalar(x: Any) -> float:
    if isinstance(x, (bool, complex, str, bytes)) or x is None:
        raise TypeError(f"Expected real number, got {type(x).__name__}: {x!r}")
    return float(x)


def _as_c_double_array(data: Any) -> tuple[Any, int, Any, bool]:
    """Convert 1D numeric input to (c_double*, length, owner_keepalive, is_numpy).

    Zero-copy for C-contiguous, aligned float64 NumPy ndarrays and array.array('d').
    """
    if isinstance(data, (str, bytes, bytearray, bool)):
        raise TypeError(f"Expected 1D numeric sequence, got {type(data).__name__}")
    np = _get_numpy()
    if np is not None and isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {data.shape}")
        _check_real_numpy_dtype(data)
        if data.dtype == np.float64 and data.flags.c_contiguous and data.flags.aligned:
            arr = data
        else:
            arr = np.require(data, dtype=np.float64, requirements=("C", "A"))
            if not arr.flags.aligned:
                arr = np.array(arr, dtype=np.float64, order="C", copy=True)
        ptr = arr.ctypes.data_as(_DOUBLE_PTR) if arr.size > 0 else NULL_DOUBLE
        return ptr, int(arr.size), arr, True

    if isinstance(data, array.array) and data.typecode == "d":
        return _array_ptr(data, _DOUBLE_PTR), len(data), data, False

    if np is not None and hasattr(data, "__array__"):
        raw_arr = np.asarray(data)
        if raw_arr.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {raw_arr.shape}")
        _check_real_numpy_dtype(raw_arr)
        arr = np.require(raw_arr, dtype=np.float64, requirements=("C", "A"))
        if not arr.flags.aligned:
            arr = np.array(arr, dtype=np.float64, order="C", copy=True)
        ptr = arr.ctypes.data_as(_DOUBLE_PTR) if arr.size > 0 else NULL_DOUBLE
        return ptr, int(arr.size), arr, True

    if isinstance(data, (list, tuple)):
        if any(isinstance(x, (bool, complex, str, bytes)) or x is None for x in data):
            raise TypeError("Time series must contain only real numeric values (not bool/complex/str/None)")
        buf = array.array("d", data)
        return _array_ptr(buf, _DOUBLE_PTR), len(buf), buf, False

    buf = array.array("d", (_coerce_real_scalar(x) for x in data))
    return _array_ptr(buf, _DOUBLE_PTR), len(buf), buf, False


def _as_c_int64_array(data: Any) -> tuple[Any, int, Any, bool]:
    """Convert 1D integer input to (c_int64*, length, owner_keepalive, is_numpy)."""
    if isinstance(data, (str, bytes, bytearray, bool)):
        raise TypeError(f"Expected 1D integer sequence, got {type(data).__name__}")
    np = _get_numpy()
    if np is not None and isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {data.shape}")
        _check_real_numpy_dtype(data)
        if data.dtype == np.int64 and data.flags.c_contiguous and data.flags.aligned:
            arr = data
        else:
            arr = np.require(data, dtype=np.int64, requirements=("C", "A"))
            if not arr.flags.aligned:
                arr = np.array(arr, dtype=np.int64, order="C", copy=True)
        ptr = arr.ctypes.data_as(_INT64_PTR) if arr.size > 0 else NULL_INT64
        return ptr, int(arr.size), arr, True

    if isinstance(data, array.array) and data.typecode == _INT64_CODE:
        return _array_ptr(data, _INT64_PTR), len(data), data, False

    if np is not None and hasattr(data, "__array__"):
        raw_arr = np.asarray(data)
        if raw_arr.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {raw_arr.shape}")
        _check_real_numpy_dtype(raw_arr)
        arr = np.require(raw_arr, dtype=np.int64, requirements=("C", "A"))
        if not arr.flags.aligned:
            arr = np.array(arr, dtype=np.int64, order="C", copy=True)
        ptr = arr.ctypes.data_as(_INT64_PTR) if arr.size > 0 else NULL_INT64
        return ptr, int(arr.size), arr, True

    try:
        buf = array.array(_INT64_CODE, data)
    except TypeError:
        buf = array.array(_INT64_CODE, (int(x) for x in data))
    return _array_ptr(buf, _INT64_PTR), len(buf), buf, False


def _as_c_double_2d(data: Any) -> tuple[Any, int, int, Any, bool]:
    """Convert 2D (D, N) row-major numeric input to (c_double*, D, N, owner, is_numpy)."""
    if isinstance(data, (str, bytes, bytearray, bool)):
        raise TypeError(f"Expected 2D numeric sequence, got {type(data).__name__}")
    np = _get_numpy()
    if np is not None and isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError(f"Expected 2D (D, N) array, got shape {data.shape}")
        _check_real_numpy_dtype(data)
        if data.dtype == np.float64 and data.flags.c_contiguous and data.flags.aligned:
            arr = data
        else:
            arr = np.require(data, dtype=np.float64, requirements=("C", "A"))
            if not arr.flags.aligned:
                arr = np.array(arr, dtype=np.float64, order="C", copy=True)
        D, N = int(arr.shape[0]), int(arr.shape[1])
        ptr = arr.ctypes.data_as(_DOUBLE_PTR) if arr.size > 0 else NULL_DOUBLE
        return ptr, D, N, arr, True

    if np is not None and hasattr(data, "__array__"):
        raw_arr = np.asarray(data)
        if raw_arr.ndim != 2:
            raise ValueError(f"Expected 2D (D, N) array, got shape {raw_arr.shape}")
        _check_real_numpy_dtype(raw_arr)
        arr = np.require(raw_arr, dtype=np.float64, requirements=("C", "A"))
        if not arr.flags.aligned:
            arr = np.array(arr, dtype=np.float64, order="C", copy=True)
        D, N = int(arr.shape[0]), int(arr.shape[1])
        ptr = arr.ctypes.data_as(_DOUBLE_PTR) if arr.size > 0 else NULL_DOUBLE
        return ptr, D, N, arr, True

    rows = list(data)
    if not rows:
        raise ValueError("Expected non-empty 2D sequence of shape (D, N)")
    D = len(rows)
    flat = array.array("d")
    N = -1
    for idx, row in enumerate(rows):
        if isinstance(row, (str, bytes, bytearray, bool)):
            raise TypeError(f"Row {idx} must be a numeric sequence, got {type(row).__name__}")
        if isinstance(row, array.array) and row.typecode == "d":
            r = row
        elif isinstance(row, (list, tuple)):
            if any(isinstance(x, (bool, complex, str, bytes)) or x is None for x in row):
                raise TypeError(f"Row {idx} contains non-real numeric values")
            r = array.array("d", row)
        else:
            r = array.array("d", (_coerce_real_scalar(x) for x in row))
        if N < 0:
            N = len(r)
        elif len(r) != N:
            raise ValueError(f"Row {idx} has length {len(r)}, expected {N}")
        flat.extend(r)
    return _array_ptr(flat, _DOUBLE_PTR), D, max(0, N), flat, False


def _alloc_double(n: int, use_numpy: bool) -> tuple[Any, Any]:
    if use_numpy:
        np = _get_numpy()
        arr = np.empty(n, dtype=np.float64)
        ptr = arr.ctypes.data_as(_DOUBLE_PTR) if n > 0 else NULL_DOUBLE
        return arr, ptr
    buf = array.array("d", (0.0,)) * n
    return buf, _array_ptr(buf, _DOUBLE_PTR)


def _alloc_int64(n: int, use_numpy: bool, fill: int = 0) -> tuple[Any, Any]:
    if use_numpy:
        np = _get_numpy()
        arr = np.full(n, fill, dtype=np.int64) if fill != 0 else np.empty(n, dtype=np.int64)
        ptr = arr.ctypes.data_as(_INT64_PTR) if n > 0 else NULL_INT64
        return arr, ptr
    buf = array.array(_INT64_CODE, (fill,)) * n
    return buf, _array_ptr(buf, _INT64_PTR)


def _alloc_int32(n: int, use_numpy: bool) -> tuple[Any, Any]:
    if use_numpy:
        np = _get_numpy()
        arr = np.empty(n, dtype=np.int32)
        ptr = arr.ctypes.data_as(_INT32_PTR) if n > 0 else NULL_INT32
        return arr, ptr
    buf = array.array(_INT32_CODE, (0,)) * n
    return buf, _array_ptr(buf, _INT32_PTR)


def _alloc_double_2d(D: int, n: int, use_numpy: bool) -> tuple[Any, Any]:
    if use_numpy:
        np = _get_numpy()
        arr = np.empty((D, n), dtype=np.float64)
        ptr = arr.ctypes.data_as(_DOUBLE_PTR) if (D * n) > 0 else NULL_DOUBLE
        return arr, ptr
    buf = array.array("d", (0.0,)) * (D * n)
    return buf, _array_ptr(buf, _DOUBLE_PTR)


def _alloc_int64_2d(D: int, n: int, use_numpy: bool) -> tuple[Any, Any]:
    if use_numpy:
        np = _get_numpy()
        arr = np.empty((D, n), dtype=np.int64)
        ptr = arr.ctypes.data_as(_INT64_PTR) if (D * n) > 0 else NULL_INT64
        return arr, ptr
    buf = array.array(_INT64_CODE, (0,)) * (D * n)
    return buf, _array_ptr(buf, _INT64_PTR)


def _default_excl_zone(m: int) -> int:
    return (int(m) + 3) // 4
