"""Exact bounded-window streaming RingMP + Ville E-Process and STUMPY stumpi drop-in."""

from __future__ import annotations

import array
import ctypes
from collections import deque
from dataclasses import dataclass
from typing import Any, NamedTuple

from ._native import (
    CTickResult,
    _alloc_double,
    _alloc_int32,
    _alloc_int64,
    _as_c_double_array,
    _default_excl_zone,
    _get_numpy,
    _use_numpy,
    get_fast,
    get_lib,
)


@dataclass(slots=True)
class TickResult:
    """Exact per-tick streaming result from RingMP + Ville anytime-valid E-process."""

    gid: int
    left_dist: float
    left_idx: int
    discord_z: float
    p_value: float
    e_value: float
    is_alert: bool
    n_repaired: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "gid": self.gid,
            "left_dist": self.left_dist,
            "left_idx": self.left_idx,
            "discord_z": self.discord_z,
            "p_value": self.p_value,
            "e_value": self.e_value,
            "is_alert": self.is_alert,
            "n_repaired": self.n_repaired,
        }


class BatchTickResult(NamedTuple):
    """Vectorized batch output returned by `StreamMP.update_batch()`."""

    left_dist: Any
    left_idx: Any
    e_value: Any
    is_alert: Any


class StreamSnapshot(NamedTuple):
    """Active-window snapshot returned by `StreamMP.snapshot()`."""

    mp: Any
    idx: Any
    left_mp: Any
    left_idx: Any


class StreamMP:
    """Exact bounded-window streaming Matrix Profile (RingMP) with Ville E-Process alerting."""

    __slots__ = (
        "_fast",
        "_fast_update",
        "_lib",
        "_destroy_fn",
        "_update_fn",
        "_ptr",
        "_c_ptr",
        "_c_res",
        "_c_res_ref",
        "_m",
        "_window",
        "_excl_zone",
        "_egress",
        "_alpha",
        "_track_full_mp",
        "_last_e_value",
        "_last_is_alert",
        "_last_tick",
        "_prefer_numpy",
    )

    def __init__(
        self,
        m: int,
        window: int = 4096,
        excl_zone: int | None = None,
        egress: bool = True,
        alpha: float = 0.05,
        track_full_mp: bool = True,
    ) -> None:
        m_int = int(m)
        win_int = int(window)
        if m_int < 3:
            raise ValueError(f"m must be >= 3, got {m_int}")
        if win_int < 4:
            raise ValueError(f"window must be >= 4, got {win_int}")
        alpha_f = float(alpha)
        if not (0.0 < alpha_f < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha_f}")

        if excl_zone is not None:
            if isinstance(excl_zone, bool):
                raise TypeError("excl_zone must be an integer or None, not bool")
            ez = int(excl_zone)
            if ez < 0:
                raise ValueError(f"excl_zone must be >= 0, got {ez}")
        else:
            ez = -1

        fast = get_fast()
        if fast is not None:
            self._fast: Any = fast.FastStream(
                m_int,
                win_int,
                ez,
                bool(egress),
                alpha_f,
                bool(track_full_mp),
            )
            self._fast_update = self._fast.update
            self._lib = None
            self._destroy_fn = None
            self._update_fn = None
            self._ptr: int | None = 1
            self._c_ptr = None
            self._c_res = None
            self._c_res_ref = None
        else:
            lib = get_lib()
            ptr = lib.fmp_stream_create(
                m_int,
                win_int,
                ez,
                1 if egress else 0,
                alpha_f,
                1 if track_full_mp else 0,
            )
            if not ptr:
                raise RuntimeError("fmp_stream_create failed to allocate native stream state")
            self._fast = None
            self._fast_update = None
            self._lib = lib
            self._destroy_fn = lib.fmp_stream_destroy
            self._update_fn = lib.fmp_stream_update
            self._ptr = int(ptr)
            self._c_ptr = ctypes.c_void_p(self._ptr)
            self._c_res = CTickResult()
            self._c_res_ref = ctypes.byref(self._c_res)

        self._m = m_int
        self._window = win_int
        self._excl_zone = _default_excl_zone(m_int) if excl_zone is None else ez
        self._egress = bool(egress)
        self._alpha = alpha_f
        self._track_full_mp = bool(track_full_mp)
        self._last_e_value: float = 1.0
        self._last_is_alert: bool = False
        self._last_tick: TickResult | None = None
        self._prefer_numpy: bool = False

    def _require_open(self) -> Any:
        if self._ptr is None:
            raise RuntimeError("StreamMP has been closed")
        return self._c_ptr

    def close(self) -> None:
        """Release native C++ stream resources (idempotent)."""
        ptr = getattr(self, "_ptr", None)
        if ptr is not None:
            self._ptr = None
            fast = getattr(self, "_fast", None)
            if fast is not None:
                fast.close()
                self._fast = None
                self._fast_update = None
            else:
                destroy = getattr(self, "_destroy_fn", None)
                if destroy is not None:
                    destroy(ctypes.c_void_p(ptr))

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> StreamMP:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def m(self) -> int:
        return self._m

    @property
    def window(self) -> int:
        return self._window

    @property
    def excl_zone(self) -> int:
        return self._excl_zone

    @property
    def egress(self) -> bool:
        return self._egress

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def track_full_mp(self) -> bool:
        return self._track_full_mp

    def init(self, data: Any) -> StreamMP:
        """Ingest an initial historical batch into the stream."""
        self._require_open()
        ptr, n, owner, is_np = _as_c_double_array(data)
        if is_np:
            self._prefer_numpy = True
        if n == 0:
            return self
        if n > 1:
            if self._fast is not None:
                mv = memoryview(owner)
                self._fast.init(mv[:-1])
                mv.release()
            else:
                rc = self._lib.fmp_stream_init(self._c_ptr, ptr, n - 1)
                if rc != 0:
                    del owner
                    raise RuntimeError(f"fmp_stream_init failed with error code {rc}")
        last_val = float(owner[n - 1])
        del owner
        self.update(last_val)
        return self

    def update(self, x: float) -> TickResult:
        """Ingest one new tick and return its exact causal TickResult."""
        if self._ptr is None:
            raise RuntimeError("StreamMP has been closed")
        if isinstance(x, bool):
            raise TypeError("Tick value x must be a real number, not bool")
        if self._fast_update is not None:
            gid, ld, li, dz, pv, ev, al, nr = self._fast_update(float(x))
            tick = TickResult(gid, ld, li, dz, pv, ev, al, nr)
        else:
            rc = self._update_fn(self._c_ptr, float(x), self._c_res_ref)
            if rc != 0:
                raise RuntimeError(f"fmp_stream_update failed with error code {rc}")
            r = self._c_res
            tick = TickResult(
                gid=int(r.gid),
                left_dist=float(r.left_dist),
                left_idx=int(r.left_idx),
                discord_z=float(r.discord_z),
                p_value=float(r.p_value),
                e_value=float(r.e_value),
                is_alert=bool(r.is_alert),
                n_repaired=int(r.n_repaired),
            )
        self._last_e_value = tick.e_value
        self._last_is_alert = tick.is_alert
        self._last_tick = tick
        return tick

    def update_batch(self, xs: Any) -> BatchTickResult:
        """Ingest a batch of ticks in a single C++ loop with minimal Python overhead."""
        self._require_open()
        ptr, count, owner, is_np = _as_c_double_array(xs)
        if is_np:
            self._prefer_numpy = True
        use_np = _use_numpy(self._prefer_numpy or is_np)

        ld, ld_ptr = _alloc_double(count, use_np)
        li, li_ptr = _alloc_int64(count, use_np)
        ev, ev_ptr = _alloc_double(count, use_np)
        al, al_ptr = _alloc_int32(count, use_np)

        if count > 0:
            if count > 1:
                if self._fast is not None:
                    mv = memoryview(owner)
                    self._fast.update_batch(mv[:-1], ld, li, ev, al)
                    mv.release()
                else:
                    rc = self._lib.fmp_stream_update_batch(
                        self._c_ptr, ptr, count - 1, ld_ptr, li_ptr, ev_ptr, al_ptr
                    )
                    if rc != 0:
                        del owner
                        raise RuntimeError(f"fmp_stream_update_batch failed with error code {rc}")
            last_val = float(owner[count - 1])
            del owner
            last_t = self.update(last_val)
            ld[count - 1] = last_t.left_dist
            li[count - 1] = last_t.left_idx
            ev[count - 1] = last_t.e_value
            al[count - 1] = 1 if last_t.is_alert else 0
        else:
            del owner
        return BatchTickResult(left_dist=ld, left_idx=li, e_value=ev, is_alert=al)

    @property
    def active_len(self) -> int:
        self._require_open()
        if self._fast is not None:
            return int(self._fast.active_len())
        return int(self._lib.fmp_stream_active_len(self._c_ptr))

    @property
    def gid_min(self) -> int:
        self._require_open()
        if self._fast is not None:
            return int(self._fast.gid_min())
        return int(self._lib.fmp_stream_gid_min(self._c_ptr))

    @property
    def e_value(self) -> float:
        return self._last_e_value

    @property
    def is_alert(self) -> bool:
        return self._last_is_alert

    @property
    def last_tick(self) -> TickResult | None:
        return self._last_tick

    def snapshot(self, relative_indices: bool = True) -> StreamSnapshot:
        """Copy the current active window's exact Full-MP and Left-MP."""
        self._require_open()
        active = self.active_len
        use_np = _use_numpy(self._prefer_numpy)

        mp, mp_ptr = _alloc_double(active, use_np)
        idx, idx_ptr = _alloc_int64(active, use_np)
        lmp, lmp_ptr = _alloc_double(active, use_np)
        lidx, lidx_ptr = _alloc_int64(active, use_np)

        if active > 0:
            if self._fast is not None:
                self._fast.snapshot(bool(relative_indices), mp, idx, lmp, lidx)
            else:
                rc = self._lib.fmp_stream_snapshot(
                    self._c_ptr,
                    1 if relative_indices else 0,
                    mp_ptr,
                    idx_ptr,
                    lmp_ptr,
                    lidx_ptr,
                )
                if rc != 0:
                    raise RuntimeError(f"fmp_stream_snapshot failed with error code {rc}")
        return StreamSnapshot(mp=mp, idx=idx, left_mp=lmp, left_idx=lidx)

    @property
    def P_(self) -> Any:
        return self.snapshot(relative_indices=True).mp

    @property
    def I_(self) -> Any:
        return self.snapshot(relative_indices=True).idx

    @property
    def left_P_(self) -> Any:
        return self.snapshot(relative_indices=True).left_mp

    @property
    def left_I_(self) -> Any:
        return self.snapshot(relative_indices=True).left_idx

    def __repr__(self) -> str:
        state = "closed" if self._ptr is None else f"active={self.active_len}, e={self._last_e_value:.2f}"
        return f"StreamMP(m={self._m}, window={self._window}, egress={self._egress}, {state})"


class stumpi:
    """Drop-in STUMPY streaming Matrix Profile (`stumpy.stumpi`) backed by RingMP."""

    __slots__ = ("_stream", "_buf", "_egress", "_use_np")

    def __init__(
        self,
        T: Any,
        m: int,
        egress: bool = True,
        excl_zone: int | None = None,
        alpha: float = 0.05,
    ) -> None:
        _, N, owner, is_np = _as_c_double_array(T)
        m_int = int(m)
        if m_int < 3:
            raise ValueError(f"m must be >= 3, got {m_int}")
        window_subseqs = N - m_int + 1
        if window_subseqs < 4:
            raise ValueError(
                f"Initial series length N={N} with m={m_int} yields {window_subseqs} subsequences; need >= 4"
            )

        self._egress = bool(egress)
        self._use_np = _use_numpy(is_np)
        self._stream = StreamMP(
            m=m_int,
            window=window_subseqs,
            excl_zone=excl_zone,
            egress=self._egress,
            alpha=alpha,
            track_full_mp=True,
        )
        self._stream.init(owner)

        if self._egress:
            self._buf: Any = deque((float(x) for x in owner), maxlen=N)
        else:
            self._buf = array.array("d", (float(x) for x in owner))

    def update(self, t: float) -> TickResult:
        """Append one new point `t` and update the streaming Matrix Profile."""
        val = float(t)
        self._buf.append(val)
        return self._stream.update(val)

    @property
    def P_(self) -> Any:
        return self._stream.P_

    @property
    def I_(self) -> Any:
        return self._stream.I_

    @property
    def left_P_(self) -> Any:
        return self._stream.left_P_

    @property
    def left_I_(self) -> Any:
        return self._stream.left_I_

    @property
    def T_(self) -> Any:
        if _use_numpy(self._use_np):
            np = _get_numpy()
            return np.asarray(self._buf, dtype=np.float64)
        return array.array("d", self._buf)

    @property
    def e_value(self) -> float:
        return self._stream.e_value

    @property
    def is_alert(self) -> bool:
        return self._stream.is_alert

    @property
    def m(self) -> int:
        return self._stream.m

    @property
    def egress(self) -> bool:
        return self._egress

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> stumpi:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"stumpi(m={self._stream.m}, len_T={len(self._buf)}, egress={self._egress})"
