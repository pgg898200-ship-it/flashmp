"""Multi-agent streaming firehose monitor (SwarmMonitor) with CRDT AlertLedger."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .stream import StreamMP, TickResult


def _ord_float(v: float) -> tuple[int, float]:
    """Total canonical ordering over all IEEE-754 floats: finite > +inf > -inf > nan."""
    if math.isfinite(v):
        return (3, v)
    if math.isinf(v):
        return (2, 0.0) if v > 0 else (1, 0.0)
    return (0, 0.0)


def _alert_lattice_key(
    alert: SwarmAlert,
) -> tuple[int, tuple[int, float], tuple[int, float], tuple[int, float], tuple[int, float], int, int, str]:
    """Strict total-order key making SwarmAlert.merge a true commutative, associative, idempotent join."""
    return (
        1 if alert.is_alert else 0,
        _ord_float(alert.e_value),
        _ord_float(alert.discord_z),
        _ord_float(alert.left_dist),
        _ord_float(-alert.p_value),
        alert.gid,
        alert.left_idx,
        alert.handle,
    )


@dataclass(frozen=True, slots=True)
class SwarmAlert:
    """Immutable alert record forming a bounded join-semilattice per (channel_id, epoch)."""

    channel_id: str
    epoch: int
    gid: int
    e_value: float
    discord_z: float
    left_dist: float
    left_idx: int
    p_value: float
    is_alert: bool
    handle: str

    def merge(self, other: SwarmAlert) -> SwarmAlert:
        """Commutative, associative, idempotent total-order lattice join for the same (channel, epoch)."""
        if self.channel_id != other.channel_id or self.epoch != other.epoch:
            raise ValueError(
                f"Cannot merge alerts with mismatched keys: "
                f"({self.channel_id!r}, {self.epoch}) vs ({other.channel_id!r}, {other.epoch})"
            )
        return self if _alert_lattice_key(self) >= _alert_lattice_key(other) else other

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "epoch": self.epoch,
            "gid": self.gid,
            "e_value": self.e_value,
            "discord_z": self.discord_z,
            "left_dist": self.left_dist,
            "left_idx": self.left_idx,
            "p_value": self.p_value,
            "is_alert": self.is_alert,
            "handle": self.handle,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SwarmAlert:
        ch = str(data["channel_id"])
        gid = int(data["gid"])
        return cls(
            channel_id=ch,
            epoch=int(data["epoch"]),
            gid=gid,
            e_value=float(data["e_value"]),
            discord_z=float(data["discord_z"]),
            left_dist=float(data["left_dist"]),
            left_idx=int(data.get("left_idx", -1)),
            p_value=float(data.get("p_value", 1.0)),
            is_alert=bool(data.get("is_alert", True)),
            handle=str(data.get("handle", f"{ch}@{gid}")),
        )


class AlertLedger:
    """Commutative, order-invariant, idempotent CRDT alert ledger keyed by (channel_id, epoch)."""

    __slots__ = ("_entries",)

    def __init__(self, initial: Iterable[SwarmAlert] = ()) -> None:
        self._entries: dict[tuple[str, int], SwarmAlert] = {}
        for item in initial:
            self.record(item)

    def record(self, alert: SwarmAlert) -> SwarmAlert:
        key = (alert.channel_id, alert.epoch)
        existing = self._entries.get(key)
        merged = alert if existing is None else existing.merge(alert)
        self._entries[key] = merged
        return merged

    def merge(self, other: AlertLedger | Iterable[SwarmAlert | dict[str, Any]]) -> AlertLedger:
        if isinstance(other, AlertLedger):
            for alert in other._entries.values():
                self.record(alert)
            return self
        for item in other:
            if isinstance(item, SwarmAlert):
                self.record(item)
            elif isinstance(item, AlertLedger):
                self.merge(item)
            elif isinstance(item, dict):
                self.record(SwarmAlert.from_dict(item))
            elif isinstance(item, (list, tuple, set)):
                self.merge(item)
            else:
                raise TypeError(f"Unsupported finding type for AlertLedger.merge: {type(item)!r}")
        return self

    @classmethod
    def merge_all(cls, items: Iterable[Any]) -> AlertLedger:
        ledger = cls()
        ledger.merge(items)
        return ledger

    def alerts(self, only_triggered: bool = False, top_k: int | None = None) -> list[SwarmAlert]:
        vals = [a for a in self._entries.values() if (not only_triggered or a.is_alert)]
        vals.sort(
            key=lambda a: (
                -(a.e_value if math.isfinite(a.e_value) else -1.0),
                -(a.discord_z if math.isfinite(a.discord_z) else -1e9),
                a.channel_id,
                a.epoch,
            )
        )
        if top_k is not None:
            return vals[: max(0, int(top_k))]
        return vals

    def copy(self) -> AlertLedger:
        out = AlertLedger()
        out._entries = dict(self._entries)
        return out

    def to_list(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self.alerts()]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self.alerts())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlertLedger):
            return NotImplemented
        return self._entries == other._entries

    def __repr__(self) -> str:
        return f"AlertLedger(count={len(self._entries)})"


class SwarmMonitor:
    """Multi-channel causal RingMP firehose monitor scaling cleanly to 1,000-4,096+ channels."""

    def __init__(
        self,
        channels: int | Sequence[str],
        m: int = 32,
        window: int = 256,
        excl_zone: int | None = None,
        alpha: float = 0.05,
        epoch_size: int | None = None,
        history_len: int | None = None,
    ) -> None:
        if isinstance(channels, bool):
            raise TypeError("channels must be a positive int or sequence of channel names, not bool")
        if isinstance(channels, int):
            if channels <= 0:
                raise ValueError(f"channels must be > 0, got {channels}")
            self._channel_names: list[str] = [f"ch{i}" for i in range(channels)]
        elif isinstance(channels, (str, bytes)):
            raise TypeError("channels must be a positive int or sequence of channel names, not str/bytes")
        else:
            self._channel_names = [str(c) for c in channels]
            if not self._channel_names:
                raise ValueError("channels sequence must not be empty")
            if len(set(self._channel_names)) != len(self._channel_names):
                raise ValueError("channel names must be unique")

        self._name_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(self._channel_names)
        }
        self._m = int(m)
        self._window = int(window)
        self._excl_zone = excl_zone
        self._alpha = float(alpha)
        self._epoch_size = max(1, int(epoch_size)) if epoch_size is not None else max(1, self._m)
        self._history_len = (
            max(self._m, int(history_len))
            if history_len is not None
            else min(self._window + self._m, 512)
        )

        # Lazily instantiated causal RingMP streams (track_full_mp=False) for instant startup
        # and minimal footprint even at 4,096 channels.
        n_ch = len(self._channel_names)
        self._streams: list[StreamMP | None] = [None] * n_ch
        self._histories: list[deque[float] | None] = [None] * n_ch
        self._last_e: list[float] = [1.0] * n_ch
        self._ticks_seen: int = 0
        self.ledger = AlertLedger()

    @property
    def n_channels(self) -> int:
        return len(self._channel_names)

    @property
    def channel_names(self) -> list[str]:
        return list(self._channel_names)

    @property
    def m(self) -> int:
        return self._m

    @property
    def window(self) -> int:
        return self._window

    def _resolve(self, ch: int | str) -> tuple[int, str]:
        if isinstance(ch, bool):
            raise TypeError("channel reference must be int or str, not bool")
        if isinstance(ch, int):
            idx = ch if ch >= 0 else len(self._channel_names) + ch
            if idx < 0 or idx >= len(self._channel_names):
                raise IndexError(f"channel index {ch} out of range [0, {len(self._channel_names)})")
            return idx, self._channel_names[idx]
        name = str(ch)
        idx = self._name_to_idx.get(name)
        if idx is None:
            raise KeyError(f"Unknown channel {name!r}")
        return idx, name

    def _get_stream(self, idx: int) -> StreamMP:
        s = self._streams[idx]
        if s is None:
            s = StreamMP(
                m=self._m,
                window=self._window,
                excl_zone=self._excl_zone,
                egress=True,
                alpha=self._alpha,
                track_full_mp=False,
            )
            self._streams[idx] = s
        return s

    def _get_history(self, idx: int) -> deque[float]:
        h = self._histories[idx]
        if h is None:
            h = deque(maxlen=self._history_len)
            self._histories[idx] = h
        return h

    def update_channel(self, ch: int | str, x: float) -> TickResult:
        """Ingest one tick for channel `ch`, recording any Ville E-process alert in `self.ledger`."""
        idx, name = self._resolve(ch)
        if isinstance(x, bool):
            raise TypeError("Tick value x must be a real number, not bool")
        val = float(x)
        stream = self._get_stream(idx)
        self._get_history(idx).append(val)
        tick = stream.update(val)
        self._ticks_seen += 1
        self._last_e[idx] = tick.e_value

        if tick.is_alert and tick.gid >= 0:
            epoch = tick.gid // self._epoch_size
            alert = SwarmAlert(
                channel_id=name,
                epoch=epoch,
                gid=tick.gid,
                e_value=tick.e_value,
                discord_z=tick.discord_z,
                left_dist=tick.left_dist,
                left_idx=tick.left_idx,
                p_value=tick.p_value,
                is_alert=True,
                handle=f"{name}@{tick.gid}",
            )
            self.ledger.record(alert)
        return tick

    def init_channel(self, ch: int | str, xs: Iterable[float]) -> None:
        """Warm up a channel's RingMP buffer in one native C++ call without recording warmup alerts."""
        idx, _ = self._resolve(ch)
        vals = list(xs)
        if not vals:
            return
        stream = self._get_stream(idx)
        stream.init(vals)
        hist = self._get_history(idx)
        tail = vals[-self._history_len :]
        hist.extend(float(v) for v in tail)
        self._ticks_seen += len(vals)
        self._last_e[idx] = stream.e_value

    def update_channel_batch(self, ch: int | str, xs: Iterable[float]) -> list[SwarmAlert]:
        """Ingest a batch of ticks for a single channel and return newly triggered alerts."""
        triggered: list[SwarmAlert] = []
        idx, name = self._resolve(ch)
        stream = self._get_stream(idx)
        hist = self._get_history(idx)
        for raw_x in xs:
            if isinstance(raw_x, bool):
                raise TypeError("Tick values must be real numbers, not bool")
            val = float(raw_x)
            hist.append(val)
            tick = stream.update(val)
            self._ticks_seen += 1
            self._last_e[idx] = tick.e_value
            if tick.is_alert and tick.gid >= 0:
                alert = SwarmAlert(
                    channel_id=name,
                    epoch=tick.gid // self._epoch_size,
                    gid=tick.gid,
                    e_value=tick.e_value,
                    discord_z=tick.discord_z,
                    left_dist=tick.left_dist,
                    left_idx=tick.left_idx,
                    p_value=tick.p_value,
                    is_alert=True,
                    handle=f"{name}@{tick.gid}",
                )
                self.ledger.record(alert)
                triggered.append(alert)
        return triggered

    def update_frame(self, values: Sequence[float] | dict[int | str, float]) -> list[TickResult]:
        """Ingest 1 synchronized tick across all (or a named subset of) channels."""
        results: list[TickResult] = []
        if isinstance(values, dict):
            for ch, val in values.items():
                results.append(self.update_channel(ch, val))
            return results

        if len(values) != len(self._channel_names):
            raise ValueError(
                f"update_frame expected {len(self._channel_names)} values, got {len(values)}"
            )
        for idx, val in enumerate(values):
            results.append(self.update_channel(idx, val))
        return results

    def merge_findings(self, findings_list: Iterable[Any]) -> AlertLedger:
        """Commutatively & idempotently merge findings from concurrent agents into `self.ledger`."""
        return self.ledger.merge(findings_list)

    def drilldown(self, handle_or_channel: str | int, length: int | None = None) -> dict[str, Any]:
        """Recover exact trailing raw float samples for a channel or alert handle (`ch@gid`)."""
        ch_ref: str | int = handle_or_channel
        gid_hint: int | None = None
        if (
            isinstance(handle_or_channel, str)
            and handle_or_channel not in self._name_to_idx
            and "@" in handle_or_channel
        ):
            ch_part, gid_part = handle_or_channel.rsplit("@", 1)
            ch_ref = ch_part
            try:
                gid_hint = int(gid_part)
            except ValueError:
                gid_hint = None

        idx, name = self._resolve(ch_ref)
        hist = self._histories[idx]
        vals = list(hist) if hist is not None else []
        if length is not None and length >= 0:
            vals = vals[-int(length) :]
        return {
            "channel_id": name,
            "channel_index": idx,
            "gid_hint": gid_hint,
            "length": len(vals),
            "values": vals,
        }

    def brief(self, max_tokens: int = 300, top_k: int = 10) -> str:
        """Return a strictly token-budgeted alert summary with drill-down handles."""
        if isinstance(max_tokens, bool) or int(max_tokens) < 0:
            raise ValueError(f"max_tokens must be >= 0, got {max_tokens}")
        max_chars = int(max_tokens) * 4
        if max_chars == 0:
            return ""
        ranked = self.ledger.alerts(only_triggered=True)
        alert_channels = {a.channel_id for a in ranked}
        max_e = max((a.e_value for a in ranked), default=max(self._last_e, default=1.0))

        header = (
            f"SWARM ch={self.n_channels} alerts={len(ranked)} "
            f"alert_ch={len(alert_channels)} maxE={max_e:.1f}"
        )
        if not ranked:
            line = f"{header}\nSTATUS ALL_CLEAR ticks={self._ticks_seen}"
            return line[:max_chars]

        lines = [header]
        used_chars = len(header)
        limit = min(len(ranked), max(0, int(top_k)))
        shown = 0

        for a in ranked[:limit]:
            entry = (
                f"  {a.handle} E={a.e_value:.1f} z={a.discord_z:.1f} "
                f"d={a.left_dist:.2f} ep={a.epoch}"
            )
            # Reserve room for a short omission footer if more alerts remain.
            reserve = 22 if (shown + 1 < len(ranked)) else 0
            if used_chars + 1 + len(entry) + reserve > max_chars:
                break
            lines.append(entry)
            used_chars += 1 + len(entry)
            shown += 1

        omitted = len(ranked) - shown
        if omitted > 0:
            footer = f"  ... +{omitted} more"
            if used_chars + 1 + len(footer) <= max_chars:
                lines.append(footer)

        out = "\n".join(lines)
        if len(out) > max_chars:
            out = out[:max_chars]
        return out

    def close(self) -> None:
        for i, s in enumerate(self._streams):
            if s is not None:
                s.close()
                self._streams[i] = None

    def __enter__(self) -> SwarmMonitor:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"SwarmMonitor(channels={self.n_channels}, m={self._m}, "
            f"window={self._window}, alerts={len(self.ledger)})"
        )
