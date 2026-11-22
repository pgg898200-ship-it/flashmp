"""Reversible Agentic Time-Series Digest (95-99.8% token reduction + 100% lossless drill-down)."""

from __future__ import annotations

import array
import json
import math
from dataclasses import dataclass
from typing import Any

from ._native import _check_real_numpy_dtype, _coerce_real_scalar, _get_numpy
from .core import auto_window, discords, find_regimes, floss, motifs, mstump, profile


def _normalize_input(
    T: Any,
    channel_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[array.array], list[str], int, bool]:
    """Normalize 1D, 2D (D, N), or dict[str, seq] input into owned lossless float64 channel arrays."""
    if isinstance(T, (str, bytes, bytearray, bool)):
        raise TypeError(f"Expected numeric time series, got {type(T).__name__}")
    if isinstance(T, dict):
        if not T:
            raise ValueError("Input channel dict must not be empty")
        names = [str(k) for k in T.keys()]
        channels = [_to_double_array(v) for v in T.values()]
        N = len(channels[0])
        for name, ch in zip(names, channels):
            if len(ch) != N:
                raise ValueError(f"Channel {name!r} length {len(ch)} != {N}")
        return channels, names, N, len(channels) > 1

    np = _get_numpy()
    if np is not None and isinstance(T, np.ndarray):
        _check_real_numpy_dtype(T)
        if T.ndim == 1:
            ch = array.array("d", T.astype(np.float64, copy=False).tolist())
            names = list(channel_names) if channel_names else ["s0"]
            return [ch], names[:1], len(ch), False
        if T.ndim == 2:
            D, N = int(T.shape[0]), int(T.shape[1])
            if D < 1:
                raise ValueError("2D input must have at least 1 channel")
            arr = T.astype(np.float64, copy=False)
            channels = [array.array("d", arr[d].tolist()) for d in range(D)]
            names = list(channel_names) if channel_names else [f"ch{d}" for d in range(D)]
            if len(names) < D:
                names.extend(f"ch{d}" for d in range(len(names), D))
            return channels, names[:D], N, D > 1
        raise ValueError(f"Expected 1D or 2D array, got shape {T.shape}")

    # Pure Python sequence: always copy into owned array.array('d') so caller mutations cannot corrupt drilldown
    if isinstance(T, array.array):
        ch = array.array("d", T) if T.typecode == "d" else array.array("d", (float(x) for x in T))
        names = list(channel_names) if channel_names else ["s0"]
        return [ch], names[:1], len(ch), False

    seq = list(T)
    if not seq:
        raise ValueError("Input time series must not be empty")
    first = seq[0]
    if isinstance(first, (list, tuple, array.array)) or (
        np is not None and isinstance(first, np.ndarray)
    ):
        channels = [_to_double_array(row) for row in seq]
        D = len(channels)
        N = len(channels[0])
        for idx, ch in enumerate(channels):
            if len(ch) != N:
                raise ValueError(f"Channel {idx} length {len(ch)} != {N}")
        names = list(channel_names) if channel_names else [f"ch{d}" for d in range(D)]
        if len(names) < D:
            names.extend(f"ch{d}" for d in range(len(names), D))
        return channels, names[:D], N, D > 1

    ch = _to_double_array(seq)
    names = list(channel_names) if channel_names else ["s0"]
    return [ch], names[:1], len(ch), False


def _to_double_array(seq: Any) -> array.array:
    if isinstance(seq, (str, bytes, bytearray, bool)):
        raise TypeError(f"Expected 1D numeric sequence, got {type(seq).__name__}")
    np = _get_numpy()
    if np is not None and isinstance(seq, np.ndarray):
        if seq.ndim != 1:
            raise ValueError(f"Expected 1D channel array, got shape {seq.shape}")
        _check_real_numpy_dtype(seq)
        return array.array("d", seq.astype(np.float64, copy=False).tolist())
    if isinstance(seq, array.array) and seq.typecode == "d":
        return array.array("d", seq)
    if isinstance(seq, (list, tuple)):
        if any(isinstance(x, (bool, complex, str, bytes)) or x is None for x in seq):
            raise TypeError("Time series must contain only real numeric values")
        return array.array("d", seq)
    return array.array("d", (_coerce_real_scalar(x) for x in seq))


def _exact_pearson(arr: array.array, i: int, j: int, m: int) -> float | None:
    """Exact centered Pearson correlation between arr[i:i+m] and arr[j:j+m]."""
    sa = 0.0
    sb = 0.0
    for k in range(m):
        va = arr[i + k]
        vb = arr[j + k]
        if not (math.isfinite(va) and math.isfinite(vb)):
            return None
        sa += va
        sb += vb
    ma = sa / m
    mb = sb / m
    cov = 0.0
    va2 = 0.0
    vb2 = 0.0
    for k in range(m):
        da = arr[i + k] - ma
        db = arr[j + k] - mb
        cov += da * db
        va2 += da * da
        vb2 += db * db
    denom = math.sqrt(va2 * vb2)
    if denom <= 1e-30:
        return 1.0 if (va2 <= 1e-30 and vb2 <= 1e-30) else 0.0
    return max(-1.0, min(1.0, cov / denom))


def _slice_stats(arr: array.array, start: int, end: int) -> dict[str, Any]:
    """Compute exact summary statistics + OLS slope over arr[start:end]."""
    length = max(0, end - start)
    finite_vals: list[float] = []
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0
    min_v = math.inf
    max_v = -math.inf

    for offset in range(length):
        v = arr[start + offset]
        if math.isfinite(v):
            finite_vals.append(v)
            x = float(offset)
            sum_x += x
            sum_y += v
            sum_xx += x * x
            sum_xy += x * v
            if v < min_v:
                min_v = v
            if v > max_v:
                max_v = v

    cnt = len(finite_vals)
    nan_count = length - cnt
    if cnt == 0:
        return {
            "n": length,
            "finite": 0,
            "nan_count": nan_count,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "p05": math.nan,
            "p50": math.nan,
            "p95": math.nan,
            "slope": 0.0,
            "trend": "flat",
        }

    mean = sum_y / cnt
    var = sum((v - mean) * (v - mean) for v in finite_vals) / cnt
    std = math.sqrt(var)

    denom = cnt * sum_xx - sum_x * sum_x
    slope = (cnt * sum_xy - sum_x * sum_y) / denom if abs(denom) > 1e-12 else 0.0

    sv = sorted(finite_vals)

    def _q(p: float) -> float:
        if cnt == 1:
            return sv[0]
        pos = p * (cnt - 1)
        lo = int(math.floor(pos))
        hi = min(cnt - 1, lo + 1)
        frac = pos - lo
        return sv[lo] * (1.0 - frac) + sv[hi] * frac

    drift = slope * max(1, length - 1)
    if std > 1e-12 and abs(drift) >= 0.25 * std:
        trend = "up" if drift > 0 else "down"
    else:
        trend = "flat"

    return {
        "n": length,
        "finite": cnt,
        "nan_count": nan_count,
        "mean": mean,
        "std": std,
        "min": min_v,
        "max": max_v,
        "p05": _q(0.05),
        "p50": _q(0.50),
        "p95": _q(0.95),
        "slope": slope,
        "trend": trend,
    }


def _estimate_raw_tokens(channels: list[array.array]) -> int:
    """Conservative token estimate (4 chars/token) for raw JSON/CSV serialization of T."""
    total_pts = sum(len(ch) for ch in channels)
    if total_pts == 0:
        return 1
    # Exact character count for moderate series; deterministic strided sample for huge series.
    if total_pts <= 20000:
        raw_chars = 2 * len(channels)
        for ch in channels:
            for v in ch:
                raw_chars += len(f"{v:.6g}") + 1
    else:
        sample_chars = 0
        sample_n = 0
        for ch in channels:
            step = max(1, len(ch) // 1000)
            for i in range(0, len(ch), step):
                sample_chars += len(f"{ch[i]:.6g}") + 1
                sample_n += 1
        avg_chars = sample_chars / max(1, sample_n)
        raw_chars = int(round(avg_chars * total_pts)) + 2 * len(channels)
    return max(1, (raw_chars + 3) // 4)


def _estimate_text_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _severity_label(z_score: float) -> str:
    if z_score >= 5.0:
        return "critical"
    if z_score >= 3.5:
        return "high"
    if z_score >= 2.0:
        return "medium"
    return "low"


def _fmt(v: float) -> str:
    if not math.isfinite(v):
        return "nan"
    if abs(v) >= 1e4 or (0.0 < abs(v) < 1e-3):
        return f"{v:.2e}"
    return f"{v:.3g}"


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


@dataclass
class SeriesDigest:
    """Reversible agentic summary of a 1D or 2D time series with 100% lossless drill-down."""

    n_points: int
    n_channels: int
    channel_names: list[str]
    m: int
    mp_len: int
    stats: list[dict[str, Any]]
    regimes: list[dict[str, Any]]
    motifs: list[dict[str, Any]]
    discords: list[dict[str, Any]]
    raw_tokens_est: int
    compact_tokens_est: int
    token_savings_pct: float
    _raw_channels: list[array.array]
    _compact_cache: str = ""

    def _resolve_channel(self, channel: int | str) -> int:
        if isinstance(channel, bool):
            raise TypeError("channel must be an int or str, not bool")
        if isinstance(channel, int):
            idx = channel if channel >= 0 else self.n_channels + channel
            if idx < 0 or idx >= self.n_channels:
                raise IndexError(f"channel {channel} out of range [0, {self.n_channels})")
            return idx
        try:
            return self.channel_names.index(str(channel))
        except ValueError as exc:
            raise KeyError(f"Unknown channel {channel!r}; available: {self.channel_names}") from exc

    def to_compact(self) -> str:
        """Return ultra-terse caveman-style structured digest (~150-350 tokens)."""
        if self._compact_cache:
            return self._compact_cache

        body: list[str] = []

        # Stats (show up to 4 channels inline; summarize rest to stay strictly compact)
        max_stat_ch = min(4, len(self.stats))
        for st in self.stats[:max_stat_ch]:
            body.append(
                f"STAT {st['name']}: mu={_fmt(st['mean'])} sd={_fmt(st['std'])} "
                f"rng=[{_fmt(st['min'])},{_fmt(st['max'])}] p50={_fmt(st['p50'])} "
                f"slp={_fmt(st['slope'])}({st['trend']}) nan={st['nan_count']}"
            )
        if len(self.stats) > max_stat_ch:
            body.append(f"STAT +{len(self.stats) - max_stat_ch} more channels (use drilldown/json)")

        # Regimes
        if self.regimes:
            reg_parts = [
                f"R{i}[{r['start']}:{r['end']}] mu={_fmt(r['mean'])} sd={_fmt(r['std'])} slp={_fmt(r['slope'])}"
                for i, r in enumerate(self.regimes)
            ]
            body.append(f"REGIMES({len(self.regimes)}): " + " | ".join(reg_parts))

        # Motifs
        if self.motifs:
            mot_parts = [
                f"M{i}({mt['index']}~{mt['neighbor']} d={mt['distance']:.2f} r={mt['corr']:.4f})"
                for i, mt in enumerate(self.motifs)
            ]
            body.append(f"MOTIFS({len(self.motifs)}): " + " ".join(mot_parts))

        # Discords
        if self.discords:
            sev_short = {"critical": "CRIT", "high": "HIGH", "medium": "MED", "low": "LOW"}
            dis_parts = [
                f"D{i}@{d['index']}(d={d['distance']:.2f} z={d['z_score']:.1f} "
                f"{sev_short.get(d['severity'], 'LOW')} nn={d['nn_idx']})"
                for i, d in enumerate(self.discords)
            ]
            body.append(f"DISCORDS({len(self.discords)}): " + " ".join(dis_parts))

        body.append("DRILL: drilldown(start,end,ch) | drilldown_discord(rank)")
        body_text = "\n".join(body)

        # Iterate to exact fixed point (at most 3 passes) so header token count matches rendered text exactly.
        tok = self.compact_tokens_est
        pct = self.token_savings_pct
        text = ""
        for _ in range(4):
            header = (
                f"TS_DIGEST n={self.n_points} ch={self.n_channels} m={self.m} "
                f"tok={self.raw_tokens_est}->{tok} (-{pct:.1f}%)"
            )
            text = f"{header}\n{body_text}"
            new_tok = _estimate_text_tokens(text)
            new_pct = max(0.0, 100.0 * (1.0 - new_tok / max(1, self.raw_tokens_est)))
            if new_tok == tok and f"{new_pct:.1f}" == f"{pct:.1f}":
                break
            tok, pct = new_tok, new_pct

        self.compact_tokens_est = tok
        self.token_savings_pct = pct
        self._compact_cache = text
        return self._compact_cache

    def to_dict(self) -> dict[str, Any]:
        self.to_compact()  # ensure exact token estimates are finalized
        return {
            "n_points": self.n_points,
            "n_channels": self.n_channels,
            "channel_names": list(self.channel_names),
            "m": self.m,
            "mp_len": self.mp_len,
            "raw_tokens_est": self.raw_tokens_est,
            "compact_tokens_est": self.compact_tokens_est,
            "token_savings_pct": round(self.token_savings_pct, 2),
            "stats": self.stats,
            "regimes": self.regimes,
            "motifs": self.motifs,
            "discords": self.discords,
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(_sanitize_json(self.to_dict()), indent=indent, allow_nan=False)

    def to_markdown(self) -> str:
        self.to_compact()
        lines = [
            f"### Time-Series Digest (`N={self.n_points}`, `channels={self.n_channels}`, `m={self.m}`)",
            f"**Token Compression:** `{self.raw_tokens_est}` → `{self.compact_tokens_est}` tokens "
            f"(**-{self.token_savings_pct:.1f}%**, 100% lossless drill-down)",
            "",
            "#### Channel Summary",
            "| Channel | Mean | Std | Min | Median | Max | Slope | Trend | NaNs |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
        for st in self.stats:
            lines.append(
                f"| `{st['name']}` | {_fmt(st['mean'])} | {_fmt(st['std'])} | {_fmt(st['min'])} "
                f"| {_fmt(st['p50'])} | {_fmt(st['max'])} | {_fmt(st['slope'])} | {st['trend']} | {st['nan_count']} |"
            )

        if self.regimes:
            lines.extend([
                "",
                "#### Semantic Regimes (FLOSS)",
                "| # | Range | Length | Mean | Std | Slope | Min | Max |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ])
            for i, r in enumerate(self.regimes):
                lines.append(
                    f"| R{i} | `[{r['start']}, {r['end']})` | {r['length']} | {_fmt(r['mean'])} "
                    f"| {_fmt(r['std'])} | {_fmt(r['slope'])} | {_fmt(r['min'])} | {_fmt(r['max'])} |"
                )

        if self.motifs:
            lines.extend([
                "",
                "#### Top Motifs",
                "| # | Index A | Index B | Distance | Correlation |",
                "|---:|---:|---:|---:|---:|",
            ])
            for i, mt in enumerate(self.motifs):
                lines.append(
                    f"| M{i} | {mt['index']} | {mt['neighbor']} | {mt['distance']:.4f} | {mt['corr']:.5f} |"
                )

        if self.discords:
            lines.extend([
                "",
                "#### Top Discords (Anomalies)",
                "| # | Index | Distance | Robust Z | Severity | Nearest Neighbor |",
                "|---:|---:|---:|---:|---|---:|",
            ])
            for i, d in enumerate(self.discords):
                lines.append(
                    f"| D{i} | {d['index']} | {d['distance']:.4f} | {d['z_score']:.2f} "
                    f"| {d['severity'].upper()} | {d['nn_idx']} |"
                )
        return "\n".join(lines)

    def drilldown(self, start: int, end: int, channel: int | str = 0) -> dict[str, Any]:
        """Losslessly recover exact raw float values for `[start, end)` on `channel`."""
        if isinstance(start, bool) or isinstance(end, bool):
            raise TypeError("start and end must be integers, not bool")
        s = int(start)
        e = int(end)
        if s < 0 or e > self.n_points or s > e:
            raise ValueError(
                f"Invalid drilldown range [{s}, {e}) for series of length {self.n_points}"
            )
        ch_idx = self._resolve_channel(channel)
        raw = self._raw_channels[ch_idx]
        vals = [float(raw[i]) for i in range(s, e)]
        st = _slice_stats(raw, s, e)
        return {
            "channel": ch_idx,
            "channel_name": self.channel_names[ch_idx],
            "start": s,
            "end": e,
            "length": e - s,
            "values": vals,
            "mean": st["mean"],
            "std": st["std"],
            "min": st["min"],
            "max": st["max"],
        }

    def drilldown_motif(self, rank: int = 0, channel: int | str = 0) -> dict[str, Any]:
        """Losslessly return the rank-th motif pair `(index, neighbor)` raw float windows."""
        if isinstance(rank, bool):
            raise TypeError("rank must be an integer, not bool")
        if rank < 0 or rank >= len(self.motifs):
            raise IndexError(f"motif rank {rank} out of range [0, {len(self.motifs)})")
        ch_idx = self._resolve_channel(channel)
        raw = self._raw_channels[ch_idx]
        mt = self.motifs[rank]
        a = int(mt["index"])
        b = int(mt["neighbor"])
        a_end = min(self.n_points, a + self.m)
        b_end = min(self.n_points, b + self.m)
        return {
            "rank": rank,
            "channel": ch_idx,
            "channel_name": self.channel_names[ch_idx],
            "motif": dict(mt),
            "start": a,
            "end": a_end,
            "values": [float(raw[i]) for i in range(a, a_end)],
            "partner_start": b,
            "partner_end": b_end,
            "partner_values": [float(raw[i]) for i in range(b, b_end)],
        }

    def drilldown_discord(self, rank: int = 0, channel: int | str = 0) -> dict[str, Any]:
        """Losslessly return the rank-th discord window alongside its nearest normal motif."""
        if isinstance(rank, bool):
            raise TypeError("rank must be an integer, not bool")
        if rank < 0 or rank >= len(self.discords):
            raise IndexError(f"discord rank {rank} out of range [0, {len(self.discords)})")
        ch_idx = self._resolve_channel(channel)
        raw = self._raw_channels[ch_idx]
        d = self.discords[rank]
        pos = int(d["index"])
        end = min(self.n_points, pos + self.m)
        vals = [float(raw[i]) for i in range(pos, end)]

        nn_idx = int(d["nn_idx"])
        nearest_neighbor: dict[str, Any] | None = None
        if 0 <= nn_idx < self.n_points:
            nn_end = min(self.n_points, nn_idx + self.m)
            nearest_neighbor = {
                "start": nn_idx,
                "end": nn_end,
                "values": [float(raw[i]) for i in range(nn_idx, nn_end)],
            }

        nearest_motif: dict[str, Any] | None = None
        if self.motifs:
            mt = self.motifs[0]
            m_pos = int(mt["index"])
            m_end = min(self.n_points, m_pos + self.m)
            nearest_motif = {
                "start": m_pos,
                "end": m_end,
                "partner_start": int(mt["neighbor"]),
                "partner_end": min(self.n_points, int(mt["neighbor"]) + self.m),
                "distance": float(mt["distance"]),
                "corr": float(mt["corr"]),
                "values": [float(raw[i]) for i in range(m_pos, m_end)],
            }

        return {
            "rank": rank,
            "channel": ch_idx,
            "channel_name": self.channel_names[ch_idx],
            "discord": dict(d),
            "window_start": pos,
            "window_end": end,
            "values": vals,
            "nearest_neighbor": nearest_neighbor,
            "nearest_normal_motif": nearest_motif,
        }


def digest(
    T: Any,
    m: int | None = None,
    top_motifs: int = 3,
    top_discords: int = 5,
    max_regimes: int = 6,
    channel_names: list[str] | tuple[str, ...] | None = None,
    threads: int = 0,
) -> SeriesDigest:
    """Build a reversible agentic time-series digest with exact MP, FLOSS, motifs & discords."""
    channels, names, N, is_multi = _normalize_input(T, channel_names=channel_names)
    if N < 6:
        raise ValueError(f"Time series length N={N} must be >= 6 for digest()")

    if m is None:
        max_auto = min(512, max(8, N // 4))
        min_auto = min(8, max(3, N // 8))
        m_int = auto_window(channels[0], min_m=min_auto, max_m=max_auto)
        m_int = max(3, min(m_int, max(3, N // 2)))
    else:
        if isinstance(m, bool):
            raise TypeError("m must be an integer or None, not bool")
        m_int = int(m)
        if m_int < 3:
            raise ValueError(f"Window size m must be >= 3, got {m_int}")
        if m_int > N:
            raise ValueError(f"Window size m={m_int} must be <= series length N={N}")

    if not is_multi:
        mp_res = profile(channels[0], m_int, threads=threads)
        mp_arr = mp_res.mp
        idx_arr = mp_res.idx
        mp_len = mp_res.n
    else:
        P_2d, I_2d = mstump(channels, m_int, threads=threads)
        D = len(channels)
        mp_arr = P_2d[D - 1]
        idx_arr = I_2d[D - 1]
        mp_len = len(mp_arr)

    # Channel summary stats
    ch_stats: list[dict[str, Any]] = []
    for name, ch in zip(names, channels):
        st = _slice_stats(ch, 0, N)
        st["name"] = name
        ch_stats.append(st)

    # FLOSS semantic regimes
    floss_L = m_int if mp_len > 12 * m_int else max(4, mp_len // 12)
    cac = floss(idx_arr, floss_L)
    cps = find_regimes(cac, floss_L, max_regimes=max_regimes, threshold=0.65)
    bounds = [0] + [cp for cp in cps if 0 < cp < N] + [N]
    regime_list: list[dict[str, Any]] = []
    primary = channels[0]
    for r_i in range(len(bounds) - 1):
        rs, re = bounds[r_i], bounds[r_i + 1]
        rst = _slice_stats(primary, rs, re)
        regime_list.append({
            "start": rs,
            "end": re,
            "length": re - rs,
            "mean": rst["mean"],
            "std": rst["std"],
            "slope": rst["slope"],
            "min": rst["min"],
            "max": rst["max"],
            "cac": float(cac[rs]) if (r_i > 0 and rs < mp_len) else None,
        })

    # Top non-overlapping motifs (exact mean per-channel correlation for multivariate inputs)
    raw_motifs = motifs(mp_arr, idx_arr, m_int, k=top_motifs)
    motif_list: list[dict[str, Any]] = []
    for mt in raw_motifs:
        if not is_multi:
            corr = max(-1.0, min(1.0, 1.0 - (mt.distance * mt.distance) / (2.0 * m_int)))
        else:
            corrs = [
                r
                for ch in channels
                if (r := _exact_pearson(ch, mt.index, mt.neighbor, m_int)) is not None
            ]
            corr = (
                sum(corrs) / len(corrs)
                if corrs
                else max(-1.0, min(1.0, 1.0 - (mt.distance * mt.distance) / (2.0 * m_int)))
            )
        motif_list.append({
            "index": mt.index,
            "neighbor": mt.neighbor,
            "distance": mt.distance,
            "corr": corr,
        })

    # Top non-overlapping discords (full-window NMS so each rank is a distinct event)
    raw_discords = discords(mp_arr, m_int, k=top_discords, excl_zone=m_int)
    discord_list: list[dict[str, Any]] = []
    for d in raw_discords:
        nn = int(idx_arr[d.index]) if 0 <= d.index < mp_len else -1
        win_st = _slice_stats(primary, d.index, min(N, d.index + m_int))
        discord_list.append({
            "index": d.index,
            "distance": d.distance,
            "z_score": d.z_score,
            "severity": _severity_label(d.z_score),
            "nn_idx": nn,
            "win_mean": win_st["mean"],
            "win_std": win_st["std"],
            "win_min": win_st["min"],
            "win_max": win_st["max"],
        })

    raw_tok = _estimate_raw_tokens(channels)
    sd = SeriesDigest(
        n_points=N,
        n_channels=len(channels),
        channel_names=names,
        m=m_int,
        mp_len=mp_len,
        stats=ch_stats,
        regimes=regime_list,
        motifs=motif_list,
        discords=discord_list,
        raw_tokens_est=raw_tok,
        compact_tokens_est=1,
        token_savings_pct=0.0,
        _raw_channels=channels,
    )
    sd.to_compact()  # finalize exact compact_tokens_est and token_savings_pct
    return sd
