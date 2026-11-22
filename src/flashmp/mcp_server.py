"""Zero-dependency native stdio JSON-RPC 2.0 MCP Server for flashmp."""

from __future__ import annotations

import csv
import json
import math
import os
import stat
import sys
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, BinaryIO

from ._native import version
from .core import auto_window, discords, find_regimes, floss, profile
from ._digest import SeriesDigest, _sanitize_json, _slice_stats, digest
from .leak import scan_leakage
from .stream import StreamMP
from .swarm import SwarmMonitor

ALLOWED_EXTENSIONS = frozenset({".csv", ".tsv", ".json", ".jsonl", ".txt"})
MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MiB file ceiling
MAX_MSG_BYTES = 16 * 1024 * 1024    # 16 MiB JSON-RPC frame ceiling
MAX_MCP_WINDOW = 262_144
MAX_MCP_SERIES_LEN = 2_000_000
MAX_MCP_CHANNELS = 4096
MAX_SESSIONS = 64
PROTOCOL_VERSION = "2024-11-05"


def _default_allowed_dirs() -> list[Path]:
    env = os.environ.get("FLASHMP_MCP_ALLOWED_DIRS", "").strip()
    if env:
        roots = [Path(p).expanduser().resolve() for p in env.split(os.pathsep) if p.strip()]
        if roots:
            return roots
    return [Path.cwd().resolve()]


def _validate_safe_path(raw_path: str, allowed_dirs: list[Path] | None = None) -> Path:
    """Validate path resolves inside allowed_dirs to an existing regular file with an allowed extension."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("path must be a non-empty string")
    if "\x00" in raw_path:
        raise ValueError("path must not contain null bytes")

    p = Path(raw_path).expanduser()
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Disallowed file extension {p.suffix!r}; allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    resolved = p.resolve(strict=True)
    if resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Resolved path has disallowed extension {resolved.suffix!r}; "
            f"allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    roots = [r.resolve() for r in allowed_dirs] if allowed_dirs is not None else _default_allowed_dirs()
    if not any(resolved.is_relative_to(root) for root in roots):
        raise PermissionError(
            "Access denied: path resolves outside allowed MCP directories "
            "(configure FLASHMP_MCP_ALLOWED_DIRS to permit additional roots)"
        )

    st = os.stat(resolved)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"Path is not a regular file: {raw_path!r}")
    if st.st_size > MAX_FILE_BYTES:
        raise ValueError(
            f"File exceeds maximum allowed size ({MAX_FILE_BYTES // (1024 * 1024)} MiB)"
        )
    return resolved


def _try_float(val: Any) -> float | None:
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def load_series_file(
    path: str,
    column: str | int | None = None,
    allowed_dirs: list[Path] | None = None,
) -> Any:
    """Safely load a 1D or 2D time series from .csv, .tsv, .json, .jsonl, or .txt within allowed_dirs."""
    resolved = _validate_safe_path(path, allowed_dirs=allowed_dirs)
    ext = resolved.suffix.lower()
    text = resolved.read_text(encoding="utf-8", errors="replace")

    if ext == ".json":
        obj = json.loads(text)
        return _extract_json_series(obj, column)

    if ext == ".jsonl":
        vals: list[float] = []
        col_key = str(column) if column is not None else None
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            item = json.loads(s)
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                vals.append(float(item))
            elif isinstance(item, dict):
                if col_key is not None and col_key in item:
                    f = _try_float(item[col_key])
                    if f is not None:
                        vals.append(f)
                else:
                    for v in item.values():
                        f = _try_float(v)
                        if f is not None:
                            vals.append(f)
                            break
            elif isinstance(item, list):
                idx = int(column) if (column is not None and str(column).lstrip("-").isdigit()) else 0
                if -len(item) <= idx < len(item):
                    f = _try_float(item[idx])
                    if f is not None:
                        vals.append(f)
        if not vals:
            raise ValueError(f"No numeric series found in {resolved.name}")
        return vals

    if ext in (".csv", ".tsv"):
        delim = "\t" if ext == ".tsv" else ","
        return _extract_delimited_series(text, delim, column)

    # .txt: whitespace or comma separated numbers (ignoring # comments)
    vals = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        for tok in s.replace(",", " ").split():
            f = _try_float(tok)
            if f is not None:
                vals.append(f)
    if not vals:
        raise ValueError(f"No numeric values parsed from {resolved.name}")
    return vals


def _extract_json_series(obj: Any, column: str | int | None) -> Any:
    if isinstance(obj, dict):
        if column is not None:
            key = str(column)
            if key not in obj:
                raise KeyError(f"Requested column {key!r} not found in JSON object")
            return [float(x) for x in obj[key]]
        # Multi-channel dict of lists if values are numeric lists
        numeric_channels: dict[str, list[float]] = {}
        for k, v in obj.items():
            if isinstance(v, list) and v and all(_try_float(x) is not None for x in v[:10]):
                numeric_channels[str(k)] = [float(x) for x in v]
        if len(numeric_channels) == 1:
            return next(iter(numeric_channels.values()))
        if numeric_channels:
            return numeric_channels
        raise ValueError("JSON object does not contain numeric array fields")

    if isinstance(obj, list):
        if not obj:
            raise ValueError("JSON array is empty")
        if isinstance(obj[0], (int, float)) and not isinstance(obj[0], bool):
            return [float(x) for x in obj]
        if isinstance(obj[0], list):
            if column is not None and str(column).lstrip("-").isdigit():
                idx = int(column)
                return [float(x) for x in obj[idx]]
            return [[float(x) for x in row] for row in obj]
        if isinstance(obj[0], dict):
            col_key = str(column) if column is not None else None
            if col_key is None:
                for k, v in obj[0].items():
                    if _try_float(v) is not None:
                        col_key = k
                        break
            if col_key is None:
                raise ValueError("No numeric column found in JSON records")
            vals: list[float] = []
            for rec in obj:
                f = _try_float(rec.get(col_key))
                if f is not None:
                    vals.append(f)
            return vals
    raise ValueError("Unsupported JSON structure for time series")


def _extract_delimited_series(text: str, delimiter: str, column: str | int | None) -> list[float]:
    rows = [r for r in csv.reader(text.splitlines(), delimiter=delimiter) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("Delimited file is empty")

    first = rows[0]
    has_header = any(_try_float(cell) is None for cell in first)
    header = [c.strip() for c in first] if has_header else []
    data_rows = rows[1:] if has_header else rows
    if not data_rows:
        raise ValueError("Delimited file has no data rows")

    col_idx: int | None = None
    if column is not None:
        if isinstance(column, int) and not isinstance(column, bool):
            col_idx = int(column)
        elif isinstance(column, str) and column.lstrip("-").isdigit():
            col_idx = int(column)
        elif has_header and str(column) in header:
            col_idx = header.index(str(column))
        else:
            raise KeyError(f"Requested column {column!r} not found in delimited header")
    else:
        n_cols = len(data_rows[0])
        numeric_cols: list[int] = []
        sample = data_rows[: min(10, len(data_rows))]
        for c in range(n_cols):
            if all(c < len(r) and _try_float(r[c]) is not None for r in sample):
                numeric_cols.append(c)
        if not numeric_cols:
            raise ValueError("No numeric column found in delimited file")
        col_idx = numeric_cols[0]
        if (
            has_header
            and len(numeric_cols) >= 2
            and col_idx < len(header)
            and header[col_idx].lower() in {"t", "time", "timestamp", "ts", "date", "index", "idx", "step"}
        ):
            col_idx = numeric_cols[1]

    vals: list[float] = []
    for r in data_rows:
        if -len(r) <= col_idx < len(r):
            f = _try_float(r[col_idx])
            if f is not None:
                vals.append(f)
    if not vals:
        raise ValueError(f"Column {col_idx} contained no numeric values")
    return vals


class _SessionStore:
    """Bounded LRU handle store (max 64 sessions) with safe native resource cleanup."""

    def __init__(self, max_size: int = MAX_SESSIONS) -> None:
        self._max_size = max(1, int(max_size))
        self._store: OrderedDict[str, tuple[str, Any]] = OrderedDict()

    def put(self, kind: str, obj: Any) -> str:
        prefix = {"digest": "dig", "stream": "str", "swarm": "swm"}.get(kind, "obj")
        handle = f"{prefix}_{uuid.uuid4().hex[:12]}"
        self._store[handle] = (kind, obj)
        self._store.move_to_end(handle)
        while len(self._store) > self._max_size:
            _, (evicted_kind, evicted_obj) = self._store.popitem(last=False)
            if evicted_kind in ("stream", "swarm") and hasattr(evicted_obj, "close"):
                try:
                    evicted_obj.close()
                except Exception:
                    pass
        return handle

    def get(self, handle: str, expected_kind: str | tuple[str, ...] | None = None) -> tuple[str, Any]:
        entry = self._store.get(handle)
        if entry is None:
            raise KeyError(f"Unknown or evicted session handle {handle!r}")
        self._store.move_to_end(handle)
        kind, obj = entry
        if expected_kind is not None:
            allowed = (expected_kind,) if isinstance(expected_kind, str) else expected_kind
            if kind not in allowed:
                raise TypeError(f"Handle {handle!r} is a {kind!r}, expected {allowed}")
        return kind, obj

    def close_all(self) -> None:
        while self._store:
            _, (kind, obj) = self._store.popitem(last=False)
            if kind in ("stream", "swarm") and hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self._store)


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "ts_digest",
        "description": (
            "Compute an exact reversible Matrix Profile + FLOSS digest of a 1D or multivariate "
            "time series (95-99.8% token reduction) and return a handle for 100% lossless drill-down."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "values": {
                    "description": "1D array of floats, 2D (D x N) array, or object mapping channel names to float arrays."
                },
                "path": {"type": "string", "description": "Safe path to .csv/.tsv/.json/.jsonl/.txt file."},
                "column": {"description": "Optional column name or index when reading a file."},
                "m": {"type": "integer", "minimum": 3, "description": "Subsequence window size (auto-detected if omitted)."},
                "top_motifs": {"type": "integer", "minimum": 0, "default": 3},
                "top_discords": {"type": "integer", "minimum": 0, "default": 5},
                "max_regimes": {"type": "integer", "minimum": 1, "default": 6},
                "format": {"type": "string", "enum": ["compact", "json", "markdown"], "default": "compact"},
            },
        },
    },
    {
        "name": "ts_drilldown",
        "description": (
            "Losslessly recover exact raw float values from a previously created ts_digest or "
            "ts_swarm_sift handle by index range [start, end) or discord_rank."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["handle"],
            "properties": {
                "handle": {"type": "string", "description": "Session handle returned by ts_digest or ts_swarm_sift."},
                "start": {"type": "integer", "minimum": 0},
                "end": {"type": "integer", "minimum": 0},
                "discord_rank": {"type": "integer", "minimum": 0, "description": "If set, returns exact anomalous window + nearest motif."},
                "channel": {"description": "Channel index or name (default 0)."},
                "length": {"type": "integer", "minimum": 1, "description": "Trailing sample count for swarm handles."},
            },
        },
    },
    {
        "name": "ts_anomalies",
        "description": "Run exact SIMD Matrix Profile discord detection on a time series.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "values": {"type": "array", "items": {"type": "number"}},
                "path": {"type": "string"},
                "column": {},
                "m": {"type": "integer", "minimum": 3},
                "k": {"type": "integer", "minimum": 1, "default": 5},
                "excl_zone": {"type": "integer", "minimum": 0},
            },
        },
    },
    {
        "name": "ts_regimes",
        "description": "Detect semantic regime changepoints via exact Matrix Profile + O(n) FLOSS.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "values": {"type": "array", "items": {"type": "number"}},
                "path": {"type": "string"},
                "column": {},
                "m": {"type": "integer", "minimum": 3},
                "L": {"type": "integer", "minimum": 1},
                "max_regimes": {"type": "integer", "minimum": 1, "default": 6},
                "threshold": {"type": "number", "default": 0.65},
            },
        },
    },
    {
        "name": "ts_leak_scan",
        "description": (
            "Scan a query (test/eval) series against a reference (train) series for verbatim, "
            "affine-rescaled/shifted, or sign-flipped subsequence leakage."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ref_values": {"type": "array", "items": {"type": "number"}},
                "ref_path": {"type": "string"},
                "ref_column": {},
                "qry_values": {"type": "array", "items": {"type": "number"}},
                "qry_path": {"type": "string"},
                "qry_column": {},
                "m": {"type": "integer", "minimum": 3, "default": 64},
                "corr_thresh": {"type": "number", "default": 0.999},
                "check_neg": {"type": "boolean", "default": True},
                "max_examples": {"type": "integer", "minimum": 0, "default": 20},
            },
        },
    },
    {
        "name": "ts_stream_create",
        "description": (
            "Create a stateful bounded-window RingMP streaming monitor with anytime-valid "
            "Ville E-process anomaly alerting and return a session handle."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["m"],
            "properties": {
                "m": {"type": "integer", "minimum": 3},
                "window": {"type": "integer", "minimum": 4, "default": 4096},
                "excl_zone": {"type": "integer", "minimum": 0},
                "egress": {"type": "boolean", "default": True},
                "alpha": {"type": "number", "default": 0.05},
                "track_full_mp": {"type": "boolean", "default": False},
                "initial_values": {"type": "array", "items": {"type": "number"}},
                "initial_path": {"type": "string"},
                "initial_column": {},
            },
        },
    },
    {
        "name": "ts_stream_push",
        "description": "Push one or more new ticks into an existing RingMP stream handle.",
        "inputSchema": {
            "type": "object",
            "required": ["handle", "values"],
            "properties": {
                "handle": {"type": "string"},
                "values": {
                    "description": "A single float tick or list of float ticks to ingest."
                },
            },
        },
    },
    {
        "name": "ts_swarm_sift",
        "description": (
            "Sift a multi-channel firehose across many channels using causal RingMP and a "
            "commutative CRDT AlertLedger, returning a strictly token-budgeted brief + handle."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["channels"],
            "properties": {
                "channels": {
                    "description": "Mapping of {channel_name: [floats]} or 2D list of channel series."
                },
                "m": {"type": "integer", "minimum": 3, "default": 32},
                "window": {"type": "integer", "minimum": 4, "default": 256},
                "alpha": {"type": "number", "default": 0.05},
                "max_tokens": {"type": "integer", "minimum": 32, "default": 300},
                "top_k": {"type": "integer", "minimum": 1, "default": 10},
            },
        },
    },
]


class MCPServer:
    """Native zero-dependency JSON-RPC 2.0 MCP server exposing flashmp tools."""

    def __init__(
        self,
        max_sessions: int = MAX_SESSIONS,
        allowed_dirs: list[str | Path] | None = None,
    ) -> None:
        self.sessions = _SessionStore(max_size=max_sessions)
        self.allowed_dirs: list[Path] | None = (
            [Path(d).expanduser().resolve() for d in allowed_dirs] if allowed_dirs is not None else None
        )

    def _resolve_series(
        self,
        args: dict[str, Any],
        val_key: str = "values",
        path_key: str = "path",
        col_key: str = "column",
    ) -> Any:
        if val_key in args and args[val_key] is not None:
            data = args[val_key]
            if isinstance(data, list) and len(data) > MAX_MCP_SERIES_LEN:
                raise ValueError(f"Series length {len(data)} exceeds MCP ceiling ({MAX_MCP_SERIES_LEN})")
            return data
        if path_key in args and args[path_key] is not None:
            return load_series_file(
                args[path_key],
                column=args.get(col_key),
                allowed_dirs=self.allowed_dirs,
            )
        raise ValueError(f"Must supply either {val_key!r} or {path_key!r}")

    def _call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "ts_digest":
            series = self._resolve_series(args)
            sd = digest(
                series,
                m=args.get("m"),
                top_motifs=int(args.get("top_motifs", 3)),
                top_discords=int(args.get("top_discords", 5)),
                max_regimes=int(args.get("max_regimes", 6)),
            )
            handle = self.sessions.put("digest", sd)
            fmt = str(args.get("format", "compact")).lower()
            if fmt == "markdown":
                rendered = sd.to_markdown()
            elif fmt == "json":
                rendered = sd.to_json()
            else:
                rendered = sd.to_compact()
            return {
                "handle": handle,
                "format": fmt,
                "n_points": sd.n_points,
                "n_channels": sd.n_channels,
                "m": sd.m,
                "raw_tokens_est": sd.raw_tokens_est,
                "compact_tokens_est": sd.compact_tokens_est,
                "token_savings_pct": round(sd.token_savings_pct, 2),
                "digest": rendered,
            }

        if name == "ts_drilldown":
            handle = str(args["handle"])
            kind, obj = self.sessions.get(handle, expected_kind=("digest", "swarm"))
            if kind == "swarm":
                mon: SwarmMonitor = obj
                ch = args.get("channel", 0)
                length = args.get("length")
                return mon.drilldown(ch, length=int(length) if length is not None else None)
            sd_obj: SeriesDigest = obj
            ch = args.get("channel", 0)
            if args.get("discord_rank") is not None:
                return sd_obj.drilldown_discord(rank=int(args["discord_rank"]), channel=ch)
            if args.get("start") is None or args.get("end") is None:
                raise ValueError("ts_drilldown requires either 'discord_rank' or both 'start' and 'end'")
            return sd_obj.drilldown(int(args["start"]), int(args["end"]), channel=ch)

        if name == "ts_anomalies":
            series = self._resolve_series(args)
            if not isinstance(series, list) or not series:
                raise ValueError("ts_anomalies requires a non-empty 1D numeric series")
            m = args.get("m")
            m_int = int(m) if m is not None else auto_window(series, min_m=min(8, max(3, len(series) // 2)), max_m=min(512, max(3, len(series))))
            res = profile(series, m_int, excl_zone=args.get("excl_zone"))
            k = int(args.get("k", 5))
            found = res.discords(k=k)
            return {
                "m": m_int,
                "n": res.n,
                "anomalies": [
                    {
                        "rank": rank,
                        "index": d.index,
                        "distance": d.distance,
                        "z_score": d.z_score,
                        "nearest_neighbor": int(res.idx[d.index]),
                    }
                    for rank, d in enumerate(found)
                ],
            }

        if name == "ts_regimes":
            series = self._resolve_series(args)
            if not isinstance(series, list) or not series:
                raise ValueError("ts_regimes requires a non-empty 1D numeric series")
            m = args.get("m")
            m_int = int(m) if m is not None else auto_window(series, min_m=min(8, max(3, len(series) // 2)), max_m=min(512, max(3, len(series))))
            res = profile(series, m_int)
            L_arg = args.get("L")
            L_int = int(L_arg) if L_arg is not None else (m_int if res.n > 12 * m_int else max(4, res.n // 12))
            cac = floss(res.idx, L_int)
            cps = find_regimes(
                cac,
                L_int,
                max_regimes=int(args.get("max_regimes", 6)),
                threshold=float(args.get("threshold", 0.65)),
            )
            import array as _arr

            raw = _arr.array("d", (float(x) for x in series))
            bounds = [0] + [c for c in cps if 0 < c < len(raw)] + [len(raw)]
            regimes = []
            for i in range(len(bounds) - 1):
                s, e = bounds[i], bounds[i + 1]
                st = _slice_stats(raw, s, e)
                regimes.append({
                    "start": s,
                    "end": e,
                    "length": e - s,
                    "mean": st["mean"],
                    "std": st["std"],
                    "slope": st["slope"],
                    "min": st["min"],
                    "max": st["max"],
                })
            return {"m": m_int, "L": L_int, "changepoints": cps, "regimes": regimes}

        if name == "ts_leak_scan":
            ref = self._resolve_series(args, "ref_values", "ref_path", "ref_column")
            qry = self._resolve_series(args, "qry_values", "qry_path", "qry_column")
            rep = scan_leakage(
                ref,
                qry,
                m=int(args.get("m", 64)),
                corr_thresh=float(args.get("corr_thresh", 0.999)),
                check_neg=bool(args.get("check_neg", True)),
                max_examples=int(args.get("max_examples", 20)),
            )
            out = rep.to_dict()
            out["summary"] = rep.summary()
            return out

        if name == "ts_stream_create":
            m_int = int(args["m"])
            win_int = int(args.get("window", 4096))
            if win_int > MAX_MCP_WINDOW:
                raise ValueError(f"window={win_int} exceeds MCP ceiling ({MAX_MCP_WINDOW})")
            if m_int > MAX_MCP_WINDOW:
                raise ValueError(f"m={m_int} exceeds MCP ceiling ({MAX_MCP_WINDOW})")
            stream = StreamMP(
                m=m_int,
                window=win_int,
                excl_zone=args.get("excl_zone"),
                egress=bool(args.get("egress", True)),
                alpha=float(args.get("alpha", 0.05)),
                track_full_mp=bool(args.get("track_full_mp", False)),
            )
            if args.get("initial_values") is not None or args.get("initial_path") is not None:
                init_data = self._resolve_series(args, "initial_values", "initial_path", "initial_column")
                stream.init(init_data)
            handle = self.sessions.put("stream", stream)
            return {
                "handle": handle,
                "m": stream.m,
                "window": stream.window,
                "active_len": stream.active_len,
                "e_value": stream.e_value,
                "is_alert": stream.is_alert,
            }

        if name == "ts_stream_push":
            handle = str(args["handle"])
            _, stream = self.sessions.get(handle, expected_kind="stream")
            raw_vals = args["values"]
            if isinstance(raw_vals, bool):
                raise ValueError("'values' must be a number or list of numbers, not bool")
            if isinstance(raw_vals, (int, float)):
                tick = stream.update(float(raw_vals))
                return {
                    "handle": handle,
                    "count": 1,
                    "active_len": stream.active_len,
                    "latest": tick.to_dict(),
                    "alert_count": 1 if tick.is_alert else 0,
                }
            if not isinstance(raw_vals, list):
                raise ValueError("'values' must be a number or list of numbers")
            if len(raw_vals) > MAX_MCP_SERIES_LEN:
                raise ValueError(f"Push batch length {len(raw_vals)} exceeds MCP ceiling ({MAX_MCP_SERIES_LEN})")
            alert_gids: list[int] = []
            max_e = stream.e_value
            last_tick_dict: dict[str, Any] | None = None
            for v in raw_vals:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError("All elements in 'values' must be real numbers")
                t = stream.update(float(v))
                if t.e_value > max_e:
                    max_e = t.e_value
                if t.is_alert:
                    alert_gids.append(t.gid)
                last_tick_dict = t.to_dict()
            return {
                "handle": handle,
                "count": len(raw_vals),
                "active_len": stream.active_len,
                "e_value": stream.e_value,
                "max_e_value": max_e,
                "is_alert": stream.is_alert,
                "alert_count": len(alert_gids),
                "alert_gids": alert_gids[:20],
                "latest": last_tick_dict,
            }

        if name == "ts_swarm_sift":
            raw_ch = args["channels"]
            m_int = int(args.get("m", 32))
            window = int(args.get("window", 256))
            if window > MAX_MCP_WINDOW:
                raise ValueError(f"window={window} exceeds MCP ceiling ({MAX_MCP_WINDOW})")
            alpha = float(args.get("alpha", 0.05))
            max_tokens = int(args.get("max_tokens", 300))
            top_k = int(args.get("top_k", 10))

            if isinstance(raw_ch, dict):
                if len(raw_ch) > MAX_MCP_CHANNELS:
                    raise ValueError(f"Channel count {len(raw_ch)} exceeds MCP ceiling ({MAX_MCP_CHANNELS})")
                names = [str(k) for k in raw_ch.keys()]
                mon = SwarmMonitor(names, m=m_int, window=window, alpha=alpha)
                for k, seq in raw_ch.items():
                    mon.update_channel_batch(str(k), seq)
            elif isinstance(raw_ch, list):
                if len(raw_ch) > MAX_MCP_CHANNELS:
                    raise ValueError(f"Channel count {len(raw_ch)} exceeds MCP ceiling ({MAX_MCP_CHANNELS})")
                mon = SwarmMonitor(len(raw_ch), m=m_int, window=window, alpha=alpha)
                for i, seq in enumerate(raw_ch):
                    mon.update_channel_batch(i, seq)
            else:
                raise ValueError("'channels' must be a dict of {name: [floats]} or 2D list")

            handle = self.sessions.put("swarm", mon)
            brief_text = mon.brief(max_tokens=max_tokens, top_k=top_k)
            return {
                "handle": handle,
                "n_channels": mon.n_channels,
                "alert_count": len(mon.ledger),
                "brief": brief_text,
                "alerts": [a.to_dict() for a in mon.ledger.alerts(only_triggered=True, top_k=top_k)],
            }

        raise ValueError(f"Unknown tool: {name!r}")

    def handle_message(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """Process one JSON-RPC 2.0 request/notification dict and return response dict (or None)."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": msg.get("id") if isinstance(msg, dict) else None,
                "error": {"code": -32600, "message": "Invalid Request"},
            }

        method = msg.get("method")
        has_id = "id" in msg
        req_id = msg.get("id")
        if not isinstance(method, str) or not method:
            return {
                "jsonrpc": "2.0",
                "id": req_id if has_id else None,
                "error": {"code": -32600, "message": "Invalid Request: 'method' must be a non-empty string"},
            }
        params = msg.get("params") or {}

        # Notifications carry no id and expect no response.
        if not has_id or method.startswith("notifications/"):
            return None

        if method == "initialize":
            client_proto = params.get("protocolVersion") if isinstance(params, dict) else None
            proto = str(client_proto) if client_proto else PROTOCOL_VERSION
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": proto,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "flashmp", "version": version()},
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOL_DEFINITIONS}}

        if method == "tools/call":
            if not isinstance(params, dict) or "name" not in params:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": "Missing tool 'name' in params"},
                }
            tool_name = str(params["name"])
            tool_args = params.get("arguments") or {}
            if not isinstance(tool_args, dict):
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": "'arguments' must be an object"},
                }
            try:
                payload = self._call_tool(tool_name, tool_args)
                text = json.dumps(_sanitize_json(payload), allow_nan=False)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": text}],
                        "isError": False,
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method!r}"},
        }

    def serve(self, in_stream: BinaryIO | None = None, out_stream: BinaryIO | None = None) -> None:
        """Serve JSON-RPC 2.0 over stdio supporting newline-delimited & Content-Length/LSP framing."""
        inp = in_stream if in_stream is not None else sys.stdin.buffer
        out = out_stream if out_stream is not None else sys.stdout.buffer

        try:
            while True:
                raw_line = inp.readline(MAX_MSG_BYTES + 1)
                if not raw_line:
                    break
                if len(raw_line) > MAX_MSG_BYTES:
                    self._write_response(
                        out,
                        {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Frame exceeds maximum size"}},
                        framed=False,
                    )
                    break
                stripped = raw_line.strip()
                if not stripped:
                    continue

                use_content_length = False
                # Detect HTTP/LSP header framing (supports Content-Length or Content-Type in any order).
                if not stripped.startswith((b"{", b"[")) and b":" in stripped:
                    use_content_length = True
                    headers: dict[bytes, bytes] = {}
                    hdr_line = raw_line
                    while hdr_line and hdr_line not in (b"\r\n", b"\n"):
                        if b":" in hdr_line:
                            k, v = hdr_line.split(b":", 1)
                            headers[k.strip().lower()] = v.strip()
                        hdr_line = inp.readline(8192)
                    if b"content-length" not in headers:
                        self._write_response(
                            out,
                            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Missing Content-Length header"}},
                            framed=True,
                        )
                        continue
                    try:
                        length = int(headers[b"content-length"])
                    except ValueError:
                        self._write_response(
                            out,
                            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Invalid Content-Length"}},
                            framed=True,
                        )
                        continue
                    if length <= 0 or length > MAX_MSG_BYTES:
                        self._write_response(
                            out,
                            {
                                "jsonrpc": "2.0",
                                "id": None,
                                "error": {
                                    "code": -32600,
                                    "message": f"Content-Length {length} out of valid bounds (1..{MAX_MSG_BYTES})",
                                },
                            },
                            framed=True,
                        )
                        continue
                    body_bytes = inp.read(length)
                    if not body_bytes:
                        break
                    raw_text = body_bytes.decode("utf-8", errors="replace")
                else:
                    raw_text = stripped.decode("utf-8", errors="replace")

                try:
                    msg = json.loads(raw_text)
                except json.JSONDecodeError as exc:
                    self._write_response(
                        out,
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": f"Parse error: {exc}"},
                        },
                        framed=use_content_length,
                    )
                    continue

                resp = self.handle_message(msg)
                if resp is not None:
                    self._write_response(out, resp, framed=use_content_length)
        finally:
            self.sessions.close_all()

    @staticmethod
    def _write_response(out: BinaryIO, resp: dict[str, Any], framed: bool) -> None:
        payload = json.dumps(_sanitize_json(resp), separators=(",", ":"), allow_nan=False).encode("utf-8")
        if framed:
            header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
            out.write(header + payload)
        else:
            out.write(payload + b"\n")
        out.flush()


def main() -> int:
    server = MCPServer()
    server.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
