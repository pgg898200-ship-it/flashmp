"""Fast zero-dependency CLI for flashmp (<20ms startup)."""

from __future__ import annotations

import argparse
import sys
from typing import Any


def _parse_text_series(text: str, col: str | None = None, ext: str = ".txt") -> Any:
    import csv
    import json

    def _to_float(val: Any) -> float | None:
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

    if ext == ".json":
        obj = json.loads(text)
        if isinstance(obj, dict):
            if col is not None:
                return [float(x) for x in obj[col]]
            ch = {str(k): [float(x) for x in v] for k, v in obj.items() if isinstance(v, list) and v}
            return next(iter(ch.values())) if len(ch) == 1 else ch
        if isinstance(obj, list) and obj:
            if isinstance(obj[0], (int, float)):
                return [float(x) for x in obj]
            if isinstance(obj[0], list):
                if col is not None and col.lstrip("-").isdigit():
                    return [float(x) for x in obj[int(col)]]
                return [[float(x) for x in row] for row in obj]
            if isinstance(obj[0], dict):
                key = col
                if key is None:
                    for k, v in obj[0].items():
                        if _to_float(v) is not None:
                            key = k
                            break
                return [float(r[key]) for r in obj if key in r and _to_float(r[key]) is not None]
        raise ValueError("Unsupported JSON time-series structure")

    if ext == ".jsonl":
        vals: list[float] = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            item = json.loads(s)
            if isinstance(item, (int, float)):
                vals.append(float(item))
            elif isinstance(item, dict):
                if col is not None and col in item:
                    f = _to_float(item[col])
                    if f is not None:
                        vals.append(f)
                else:
                    for v in item.values():
                        f = _to_float(v)
                        if f is not None:
                            vals.append(f)
                            break
        return vals

    if ext in (".csv", ".tsv") or ("," in text.splitlines()[0] if text.splitlines() else False):
        delim = "\t" if ext == ".tsv" else ","
        rows = [r for r in csv.reader(text.splitlines(), delimiter=delim) if any(c.strip() for c in r)]
        if not rows:
            raise ValueError("Empty input")
        first = rows[0]
        has_header = any(_to_float(c) is None for c in first)
        header = [c.strip() for c in first] if has_header else []
        data_rows = rows[1:] if has_header else rows
        if not data_rows:
            raise ValueError("No data rows found")
        if col is not None:
            if col.lstrip("-").isdigit():
                cidx = int(col)
            elif has_header and col in header:
                cidx = header.index(col)
            else:
                raise KeyError(f"Column {col!r} not found in header {header}")
        else:
            sample = data_rows[: min(10, len(data_rows))]
            num_cols = [
                c
                for c in range(len(data_rows[0]))
                if all(c < len(r) and _to_float(r[c]) is not None for r in sample)
            ]
            if not num_cols:
                raise ValueError("No numeric column found")
            cidx = num_cols[0]
            if (
                has_header
                and len(num_cols) >= 2
                and cidx < len(header)
                and header[cidx].lower() in {"t", "time", "timestamp", "ts", "date", "index", "idx", "step"}
            ):
                cidx = num_cols[1]
        vals = []
        for r in data_rows:
            if -len(r) <= cidx < len(r):
                f = _to_float(r[cidx])
                if f is not None:
                    vals.append(f)
        return vals

    vals = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        for tok in s.replace(",", " ").split():
            f = _to_float(tok)
            if f is not None:
                vals.append(f)
    if not vals:
        raise ValueError("No numeric values found in input")
    return vals


def _load_source(source: str, col: str | None = None) -> Any:
    if source == "-":
        text = sys.stdin.read()
        stripped = text.lstrip()
        ext = ".json" if stripped.startswith(("[", "{")) else ".txt"
        return _parse_text_series(text, col=col, ext=ext)

    from pathlib import Path

    p = Path(source)
    if not p.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")
    text = p.read_text(encoding="utf-8", errors="replace")
    return _parse_text_series(text, col=col, ext=p.suffix.lower() or ".txt")


def _cmd_digest(args: argparse.Namespace) -> int:
    from ._digest import digest

    data = _load_source(args.source, col=args.col)
    sd = digest(data, m=args.m)
    if args.format == "json":
        print(sd.to_json(indent=2))
    elif args.format in ("md", "markdown"):
        print(sd.to_markdown())
    else:
        print(sd.to_compact())
    return 0


def _cmd_mp(args: argparse.Namespace) -> int:
    import math
    from .core import profile

    data = _load_source(args.source, col=args.col)
    res = profile(data, args.m, threads=args.threads)

    if args.out is not None:
        header = "index,mp,neighbor,left_mp,left_neighbor,right_mp,right_neighbor\n"
        if args.out == "-":
            out = sys.stdout
            out.write(header)
            for i in range(res.n):
                out.write(
                    f"{i},{res.mp[i]:.10g},{res.idx[i]},"
                    f"{res.left_mp[i]:.10g},{res.left_idx[i]},"
                    f"{res.right_mp[i]:.10g},{res.right_idx[i]}\n"
                )
        else:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(header)
                for i in range(res.n):
                    fh.write(
                        f"{i},{res.mp[i]:.10g},{res.idx[i]},"
                        f"{res.left_mp[i]:.10g},{res.left_idx[i]},"
                        f"{res.right_mp[i]:.10g},{res.right_idx[i]}\n"
                    )
            print(f"Wrote {res.n} Matrix Profile rows (m={res.m}) to {args.out}")
        return 0

    finite = [float(x) for x in res.mp if math.isfinite(float(x))]
    mp_min = min(finite) if finite else math.nan
    mp_max = max(finite) if finite else math.nan
    mots = res.motifs(k=3)
    discs = res.discords(k=3)
    print(f"MP n={res.n} m={res.m} excl_zone={res.excl_zone} min={mp_min:.4f} max={mp_max:.4f}")
    if mots:
        print("MOTIFS:", " ".join(f"({mt.index}~{mt.neighbor} d={mt.distance:.4f})" for mt in mots))
    if discs:
        print("DISCORDS:", " ".join(f"@{d.index}(d={d.distance:.4f} z={d.z_score:.2f})" for d in discs))
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    import json
    from ._digest import _sanitize_json
    from .stream import StreamMP

    total = 0
    alerts = 0
    with StreamMP(
        m=args.m,
        window=args.window,
        egress=True,
        alpha=args.alpha,
        track_full_mp=not args.causal,
    ) as stream:
        for line in sys.stdin:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for tok in s.replace(",", " ").split():
                try:
                    val = float(tok)
                except ValueError:
                    continue
                tick = stream.update(val)
                total += 1
                if tick.is_alert:
                    alerts += 1
                    if args.json:
                        print(json.dumps(_sanitize_json(tick.to_dict()), allow_nan=False), flush=True)
                    else:
                        print(
                            f"ALERT gid={tick.gid} E={tick.e_value:.2f} z={tick.discord_z:.2f} "
                            f"d={tick.left_dist:.4f} nn={tick.left_idx} p={tick.p_value:.4f}",
                            flush=True,
                        )
    print(f"WATCH_DONE ticks={total} alerts={alerts}", file=sys.stderr)
    return 0


def _cmd_leak_scan(args: argparse.Namespace) -> int:
    import json
    from ._digest import _sanitize_json
    from .leak import scan_leakage

    ref = _load_source(args.ref_file, col=args.col)
    qry = _load_source(args.qry_file, col=args.col)
    rep = scan_leakage(
        ref,
        qry,
        m=args.m,
        corr_thresh=args.thresh,
        check_neg=not args.no_neg,
    )
    if args.json:
        print(json.dumps(_sanitize_json(rep.to_dict()), indent=2, allow_nan=False))
    else:
        print(rep.summary())
    return 0


def _cmd_bench(_args: argparse.Namespace) -> int:
    import array
    import math
    import time
    from ._digest import digest
    from ._native import simd_isa, version
    from .core import stump
    from .stream import StreamMP

    print(f"flashmp v{version()} | SIMD ISA: {simd_isa()}")

    # Synthetic benchmark data (pure stdlib, zero numpy dependency)
    N = 10000
    m = 64
    T = array.array("d", (math.sin(0.08 * i) + 0.35 * math.cos(0.21 * i) for i in range(N)))

    t0 = time.perf_counter()
    mp = stump(T, m)
    dt_batch = time.perf_counter() - t0
    n_sub = len(mp)
    pairs = n_sub * (n_sub - 1) // 2
    print(
        f"  Batch stump (N={N}, m={m}): {dt_batch * 1000:.2f} ms "
        f"({pairs / max(1e-9, dt_batch) / 1e6:.1f}M pairs/s)"
    )

    # Streaming benchmark
    n_ticks = 3000
    with StreamMP(m=m, window=1024, track_full_mp=False) as s:
        s.init(T[:1024])
        t0 = time.perf_counter()
        s.update_batch(T[1024 : 1024 + n_ticks])
        dt_stream = time.perf_counter() - t0
    print(
        f"  RingMP causal (W=1024, m={m}, {n_ticks} ticks): {dt_stream * 1000:.2f} ms "
        f"({n_ticks / max(1e-9, dt_stream):,.0f} ticks/s, {dt_stream * 1e6 / n_ticks:.1f} us/tick)"
    )

    # Digest benchmark
    t0 = time.perf_counter()
    sd = digest(T, m=m)
    dt_dig = time.perf_counter() - t0
    print(
        f"  Agentic digest (N={N}, m={m}): {dt_dig * 1000:.2f} ms | "
        f"tokens {sd.raw_tokens_est} -> {sd.compact_tokens_est} (-{sd.token_savings_pct:.1f}%)"
    )
    return 0


def _cmd_demo(_args: argparse.Namespace) -> int:
    import array
    import math
    import time
    from ._digest import digest
    from ._native import simd_isa, version
    from .leak import scan_leakage
    from .stream import StreamMP

    t_start = time.perf_counter()
    N = 3000
    m = 48

    # Deterministic synthetic multi-regime signal:
    # Regime 0 [0..1500): slow sine + harmonic with planted exact motif at 200 & 900
    # Regime 1 [1500..3000): faster shifted oscillation with planted discord spike at 2250
    T = array.array("d")
    for i in range(N):
        noise = 0.08 * math.sin(1.7 * i + 0.3 * math.cos(0.41 * i))
        if i < 1500:
            v = math.sin(2.0 * math.pi * i / 48.0) + 0.3 * math.sin(4.0 * math.pi * i / 48.0) + noise
        else:
            v = 2.5 + 0.9 * math.cos(2.0 * math.pi * i / 20.0) + noise
        T.append(v)

    # Plant exact recurring motif at 200 and 900
    for k in range(m):
        shape = 1.4 * math.sin(2.0 * math.pi * k / m) * math.exp(-((k - m / 2.0) ** 2) / 140.0)
        T[200 + k] = shape
        T[900 + k] = shape

    # Plant sharp anomalous discord at 2250
    for k in range(m):
        T[2250 + k] = 2.5 + 3.5 * math.sin(0.9 * k * k / m) * (1.0 if k % 3 else -1.5)

    sd = digest(T, m=m, top_motifs=3, top_discords=3, max_regimes=4)
    dd = sd.drilldown_discord(0)

    # Live RingMP streaming alert demo: warm up on normal Regime 1, then hit planted discord at 2250
    alert_tick = None
    with StreamMP(m=m, window=512, alpha=0.05, track_full_mp=False) as stream:
        stream.init(T[1500:2150])
        for idx_t in range(2150, 2340):
            tick = stream.update(T[idx_t])
            if tick.is_alert and alert_tick is None:
                alert_tick = tick

    # Affine-invariant leakage scan demo
    qry = array.array("d", (math.sin(0.31 * i + 1.1) + 0.5 * math.cos(0.73 * i) for i in range(400)))
    for k in range(80):
        qry[120 + k] = 3.7 * T[300 + k] - 42.5  # Affine-rescaled + shifted leak!
    leak_rep = scan_leakage(T, qry, m=64, corr_thresh=0.999)

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    print(f"=== flashmp v{version()} Demo ({simd_isa()}, {elapsed_ms:.1f} ms) ===")
    print()
    print("[1] Reversible Agentic Digest:")
    print(sd.to_compact())
    print()
    print(
        f"[2] Lossless Discord Drill-Down (rank=0 @ idx={dd['window_start']}): "
        f"recovered {len(dd['values'])} exact floats, first 4 = "
        f"{[round(x, 3) for x in dd['values'][:4]]}"
    )
    print()
    if alert_tick is not None:
        print(
            f"[3] Streaming RingMP Ville E-Process Alert: FIRED at gid={alert_tick.gid} "
            f"(E={alert_tick.e_value:.2f} >= 20.0, z={alert_tick.discord_z:.2f}, d={alert_tick.left_dist:.2f})"
        )
    else:
        print("[3] Streaming RingMP Ville E-Process: no alert triggered")
    print()
    print("[4] Affine-Invariant Leakage Scan (3.7*x - 42.5 planted leak):")
    print(leak_rep.summary())
    return 0


def _cmd_mcp(_args: argparse.Namespace) -> int:
    from .mcp_server import main as mcp_main

    return mcp_main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flashmp",
        description="SIMD-accelerated exact batch & streaming Matrix Profile + agentic digests.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_dig = sub.add_parser("digest", help="Compute a reversible agentic time-series digest.")
    p_dig.add_argument("source", help="Input file (.csv/.tsv/.json/.jsonl/.txt) or '-' for stdin.")
    p_dig.add_argument("-m", type=int, default=None, help="Subsequence length (auto-selected if omitted).")
    p_dig.add_argument("--col", default=None, help="Column name or index for CSV/TSV/JSON.")
    p_dig.add_argument(
        "--format",
        choices=("compact", "json", "md"),
        default="compact",
        help="Output format (default: compact).",
    )
    p_dig.set_defaults(func=_cmd_digest)

    p_mp = sub.add_parser("mp", help="Compute exact batch Matrix Profile.")
    p_mp.add_argument("source", help="Input file or '-' for stdin.")
    p_mp.add_argument("-m", type=int, required=True, help="Subsequence length (m >= 3).")
    p_mp.add_argument("--col", default=None, help="Column name or index.")
    p_mp.add_argument("--out", default=None, help="Write full CSV to path (or '-' for stdout).")
    p_mp.add_argument("--threads", type=int, default=0, help="Worker thread count (0 = auto).")
    p_mp.set_defaults(func=_cmd_mp)

    p_watch = sub.add_parser("watch", help="Watch streaming numbers from stdin and emit live alerts.")
    p_watch.add_argument("-m", type=int, default=32, help="Subsequence length (default: 32).")
    p_watch.add_argument("-w", "--window", type=int, default=1024, help="Sliding window size (default: 1024).")
    p_watch.add_argument("--alpha", type=float, default=0.05, help="Ville E-process alert level (default: 0.05).")
    p_watch.add_argument("--causal", action="store_true", help="Causal Left-MP only mode (lowest latency/memory).")
    p_watch.add_argument("--json", action="store_true", help="Emit alerts as JSON lines.")
    p_watch.set_defaults(func=_cmd_watch)

    p_leak = sub.add_parser("leak-scan", help="Scan query series against reference for verbatim/affine leakage.")
    p_leak.add_argument("ref_file", help="Reference (train) series file.")
    p_leak.add_argument("qry_file", help="Query (test/eval) series file.")
    p_leak.add_argument("-m", type=int, default=64, help="Subsequence length (default: 64).")
    p_leak.add_argument("--thresh", type=float, default=0.999, help="Correlation threshold (default: 0.999).")
    p_leak.add_argument("--no-neg", action="store_true", help="Disable sign-flipped match detection.")
    p_leak.add_argument("--col", default=None, help="Optional column name or index.")
    p_leak.add_argument("--json", action="store_true", help="Output full report as JSON.")
    p_leak.set_defaults(func=_cmd_leak_scan)

    p_bench = sub.add_parser("bench", help="Run built-in SIMD batch, streaming & digest self-benchmark.")
    p_bench.set_defaults(func=_cmd_bench)

    p_mcp = sub.add_parser("mcp", help="Start the native stdio JSON-RPC 2.0 MCP server.")
    p_mcp.set_defaults(func=_cmd_mcp)

    p_demo = sub.add_parser("demo", help="Run a zero-input 1-second showcase of all flashmp capabilities.")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, PermissionError, ValueError, TypeError, KeyError, OSError) as exc:
        print(f"flashmp: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
