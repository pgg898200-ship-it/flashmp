---
name: flashmp
description: Find recurring shapes and unusual windows in numeric time series. Compute CPU matrix profiles, monitor bounded streams, scan for train/test subsequence overlap, or give an agent a compact time-series summary with exact sample retrieval.
---

# FlashMP

Use FlashMP when a task calls for shape matching over numeric sequences. Choose the window length `m` in samples. The Python package provides batch and streaming analysis, a file-oriented CLI and a local stdio MCP server.

- For a single trace, `stump(x, m)` returns nearest-neighbor distances (`P_`) and start indices (`I_`). A low finite distance means a close shape match after z-normalization.
- For live data with a fixed retention length, construct `StreamMP` with `egress=True` and feed it samples in order.
- For an evaluation audit, use `scan_leakage(train, test, m)` to identify windows with high positive or negative correlation for inspection.
- For an agent reading a long series, call `ts_digest` with `values` or a file `path`. Use the returned handle in the same server session for `ts_drilldown` requests. It returns the original samples for a half-open range `[start, end)`.
- For many channels, `SwarmMonitor` collects alerts into a bounded brief and keeps a rolling window of samples for drill-down.

[Install and run a worked example](README.md#find-the-second-occurrence) · [Measured workloads and limitations](benchmarks/RESULTS.md)

## Quick Commands

### CLI
```bash
# 1-second self-contained showcase & built-in SIMD benchmark
flashmp demo
flashmp bench

# Compress a CSV/TSV/JSON/JSONL/txt (or stdin '-') into a ~130-token structural digest
flashmp digest metrics.csv --col latency_ms --format compact

# Check train/test splits for exact & affine-invariant subsequence leakage
flashmp leak-scan train.csv test.csv -m 64 --thresh 0.999

# Stream live values from stdin with anytime-valid e-CUSUM alerts
tail -f sensor.txt | flashmp watch -m 64 -w 4096 --alpha 0.01 --json

# Launch sandboxed zero-dependency stdio JSON-RPC 2.0 MCP server
flashmp-mcp
```

### Python
```python
import flashmp

# Drop-in STUMPY replacement (0.78 ms import, zero JIT)
mp = flashmp.stump(x, m=64)           # [P, I, left_I, right_I], shape (n, 4)
P, I = flashmp.mstump(X_multivar, 64) # 48.6x faster multivariate mSTUMP, shape (D, n)

# Reversible agentic digest (-99.3% tokens, bit-exact lossless drill-down)
d = flashmp.digest(x, m=64)
print(d.to_compact())
raw_anom = d.drilldown_discord(0)
raw_motif = d.drilldown_motif(0)

# Exact bounded-window streaming + anytime-valid e-CUSUM alerting
with flashmp.StreamMP(m=64, window=4096, egress=True, alpha=0.01) as stream:
    stream.init(warmup_data)
    tick = stream.update(new_value)
    if tick.is_alert:
        print(f"Anomaly at gid={tick.gid}, e_value={tick.e_value:.1f}")

# Affine-invariant train/test contamination scan
report = flashmp.scan_leakage(train_series, test_series, m=64, corr_thresh=0.999)
print(report.summary())
```

## MCP configuration for clients using the JSON mcpServers format

```json
{
  "mcpServers": {
    "flashmp": {
      "command": "flashmp-mcp",
      "env": {
        "FLASHMP_MCP_ALLOWED_DIRS": "/path/to/workspace"
      }
    }
  }
}
```

### MCP Tools Exposed
- `ts_digest` — Structural compression of array or sandboxed file into a compact digest and session handle
- `ts_drilldown` — Bit-exact lossless recovery of raw slice `[start:end]` or `discord_rank` from session handle
- `ts_anomalies` — Exact top-k shape discords with z-scores and nearest normal neighbors
- `ts_regimes` — Exact FLOSS semantic regime boundaries
- `ts_leak_scan` — Exact & affine-invariant subsequence contamination audit
- `ts_stream_create` / `ts_stream_push` — Stateful bounded-window RingMP streaming over MCP
- `ts_swarm_sift` — Multi-channel firehose anomaly sifter with strict token cap
