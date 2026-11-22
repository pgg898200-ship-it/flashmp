<img align="right" src="docs/product-mark.svg" width="88" height="88" alt="Two repeated subsequences align across an observed series. Blue brackets identify their locations.">

# FlashMP

**Find a shape that repeats in a time series, even when its level and amplitude change.**

[![CI](https://github.com/pgg898200-ship-it/flashmp/actions/workflows/ci.yml/badge.svg)](https://github.com/pgg898200-ship-it/flashmp/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/pgg898200-ship-it/flashmp?display_name=tag)](https://github.com/pgg898200-ship-it/flashmp/releases)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Zero Required Deps](https://img.shields.io/badge/Dependencies-0%20required-brightgreen.svg)](pyproject.toml)

FlashMP takes a sequence of numbers and a window length. For each window, its matrix profile gives the distance to the closest matching shape and the sample index where that match begins. A small distance identifies a recurring pattern (a *motif*). A large distance flags a window worth inspecting (a *discord*).

Use it after collecting sensor readings or service metrics, before sending findings to a dashboard, an alert rule, or an agent. The Python package runs its zero-dependency C++20 SIMD engine (`AVX2+FMA` / `ARM64-NEON`) with a zero-copy PEP 3118 C-API accelerator (`_fast`) on your CPU. A CLI reads files, zero-SDK adapters expose OpenAI/Anthropic function schemas, and a local Model Context Protocol (MCP) server exposes the same analyses to agents.

![A five-sample shape at index 0 recurs at index 8 with twice the amplitude and an offset of six.](docs/repeated-shape.svg)

[Interactive Benchmark & Token Calculator](https://pgg898200-ship-it.github.io/flashmp/) · [Run the example in Colab](https://colab.research.google.com/github/pgg898200-ship-it/flashmp/blob/main/examples/find_a_repeated_shape.ipynb) · [Read the notebook on GitHub](examples/find_a_repeated_shape.ipynb) · [Reproduce the benchmarks](benchmarks/RESULTS.md)

## Find the second occurrence

Install a prebuilt native wheel (`Linux x86_64 AVX2+FMA` or `macOS arm64 Apple Silicon NEON`) from **[GitHub Releases](https://github.com/pgg898200-ship-it/flashmp/releases)**, or install from source in Python 3.10+ with Git and a C++20 compiler (`g++` or `clang++`). No Python runtime dependencies are required.

```bash
python -m pip install "git+https://github.com/pgg898200-ship-it/flashmp.git@v0.2.0"
```

```python
import flashmp

signal = [1, 3, 2, 5, 4, 0, -2, 0, 8, 12, 10, 16, 14]
profile = flashmp.stump(signal, m=5)

match = int(profile.I_[0])
print(match, signal[match:match + 5])
print(f"{profile.P_[0]:.6f}")
```

```text
8 [8, 12, 10, 16, 14]
0.000000
```

The second window equals twice the first plus six. FlashMP standardizes each window before comparing it, so the two shapes have zero distance to the displayed precision. The index and distance arrays use zero-based window start positions.

## Choose an entry point

| If you have | Use | What you get |
|---|---|---|
| A numeric series, NumPy array, or Pandas/Polars Series/DataFrame | `stump(values, m)` | Nearest-neighbor distances and positions for each window |
| A known incident and a second recording | [`abjoin(incident, recording, m)`](#find-a-known-incident-in-a-second-recording) | The closest window in the other recording for each input window |
| Live samples and a fixed retention window | `StreamMP(m=…, window=…, egress=True)` | An exact profile over retained samples as old observations expire + anytime-valid alerts |
| Training and evaluation series | `scan_leakage(train, test, m=…)` | Candidate overlapping windows, including affine-scaled or sign-flipped matches |
| A long series and an OpenAI, Anthropic, or MCP agent | `digest(values)` / `openai_tools()` / `anthropic_tools()` / MCP `ts_digest` + `ts_drilldown` | A `-99.3%` to `-99.7%` token structural summary + bit-exact lossless raw sample retrieval |

## Verified Benchmarks (`2026-11-22`, 2-vCPU x86-64 AVX2+FMA)

| Workload | **FlashMP `0.2.0`** | `pyscamp` (fp64) | `STUMPY 1.14.1` | Speedup / Edge |
|---|---:|---:|---:|---|
| Cold `import` only | **`0.57 ms`** | — | — | **0 required deps** (no NumPy/Numba) |
| Cold import + first `N=4,096` join | **`43.2 ms`** | `30.0 ms` | `38,025.1 ms` | **`880×`** vs STUMPY |
| Batch Self-Join (`N=32,768, m=64, 2T`) | **`87.3 ms`** | `767.6 ms` | `2,377.5 ms` | **`8.8×`** vs pyscamp · **`27.2×`** vs STUMPY (`3.7e-12` err) |
| Bidirectional AB-Join (`32,768 × 4,096`) | **`31.8 ms`** *(2-way)* | `103.5 ms` *(1-way)* | `502.8 ms` *(1-way)* | **`3.25×`** vs pyscamp · **`15.8×`** vs STUMPY |
| Multivariate `mSTUMP` (`D=8, N=8,192`) | **`154.9 ms`** | Unsupported | `10,609.7 ms` | **`68.5×`** vs STUMPY (`1.2e-10` err) |
| Streaming `W=1,024` (`egress=True`) | **`2.54 µs`** *(batch)* / **`3.63 µs`** *(py-loop)* / **`1.27 µs`** *(causal)* | Unsupported | `454.5 µs` | **`125×`–`358×`** faster |
| Streaming `W=16,384` (`egress=True`) | **`12.06 µs`** *(batch)* / **`13.34 µs`** *(py-loop)* / **`7.80 µs`** *(causal)* | Unsupported | `712.0 µs` | **`53×`–`91×`** faster (`3.7e-13` err) |
| Streaming Egress Parity (`W=1,024..16,384`) | **`2.2e-13`–`4.7e-13`** | Unsupported | **`1.02`–`1.57`** *(stale)* | **Exact** sliding-window MP every tick |
| High-DC Stability (`DC=1e10`, `N=2,048`) | **`4.65e-6`** max err | `2.51e-4` | `2.19e-3` | **`54×`** vs pyscamp · **`471×`** vs STUMPY |
| Post-`1e10`-Spike Exact Motif (True = `0`) | **`2.03e-16`** | **`NaN`** | `0.0` | Immune to transient giant spikes |
| Regime Segmentation (`N=4,000` vs `ruptures`) | **`3.84 ms`** (`24`-tick err) | Unsupported | — | **`9,924×`** vs `ruptures` PELT `rbf` (`38.1 s`) |
| Agentic Context Digest (`20 × N=10,000`) | **`129 tok`** (`33.5 ms`) | — | — | **`-99.31%`** tokens · **`100%`** recall vs **`0%`** LTTB/M4 |
| 5-Domain Real-World Eval (`N=10,000` each) | **`122.6 tok`** (`40.6 ms`) | — | — | **`-99.47%`** (`$763.53 → $3.68`/10k) · **`5/5` Rank #1** vs **`0/5`** LTTB/M4 |
| Multi-Agent `SwarmMonitor` (`64–1,024` ch) | **`205k–276k ch-ticks/s`** | — | — | **`100%`** fault recall · **`0`** loud-distractor FPs · **`84–119 tok`** |

Full 10-section receipts, matched Python-loop streaming timings, 5-domain telemetry evaluation, and reproduction commands: **[benchmarks/RESULTS.md](benchmarks/RESULTS.md)**.  
Head-to-head guides: **[FlashMP vs STUMPY](docs/vs-stumpy.md)** · **[FlashMP vs pyscamp](docs/vs-pyscamp.md)** · **[Architecture & Proofs](docs/architecture.md)**.

---

## Python and command-line examples

### Find a known incident in a second recording

A five-sample latency spike from yesterday is your query. Search today's recording for the same shape, even if the baseline and spike height have changed.

```python
import flashmp

incident = [20, 20, 34, 30, 22]
today = [81, 75, 84, 81, 88, 88, 116, 108, 92, 79, 80]
distance, position, _, _ = flashmp.abjoin(incident, today, m=len(incident))
start = int(position[0])
print(start, today[start:start + len(incident)], f"{distance[0]:.6f}")
```

```text
4 [88, 88, 116, 108, 92] 0.000000
```

The closest window starts at zero-based index 4. Each value in that window equals twice the corresponding incident value plus 48. The z-normalized distance therefore rounds to zero. This identifies a shape match. An alert that depends on absolute latency should also inspect the raw values.

For longer inputs, `abjoin(a, b, m)` returns four arrays. The first pair gives distances and start indices in B for each window of A. The second pair gives the reverse direction. Both inputs need at least `m` samples, and `m` must be at least 3. A constant query window has no defined z-normalized distance and returns `inf` and index `-1`.

In the [two-thread CPU benchmark](benchmarks/RESULTS.md), FlashMP `0.2.0` computed both directions in **31.84 ms** for series of 32,768 and 4,096 samples with a 64-sample window (`15.8×` vs STUMPY, `3.25×` vs pyscamp).

### Batch and multivariate profiles (plus zero-import Pandas/Polars support)

```python
import flashmp as stumpy

mp = stumpy.stump(x, m=64)            # Accepts NumPy, lists, array.array, Pandas/Polars Series/DataFrame
P, I = stumpy.mstump(X_multivar, 64)  # 68.5x faster multivariate mSTUMP, shape (D, n)
stream = stumpy.stumpi(warmup, m=64, egress=True)
stream.update(new_value)              # Exact bounded-window egress (no stale neighbors)
```

### Bounded-window streaming

```python
import flashmp

with flashmp.StreamMP(m=64, window=4096, egress=True, alpha=0.01) as stream:
    stream.init(warmup_data)
    tick = stream.update(new_value)   # 3.63-5.90 us/tick in Python via _fast C-API
    if tick.is_alert:
        print(f"Shape anomaly at gid={tick.gid}: E={tick.e_value:.1f}, z={tick.discord_z:.2f}")
```

### Summarize a series and inspect its raw samples

```python
import flashmp

d = flashmp.digest(df_or_array, m=64) # Compresses 10,000 floats (~18k-41k tok) -> ~125 tokens
print(d.to_compact())                 # Feed into any LLM prompt or agent memory

# Losslessly inspect exact bit-identical raw floats whenever the agent wants to zoom in:
anom = d.drilldown_discord(0)
exact_slice = d.drilldown(start=2200, end=2264)
```

### OpenAI, Anthropic, and MCP agent toolkits (zero SDK dependencies)

```python
import flashmp

oa_tools = flashmp.openai_tools(format="chat")          # or format="responses", strict=True
cl_tools = flashmp.anthropic_tools()

toolkit = flashmp.AgentToolkit(allowed_dirs=["/path/to/data"])
result = toolkit.call("ts_digest", {"path": "/path/to/data/metrics.csv", "m": 64})
```

### Scan training and evaluation splits

Catches verbatim copies, affine-rescaled/shifted ($a x + b$) windows, and sign-flipped overlaps between training corpora and evaluation splits:

```python
import flashmp

report = flashmp.scan_leakage(train_series, test_series, m=64, corr_thresh=0.999)
print(report.summary())
```

### Monitor multiple channels

Monitor 64–4,096 concurrent streams at **`205k–276k ch-ticks/s`** with scale-invariant causal `RingMP` (immune to $3.5\times$ high-amplitude normal-shape distractors), order-invariant CRDT alert fusion (`AlertLedger`), and strict token-capped briefs:

```python
import flashmp

with flashmp.SwarmMonitor(channel_names, m=32, window=256, alpha=0.05) as mon:
    mon.init_channel("node_0", warmup_0)
    mon.update_frame(live_tick_dict)
    print(mon.brief(max_tokens=300))   # Strictly <= 300 tokens, ranked by E-value
```

### CLI and MCP server

```bash
# 1-second showcase & built-in SIMD benchmark
flashmp demo
flashmp bench

# Compress any CSV/TSV/JSON/JSONL/txt (or stdin '-') into a ~125-token digest
flashmp digest metrics.csv --col latency_ms --format compact

# Audit train/test CSVs for verbatim & affine subsequence leakage
flashmp leak-scan train.csv test.csv -m 64 --thresh 0.999

# Stream live telemetry from stdin with Ville E-value alerts
tail -f sensor.txt | flashmp watch -m 64 -w 4096 --alpha 0.01 --json

# Print ready-to-paste MCP client config (claude-code | cursor | codex | gemini)
flashmp mcp-config --client claude-code --allowed-dir /path/to/your/data

# Install SKILL.md agent playbook in 1 command (claude | cursor | codex | all)
flashmp install-skill --target claude

# Launch sandboxed zero-dependency stdio JSON-RPC 2.0 MCP server
flashmp-mcp
```

For clients that use JSON `mcpServers` configuration, point the command at the installed executable (or run `flashmp mcp-config`). Set the data directory to the location of your files.

```json
{
  "mcpServers": {
    "flashmp": {
      "command": "flashmp-mcp",
      "args": [],
      "env": {
        "FLASHMP_MCP_ALLOWED_DIRS": "/path/to/your/data"
      }
    }
  }
}
```

Use `ts_digest` with `values` or a file `path`. Pass the returned `handle`, `start` and exclusive `end` to `ts_drilldown` in that server session to recover a raw interval.

MCP exposes eight tools — `ts_digest`, `ts_drilldown`, `ts_anomalies`, `ts_regimes`, `ts_leak_scan`, `ts_stream_create`, `ts_stream_push`, and `ts_swarm_sift` — with strict `allowed_dirs` path/symlink confinement, bounded 64-session LRU cleanup, and DoS caps.

---

## Scope and limitations

- **Massive single-series batch joins ($N > 10^6$) on an NVIDIA GPU:** Use **[pyscamp](https://github.com/zpzim/SCAMP)** with CUDA GPU acceleration. FlashMP currently targets CPU SIMD (AVX2+FMA / ARM64-NEON) and does not yet include a CUDA backend.
- **Distributed multi-node Dask clusters or specialized niche MP routines:** Use **[STUMPY](https://github.com/TDAmeritrade/stumpy)** if you specifically rely on `stumped` (Dask), `ostinato` (consensus motifs), `mpdist`, or `pan` matrix profiles not yet implemented in FlashMP.
- **Flat subsequences:** In univariate `stump`/`abjoin`/`scan_leakage`, zero-variance (constant) subsequences have undefined z-normalization ($0/0$) and are masked to `+inf` (`idx = -1`) to prevent false zero-distance motifs or leakage hits on flat telemetry (whereas `STUMPY` reports distance `0.0` between flat windows). In `mstump`, FlashMP matches `stumpy.mstump`'s exact flat-window convention.

---

## Documentation & Verification

- **[benchmarks/RESULTS.md](benchmarks/RESULTS.md)** — Full 10-section live-measured benchmark suite & reproduction scripts
- **[docs/vs-stumpy.md](docs/vs-stumpy.md)** — Head-to-head comparison with STUMPY
- **[docs/vs-pyscamp.md](docs/vs-pyscamp.md)** — Head-to-head comparison with pyscamp / SCAMP
- **[docs/architecture.md](docs/architecture.md)** — Mathematical proofs, 48-wide two-stage float32-screened exact float64 engine & SIMD microarchitecture
- **[docs/index.html](docs/index.html)** — Self-contained interactive benchmark explorer & LLM token-cost calculator
- **[SKILL.md](SKILL.md)** & **[llms.txt](llms.txt)** — Agent skill playbook & LLM-readable index

Run the full verification suite (49 tests, including C++20 AddressSanitizer + UndefinedBehaviorSanitizer stress harness, Red-Team regressions, and bare-stdlib `_fast`/AI-integration tests):

```bash
pytest tests/ -v
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
