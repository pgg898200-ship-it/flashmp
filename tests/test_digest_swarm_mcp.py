"""Tests for Reversible Agentic Digest, Multi-Agent SwarmMonitor, Native MCP Server, and CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import flashmp


def test_reversible_digest_lossless_and_high_compression():
    rng = np.random.default_rng(11)
    N = 4000
    t = np.arange(N, dtype=np.float64)
    x = np.sin(2.0 * np.pi * t / 40.0) + 0.04 * rng.standard_normal(N)
    # Inject a shape discord
    x[1500:1540] = np.linspace(-1.0, 1.0, 40) * np.cos(np.linspace(0, 9 * np.pi, 40))

    d = flashmp.digest(x, m=40, top_motifs=3, top_discords=3, max_regimes=4)
    compact = d.to_compact()
    assert len(compact) < 2500
    assert d.token_savings_pct > 95.0
    assert any(abs(disc["index"] - 1500) <= 40 for disc in d.discords[:2])

    # Lossless drilldown must be bit-identical to original samples
    dd = d.drilldown(1500, 1540)
    rec = np.asarray(dd["values"], dtype=np.float64)
    assert np.array_equal(rec, x[1500:1540])

    dd_disc = d.drilldown_discord(0)
    assert "values" in dd_disc and len(dd_disc["values"]) == 40
    assert dd_disc["nearest_normal_motif"] is not None


def test_swarm_monitor_commutative_and_distractor_immune():
    rng = np.random.default_rng(22)
    C, T, m = 40, 360, 24
    t = np.arange(T, dtype=np.float64)
    data = np.empty((C, T), dtype=np.float64)
    for c in range(C):
        amp = 4.0 if c in (1, 2, 3) else 1.0  # 4x loud distractors
        data[c] = amp * np.sin(2 * np.pi * t / m + c * 0.2) + 0.03 * amp * rng.standard_normal(T)
    # True shape anomaly on channel 17
    data[17, 290:320] = 0.8 * np.sign(np.sin(np.linspace(0, 9 * np.pi, 30)))

    names = [f"s{c:02d}" for c in range(C)]
    mon = flashmp.SwarmMonitor(channels=names, m=m, window=200, alpha=0.01)
    for c in range(C):
        mon.init_channel(names[c], data[c, :220])
    for step in range(220, T):
        mon.update_frame(data[:, step])

    brief = mon.brief(max_tokens=300, top_k=5)
    assert len(brief) <= 1300
    assert "s17" in brief
    # Verify 4x loud distractors caused zero false alarms
    alerted = {a.channel_id for a in mon.ledger.alerts(only_triggered=True)}
    assert not (alerted & {"s01", "s02", "s03"})

    # Verify commutativity under reversed order + duplicate idempotency
    findings = mon.ledger.alerts()
    l1 = flashmp.AlertLedger.merge_all(findings)
    l2 = flashmp.AlertLedger.merge_all(list(reversed(findings)) + findings[:5])
    assert l1 == l2
    mon.close()


def test_mcp_server_stdio_jsonrpc():
    """Drive the zero-dependency stdio MCP server over real stdin/stdout JSON-RPC 2.0."""
    rng = np.random.default_rng(33)
    arr = (np.sin(np.linspace(0, 20 * np.pi, 300)) + 0.03 * rng.standard_normal(300)).tolist()
    arr[120:140] = [(-1.0) ** i * 0.9 for i in range(20)]

    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "ts_digest", "arguments": {"values": arr, "m": 20}},
        },
    ]
    payload = "\n".join(json.dumps(r) for r in reqs) + "\n"
    proc = subprocess.run(
        [sys.executable, "-m", "flashmp.mcp_server"],
        input=payload,
        text=True,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
        timeout=15,
        check=True,
    )
    lines = [json.loads(ln) for ln in proc.stdout.strip().splitlines() if ln.strip()]
    assert len(lines) == 3
    assert lines[0]["id"] == 1 and "serverInfo" in lines[0]["result"]
    tool_names = {t["name"] for t in lines[1]["result"]["tools"]}
    assert {"ts_digest", "ts_drilldown", "ts_anomalies", "ts_leak_scan", "ts_swarm_sift"} <= tool_names
    dig_text = lines[2]["result"]["content"][0]["text"]
    assert "handle" in dig_text.lower() or "discord" in dig_text.lower()


def test_cli_demo_and_help():
    env = {"PYTHONPATH": str(ROOT / "src")}
    out_help = subprocess.check_output(
        [sys.executable, "-m", "flashmp.cli", "--help"],
        env=env,
        text=True,
    )
    assert "digest" in out_help and "leak-scan" in out_help and "watch" in out_help

    out_demo = subprocess.check_output(
        [sys.executable, "-m", "flashmp.cli", "demo"],
        env=env,
        text=True,
    )
    assert "flashmp" in out_demo.lower() and "TS_DIGEST" in out_demo and "LEAK_SCAN" in out_demo
