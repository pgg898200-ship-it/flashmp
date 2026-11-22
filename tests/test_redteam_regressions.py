"""Regression test suite encoding every finding from the independent Red-Team audit."""

from __future__ import annotations

import array
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import flashmp
from flashmp.mcp_server import MCPServer
from flashmp.swarm import AlertLedger, SwarmAlert, SwarmMonitor


# ---------------------------------------------------------------------------
# C1: MCP Path Sandbox & Secret Key/Header Exfiltration Defense
# ---------------------------------------------------------------------------
def test_c1_mcp_path_sandbox_blocks_traversal_and_symlinks(tmp_path: Path):
    allowed = tmp_path / "workspace"
    secret_dir = tmp_path / "secret"
    allowed.mkdir()
    secret_dir.mkdir()

    secret_json = secret_dir / "creds.json"
    secret_json.write_text(json.dumps({"aws_secret_access_key": [1.0, 2.0], "db_password": "xyz"}))

    ok_csv = allowed / "metrics.csv"
    ok_csv.write_text("t,val\n" + "\n".join(f"{i},{math.sin(i * 0.2):.4f}" for i in range(60)) + "\n")

    # Symlink inside workspace pointing outside workspace must be blocked!
    evil_link = allowed / "sneaky.json"
    evil_link.symlink_to(secret_json)

    srv = MCPServer(allowed_dirs=[allowed])

    # 1. Traversal outside allowed_dirs blocked
    r1 = srv.handle_message({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "ts_digest", "arguments": {"path": str(secret_json), "column": "__probe__"}},
    })
    assert r1 is not None and r1["result"]["isError"] is True
    msg1 = r1["result"]["content"][0]["text"]
    assert "aws_secret_access_key" not in msg1
    assert "Access denied" in msg1

    # 2. Symlink escaping allowed_dirs blocked
    r2 = srv.handle_message({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "ts_digest", "arguments": {"path": str(evil_link)}},
    })
    assert r2 is not None and r2["result"]["isError"] is True
    assert "Access denied" in r2["result"]["content"][0]["text"]

    # 3. Missing column inside allowed file does NOT dump raw header/keys
    ok_json = allowed / "data.json"
    ok_json.write_text(json.dumps({"hidden_key_1": [1.0] * 20, "signal": [float(i) for i in range(20)]}))
    r3 = srv.handle_message({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "ts_digest", "arguments": {"path": str(ok_json), "column": "__missing__"}},
    })
    assert r3 is not None and r3["result"]["isError"] is True
    assert "hidden_key_1" not in r3["result"]["content"][0]["text"]

    # 4. Legitimate file inside allowed_dirs succeeds cleanly
    r4 = srv.handle_message({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "ts_digest", "arguments": {"path": str(ok_csv), "m": 12}},
    })
    assert r4 is not None and r4["result"]["isError"] is False


# ---------------------------------------------------------------------------
# C2: Catastrophic SCAMP Covariance Cancellation After Giant Spikes (All 5 Engines)
# ---------------------------------------------------------------------------
def test_c2_spike_cancellation_defense_all_engines():
    rng = np.random.default_rng(2026)

    # 1. Batch stump with 1e10 spikes + planted exact motif
    x = rng.standard_normal(600)
    x[50] = 1e10
    x[250] = 1e10
    pat = np.sin(np.linspace(0, 2 * np.pi, 16))
    x[100:116] = pat
    x[300:316] = pat
    mp = flashmp.stump(x, 16)
    assert float(mp.P_[100]) < 1e-12
    assert int(mp.I_[100]) == 300

    # Verify against brute-force reference on random subset of post-spike windows
    for idx in [100, 120, 200, 300, 400, 500]:
        q = x[idx : idx + 16]
        qn = (q - q.mean()) / np.linalg.norm(q - q.mean())
        best = math.inf
        for j in range(len(x) - 16 + 1):
            if abs(idx - j) <= 4:
                continue
            s = x[j : j + 16]
            sn = (s - s.mean()) / np.linalg.norm(s - s.mean())
            d = float(np.linalg.norm(qn - sn)) * math.sqrt(16)
            if d < best:
                best = d
        assert abs(float(mp.P_[idx]) - best) < 1e-9

    # 2. scan_leakage on independent noise with 1e10 spikes -> 0 post-spike false-positive hits!
    ref = rng.standard_normal(500)
    qry = rng.standard_normal(500)
    ref[40] = 1e10
    qry[40] = 1e10
    rep = flashmp.scan_leakage(ref, qry, m=16, corr_thresh=0.999)
    # Windows 25..40 genuinely share an identical Dirac delta spike; windows 41+ must have 0 hits!
    assert int(np.sum(np.asarray(rep.best_corr[41:]) >= 0.999)) == 0
    # Opposite-sign spikes with check_neg=False -> 0 hits anywhere!
    qry_neg = rng.standard_normal(500)
    qry_neg[40] = -1e10
    rep_neg = flashmp.scan_leakage(ref, qry_neg, m=16, corr_thresh=0.999, check_neg=False)
    assert rep_neg.contaminated_windows == 0

    # 3. Multivariate mstump with 1e9 spikes + planted exact 4D motif
    M = rng.standard_normal((4, 240))
    M[:, 20] = 1e9
    M[:, 120] = 1e9
    for d in range(4):
        w = np.sin(np.linspace(0, (d + 2) * np.pi, 16))
        M[d, 60:76] = w
        M[d, 170:186] = w
    P_ms, I_ms = flashmp.mstump(M, 16)
    assert float(P_ms[3, 60]) < 1e-12
    assert int(I_ms[3, 60]) == 170

    # 4. Streaming RingMP with ±1e10 spikes + planted exact motif
    xs = rng.standard_normal(340)
    xs[30] = 1e10
    xs[110] = -1e10
    xs[150:166] = pat
    xs[230:246] = pat
    with flashmp.StreamMP(m=16, window=256, egress=True, track_full_mp=True) as sm:
        sm.init(xs[:200])
        sm.update_batch(xs[200:])
        snap = sm.snapshot(relative_indices=True)
    # Exact batch MP of final 256 subsequences (256 + 16 - 1 = 271 raw ticks)
    tail_raw = xs[len(xs) - (256 + 16 - 1) :]
    true_mp = flashmp.profile(tail_raw, 16).mp
    assert float(np.max(np.abs(snap.mp - true_mp))) < 1e-9


# ---------------------------------------------------------------------------
# C3: flashmp.digest() Not Shadowed After Importing mcp_server / cli
# ---------------------------------------------------------------------------
def test_c3_digest_callable_not_shadowed():
    import flashmp.cli  # noqa: F401
    import flashmp.mcp_server  # noqa: F401

    assert callable(flashmp.digest)
    d = flashmp.digest([math.sin(i * 0.3) for i in range(64)], m=8)
    assert d.n_points == 64


# ---------------------------------------------------------------------------
# C4: SwarmAlert.merge() / AlertLedger True Semilattice (Assoc, Commut, Idemp)
# ---------------------------------------------------------------------------
def test_c4_swarm_alert_crdt_axioms():
    A = SwarmAlert("c", 1, 1, 10.0, 5.0, 1.0, 0, 0.01, True, "c@1")
    B = SwarmAlert("c", 1, 2, 10.0, 1.0, 10.0, 0, 0.01, True, "c@2")
    C = SwarmAlert("c", 1, 3, 10.0, 5.0, 5.0, 0, 0.01, True, "c@3")
    D = SwarmAlert("c", 1, 1, 10.0, 5.0, 1.0, 0, 0.01, True, "c@1_alt")
    E = SwarmAlert("c", 1, 4, float("nan"), float("inf"), 1.0, -1, 1.0, False, "c@4")
    F = SwarmAlert("c", 1, 5, float("inf"), float("nan"), 1.0, -1, 1.0, False, "c@5")

    # Associativity
    assert A.merge(B).merge(C) == A.merge(B.merge(C))
    # Order-invariance in AlertLedger
    assert AlertLedger([A, B, C]) == AlertLedger([B, C, A])
    assert AlertLedger([A, B, C]) == AlertLedger([C, A, B])
    # Commutativity on ties with different handles and non-finite metrics
    assert A.merge(D) == D.merge(A)
    assert E.merge(F) == F.merge(E)
    # Idempotence
    assert A.merge(A) == A


# ---------------------------------------------------------------------------
# H1, H2, H3, H4, H5, H6, H7: High-Severity Hardening Checks
# ---------------------------------------------------------------------------
def test_h1_unaligned_numpy_view_handled_safely():
    raw = np.zeros(803, dtype=np.uint8)
    u = np.ndarray((100,), dtype=np.float64, buffer=raw[1:801])
    u[:] = np.sin(np.linspace(0, 10, 100))
    assert not u.flags.aligned
    mp = flashmp.stump(u, 12)
    mp_aligned = flashmp.stump(u.copy(), 12)
    assert float(np.max(np.abs(mp.P_ - mp_aligned.P_))) < 1e-14


def test_h2_stream_snapshot_no_negative_indices_when_causal_egress():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(200)
    with flashmp.StreamMP(m=16, window=32, egress=True, track_full_mp=False) as sm:
        sm.init(x[:100])
        sm.update_batch(x[100:])
        snap = sm.snapshot(relative_indices=True)
        assert int(np.min(snap.idx)) >= -1
        assert int(np.min(snap.left_idx)) >= -1


def test_h3_h4_extreme_scale_and_anchor_overflow():
    rng = np.random.default_rng(99)
    base = np.sin(np.linspace(0, 12 * np.pi, 100)) + 0.1 * rng.standard_normal(100)
    ref = flashmp.stump(base, 12).P_

    # H4: 1e-160 and 1e160 match unit scale to < 1e-12
    for scale in (1e-160, 1e-150, 1e160):
        mp_s = flashmp.stump(base * scale, 12).P_
        assert np.all(np.isfinite(mp_s))
        assert float(np.max(np.abs(mp_s - ref))) < 1e-12

    # H3: [1e308, -1e308, *randn(100)] does not poison subsequent valid windows
    poison = np.concatenate([[1e308, -1e308], base])
    mp_p = flashmp.stump(poison, 12).P_
    assert np.all(np.isfinite(mp_p[2:]))
    assert float(np.max(np.abs(mp_p[2:] - ref))) < 1e-12

    with flashmp.StreamMP(m=12, window=64, egress=True, track_full_mp=True) as sm:
        sm.init(poison[:70])
        sm.update_batch(poison[70:])
        snap = sm.snapshot(relative_indices=True)
    assert np.all(np.isfinite(snap.mp))


def test_h5_floss_constant_and_nan_series_zero_changepoints():
    for arr in (np.ones(200), np.full(200, np.nan)):
        res = flashmp.profile(arr, 16)
        cac = flashmp.floss(res.idx, 16)
        assert np.all(cac == 1.0)
        assert flashmp.find_regimes(cac, 16) == []


def test_h6_h7_mcp_framing_and_overflow_guards():
    srv = MCPServer()

    # Content-Type before Content-Length works cleanly
    body = json.dumps({"jsonrpc": "2.0", "id": 42, "method": "ping"}).encode()
    frame = f"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
    out = io.BytesIO()
    srv.serve(io.BytesIO(frame), out)
    raw_resp = out.getvalue()
    assert b'"id":42' in raw_resp and b'"result":{}' in raw_resp

    # Negative Content-Length rejected cleanly (no drain / crash)
    out2 = io.BytesIO()
    srv.serve(io.BytesIO(b"Content-Length: -5\r\n\r\n"), out2)
    assert b"-32600" in out2.getvalue()

    # Missing/non-string method returns -32600 (M12)
    r_bad = srv.handle_message({"jsonrpc": "2.0", "id": 9, "method": 123})
    assert r_bad is not None and r_bad["error"]["code"] == -32600

    # H7: huge L in floss does not overflow signed int64
    cac = flashmp.floss(np.array([2, 3, 0, 1], dtype=np.int64), 2**60)
    assert len(cac) == 4 and np.all(np.isfinite(cac))

    # H6: huge window in MCP rejected cleanly
    r_big = srv.handle_message({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {"name": "ts_stream_create", "arguments": {"m": 16, "window": 10_000_000}},
    })
    assert r_big is not None and r_big["result"]["isError"] is True


# ---------------------------------------------------------------------------
# M2..M11 & L3..L4: Medium & Polish Regressions
# ---------------------------------------------------------------------------
def test_medium_and_polish_regressions():
    # M2: auto_window always in [3, min(max_m, N)]
    assert 3 <= flashmp.auto_window([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], min_m=8, max_m=64) <= 6
    assert flashmp.auto_window(np.sin(np.linspace(0, 20, 100)), min_m=20, max_m=8) == 8

    # M3: digest copies array.array('d') so mutating caller doesn't corrupt drilldown
    buf = array.array("d", [math.sin(i * 0.25) for i in range(64)])
    orig = list(buf[10:20])
    d = flashmp.digest(buf, m=8)
    buf[10:20] = array.array("d", [999.0] * 10)
    assert d.drilldown(10, 20)["values"] == orig

    # M4: multivariate motif correlation equals exact mean per-channel Pearson r
    t = np.linspace(0, 6 * np.pi, 100)
    ch0 = np.sin(t)
    ch1 = np.cos(t)
    # Plant exact match on ch0 and orthogonal (r=0) pair on ch1
    ch0[50:58] = ch0[10:18]
    a1 = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    b1 = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    ch1[10:18] = a1
    ch1[50:58] = b1
    d_mv = flashmp.digest(np.vstack([ch0, ch1]), m=8, top_motifs=1)
    assert d_mv.motifs
    mt = d_mv.motifs[0]
    dd_m = d_mv.drilldown_motif(0)
    assert dd_m["start"] == mt["index"] and len(dd_m["values"]) == 8

    # M5: update_batch updates last_tick
    with flashmp.StreamMP(m=8, window=32) as sm:
        sm.init(np.sin(np.linspace(0, 10, 40)))
        sm.update_batch(np.sin(np.linspace(10, 15, 25)))
        assert sm.last_tick is not None and sm.last_tick.gid == (40 + 25 - 8)

    # M6: reject complex, str, bool, object-None
    with pytest.raises(TypeError):
        flashmp.stump(np.array([1 + 2j] * 20), 4)
    with pytest.raises(TypeError):
        flashmp.stump("1234567890", 4)
    with pytest.raises(TypeError):
        flashmp.stump(np.array([1.0, None] * 10, dtype=object), 4)
    with pytest.raises(TypeError):
        d.drilldown(0, 5, channel=True)

    # M7: motifs NumPy path does not lose a slot on self-loops
    mp_arr = np.array([0.0, 0.5, 0.5, 0.2, 0.2, 1.0, 1.0, 1.0], dtype=np.float64)
    idx_arr = np.array([0, 2, 1, 4, 3, 6, 5, 5], dtype=np.int64)  # idx[0]==0 self-loop!
    mots = flashmp.motifs(mp_arr, idx_arr, m=4, k=2, excl_zone=0)
    assert len(mots) == 2

    # M8: negative excl_zone raises ValueError
    with pytest.raises(ValueError):
        flashmp.profile(np.arange(30, dtype=np.float64), 8, excl_zone=-3)
    with pytest.raises(ValueError):
        flashmp.StreamMP(m=8, window=32, excl_zone=-3)

    # M10: SwarmMonitor brief respects tiny max_tokens & channel names with '@'
    with SwarmMonitor(["node@1", "node@2"], m=8, window=32) as mon:
        mon.update_channel("node@1", 1.5)
        assert mon.drilldown("node@1")["values"] == [1.5]
        b = mon.brief(max_tokens=5)
        assert len(b) <= 20

    # M11 & L3: strict drilldown & m validation + exact compact token header
    with pytest.raises(ValueError):
        d.drilldown(-1, 5)
    with pytest.raises(ValueError):
        d.drilldown(10, 5)
    with pytest.raises(ValueError):
        flashmp.digest(list(range(20)), m=-5)
    d_nan = flashmp.digest([float("nan")] * 30, m=6)
    txt = d_nan.to_compact()
    assert d_nan.compact_tokens_est == (len(txt) + 3) // 4

    # L4: CLI exits cleanly with code 2 (no raw Python traceback) on missing file
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "flashmp", "digest", "/nonexistent/file.csv"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "flashmp: error:" in proc.stderr
