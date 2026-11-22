"""flashmp: SIMD-accelerated exact batch & streaming Matrix Profile + agentic digests."""

from __future__ import annotations

__version__ = "0.2.0"

_EXPORTS: dict[str, tuple[str, str]] = {
    # Native library info
    "version": ("._native", "version"),
    "simd_isa": ("._native", "simd_isa"),
    # Batch Matrix Profile & analysis
    "stump": (".core", "stump"),
    "abjoin": (".core", "abjoin"),
    "mstump": (".core", "mstump"),
    "profile": (".core", "profile"),
    "floss": (".core", "floss"),
    "find_regimes": (".core", "find_regimes"),
    "auto_window": (".core", "auto_window"),
    "motifs": (".core", "motifs"),
    "discords": (".core", "discords"),
    "MatrixProfileResult": (".core", "MatrixProfileResult"),
    "MPArray": (".core", "MPArray"),
    "Motif": (".core", "Motif"),
    "Discord": (".core", "Discord"),
    # Streaming RingMP + Ville E-process
    "StreamMP": (".stream", "StreamMP"),
    "TickResult": (".stream", "TickResult"),
    "stumpi": (".stream", "stumpi"),
    # Affine-invariant leakage scanning
    "scan_leakage": (".leak", "scan_leakage"),
    "LeakReport": (".leak", "LeakReport"),
    "LeakExample": (".leak", "LeakExample"),
    # Reversible agentic time-series digest
    "digest": ("._digest", "digest"),
    "SeriesDigest": ("._digest", "SeriesDigest"),
    # Multi-agent swarm monitoring
    "SwarmMonitor": (".swarm", "SwarmMonitor"),
    "SwarmAlert": (".swarm", "SwarmAlert"),
    "AlertLedger": (".swarm", "AlertLedger"),
    # AI agent & DataFrame integrations
    "openai_tools": (".integrations", "openai_tools"),
    "anthropic_tools": (".integrations", "anthropic_tools"),
    "AgentToolkit": (".integrations", "AgentToolkit"),
    "mcp_client_config": (".integrations", "mcp_client_config"),
    "install_skill": (".integrations", "install_skill"),
    "extract_series": (".integrations", "extract_series"),
}

__all__ = [
    "__version__",
    "stump",
    "abjoin",
    "mstump",
    "stumpi",
    "profile",
    "floss",
    "find_regimes",
    "auto_window",
    "motifs",
    "discords",
    "StreamMP",
    "TickResult",
    "MatrixProfileResult",
    "MPArray",
    "Motif",
    "Discord",
    "scan_leakage",
    "LeakReport",
    "LeakExample",
    "digest",
    "SeriesDigest",
    "SwarmMonitor",
    "SwarmAlert",
    "AlertLedger",
    "openai_tools",
    "anthropic_tools",
    "AgentToolkit",
    "mcp_client_config",
    "install_skill",
    "extract_series",
    "version",
    "simd_isa",
]


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'flashmp' has no attribute {name!r}")
    mod_rel, attr = target
    mod = __import__(f"{__name__}{mod_rel}", fromlist=[attr])
    value = getattr(mod, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
