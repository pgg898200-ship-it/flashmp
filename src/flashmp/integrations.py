"""Zero-dependency AI agent adapters (OpenAI, Anthropic, MCP, Skills) and DataFrame helpers."""

from __future__ import annotations

import array
import copy
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._digest import _sanitize_json
from .mcp_server import MAX_SESSIONS, MCPServer, TOOL_DEFINITIONS

__all__ = [
    "openai_tools",
    "anthropic_tools",
    "AgentToolkit",
    "mcp_client_config",
    "install_skill",
    "extract_series",
]

_TIME_LIKE_COLUMNS = frozenset(
    {"t", "time", "timestamp", "ts", "date", "datetime", "index", "idx", "step"}
)

_SUPPORTED_MCP_CLIENTS = ("claude-code", "cursor", "codex", "gemini")
_SKILL_TARGETS = ("claude", "cursor", "codex")


def _normalize_allowed_dirs(
    allowed_dirs: Sequence[str | Path] | str | Path | None,
) -> list[str | Path] | None:
    if allowed_dirs is None:
        return None
    if isinstance(allowed_dirs, (str, Path)):
        return [allowed_dirs]
    dirs = list(allowed_dirs)
    if not dirs:
        raise ValueError("allowed_dirs cannot be empty; pass None to use the default working directory")
    return dirs


# ---------------------------------------------------------------------------
# 1. OpenAI & Anthropic Tool Schema Adapters
# ---------------------------------------------------------------------------


def openai_tools(format: str = "chat", strict: bool = False) -> list[dict[str, Any]]:
    """Return OpenAI function-calling tool definitions for all 8 flashmp tools.

    Parameters
    ----------
    format:
        - ``"chat"`` (default): Chat Completions API schema
          ``{"type": "function", "function": {"name", "description", "parameters"}}``.
        - ``"responses"``: OpenAI Responses API flat schema
          ``{"type": "function", "name", "description", "parameters"}``.
    strict:
        If True, sets ``"strict": True`` and ``"additionalProperties": False`` on each schema.
    """
    fmt = str(format).strip().lower()
    if fmt not in ("chat", "responses"):
        raise ValueError(f"Unsupported OpenAI tool format {format!r}; expected 'chat' or 'responses'")

    tools: list[dict[str, Any]] = []
    for raw in TOOL_DEFINITIONS:
        params = copy.deepcopy(raw["inputSchema"])
        if strict:
            params["additionalProperties"] = False
        if fmt == "chat":
            fn_def: dict[str, Any] = {
                "name": raw["name"],
                "description": raw["description"],
                "parameters": params,
            }
            if strict:
                fn_def["strict"] = True
            tools.append({"type": "function", "function": fn_def})
        else:
            entry: dict[str, Any] = {
                "type": "function",
                "name": raw["name"],
                "description": raw["description"],
                "parameters": params,
            }
            if strict:
                entry["strict"] = True
            tools.append(entry)
    return tools


def anthropic_tools() -> list[dict[str, Any]]:
    """Return Anthropic Messages API tool-use definitions for all 8 flashmp tools."""
    return [
        {
            "name": raw["name"],
            "description": raw["description"],
            "input_schema": copy.deepcopy(raw["inputSchema"]),
        }
        for raw in TOOL_DEFINITIONS
    ]


class AgentToolkit:
    """Sandboxed multi-framework tool runner for OpenAI, Anthropic, and MCP agents.

    Wraps a stateful, path-sandboxed :class:`flashmp.mcp_server.MCPServer` instance so
    handles returned by ``ts_digest``, ``ts_stream_create``, and ``ts_swarm_sift`` persist
    across tool calls for lossless ``ts_drilldown`` and ``ts_stream_push``.
    """

    def __init__(
        self,
        allowed_dirs: Sequence[str | Path] | str | Path | None = None,
        max_sessions: int = MAX_SESSIONS,
    ) -> None:
        norm_dirs = _normalize_allowed_dirs(allowed_dirs)
        self._server = MCPServer(max_sessions=max_sessions, allowed_dirs=norm_dirs)

    @property
    def allowed_dirs(self) -> list[Path] | None:
        return self._server.allowed_dirs

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(t["name"] for t in TOOL_DEFINITIONS)

    def openai_tools(self, format: str = "chat", strict: bool = False) -> list[dict[str, Any]]:
        """Return OpenAI tool definitions (Chat Completions or Responses API)."""
        return openai_tools(format=format, strict=strict)

    def anthropic_tools(self) -> list[dict[str, Any]]:
        """Return Anthropic Messages API tool definitions."""
        return anthropic_tools()

    def mcp_tools(self) -> list[dict[str, Any]]:
        """Return native MCP ``tools/list`` definitions."""
        return copy.deepcopy(TOOL_DEFINITIONS)

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any] | str | None = None,
        *,
        raise_on_error: bool = True,
    ) -> dict[str, Any]:
        """Execute a flashmp tool by name with a dict or JSON-encoded string of arguments.

        Parameters
        ----------
        name:
            Tool name (e.g. ``"ts_digest"``, ``"ts_drilldown"``, ``"ts_leak_scan"``).
        arguments:
            Tool arguments as a dict/mapping, JSON object string (as emitted by OpenAI
            tool calls), or None.
        raise_on_error:
            If True (default), re-raises sandbox/validation/runtime errors (e.g.
            ``PermissionError`` on path traversal). If False, returns
            ``{"is_error": True, "error": "..."}``.
        """
        try:
            parsed_args = self._parse_arguments(arguments)
            payload = self._server._call_tool(str(name), parsed_args)
            return _sanitize_json(payload)
        except Exception as exc:
            if raise_on_error:
                raise
            return {"is_error": True, "error": str(exc)}

    def execute_json(
        self,
        name: str,
        arguments: Mapping[str, Any] | str | None = None,
    ) -> str:
        """Execute a tool and return a strict RFC-8259 JSON string for LLM tool results."""
        result = self.execute(name, arguments, raise_on_error=False)
        return json.dumps(result, allow_nan=False)

    @staticmethod
    def _parse_arguments(arguments: Mapping[str, Any] | str | None) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, str):
            raw = arguments.strip()
            if not raw:
                return {}
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON tool arguments: {exc}") from exc
            if not isinstance(decoded, dict):
                raise TypeError("Tool arguments JSON must decode to an object")
            return decoded
        if isinstance(arguments, Mapping):
            return dict(arguments)
        raise TypeError(f"Tool arguments must be a dict, JSON string, or None; got {type(arguments).__name__}")

    def close(self) -> None:
        """Release all native RingMP/SwarmMonitor session handles."""
        self._server.sessions.close_all()

    def __enter__(self) -> AgentToolkit:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"AgentToolkit(sessions={len(self._server.sessions)}, tools={len(TOOL_DEFINITIONS)})"


# ---------------------------------------------------------------------------
# 2. MCP Client Configuration Generator
# ---------------------------------------------------------------------------


def _toml_string(value: str) -> str:
    """Encode a Python string as a valid TOML basic string."""
    return json.dumps(value, ensure_ascii=False)


def mcp_client_config(
    client: str = "claude-code",
    allowed_dirs: Sequence[str | Path] | str | Path | None = None,
    *,
    command: str = "flashmp-mcp",
) -> str:
    """Generate ready-to-paste JSON or TOML MCP configuration for popular AI coding clients.

    Parameters
    ----------
    client:
        One of ``"claude-code"``, ``"cursor"``, ``"codex"``, or ``"gemini"``.
    allowed_dirs:
        Optional directory (or sequence of directories) exposed to the sandboxed MCP server
        via ``FLASHMP_MCP_ALLOWED_DIRS``.
    command:
        Executable command name or path for the MCP server (default ``"flashmp-mcp"``).
    """
    key = str(client).strip().lower()
    if key not in _SUPPORTED_MCP_CLIENTS:
        raise ValueError(
            f"Unsupported MCP client {client!r}; supported clients: {', '.join(_SUPPORTED_MCP_CLIENTS)}"
        )

    norm_dirs = _normalize_allowed_dirs(allowed_dirs)
    env_dirs: str | None = None
    if norm_dirs is not None:
        resolved = [str(Path(d).expanduser().resolve()) for d in norm_dirs]
        env_dirs = os.pathsep.join(resolved)

    if key == "codex":
        lines = [
            "[mcp_servers.flashmp]",
            f"command = {_toml_string(command)}",
            "args = []",
        ]
        if env_dirs is not None:
            lines.extend(
                [
                    "",
                    "[mcp_servers.flashmp.env]",
                    f"FLASHMP_MCP_ALLOWED_DIRS = {_toml_string(env_dirs)}",
                ]
            )
        return "\n".join(lines) + "\n"

    server_entry: dict[str, Any] = {
        "command": command,
        "args": [],
    }
    if env_dirs is not None:
        server_entry["env"] = {"FLASHMP_MCP_ALLOWED_DIRS": env_dirs}

    payload = {"mcpServers": {"flashmp": server_entry}}
    return json.dumps(payload, indent=2) + "\n"


# ---------------------------------------------------------------------------
# 3. 1-Command Agent Skill Installer
# ---------------------------------------------------------------------------


def _locate_skill_md(skill_source: str | Path | None = None) -> Path:
    if skill_source is not None:
        src = Path(skill_source).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Skill file not found at {src}")
        return src

    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "SKILL.md",
        pkg_dir.parent / "SKILL.md",
        pkg_dir.parents[1] / "SKILL.md" if len(pkg_dir.parents) >= 2 else pkg_dir / "SKILL.md",
        Path.cwd() / "SKILL.md",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "Could not locate flashmp SKILL.md (checked package directory, repository root, and cwd)"
    )


def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".skill_", suffix=".tmp", dir=str(dest.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp_name, dest)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def install_skill(
    target: str = "claude",
    dest_dir: str | Path | None = None,
    *,
    skill_source: str | Path | None = None,
) -> list[Path]:
    """Install flashmp's ``SKILL.md`` agent playbook into Claude/Cursor/Codex skill directories.

    Parameters
    ----------
    target:
        ``"claude"``, ``"cursor"``, ``"codex"``, or ``"all"``.
    dest_dir:
        Optional override root directory (useful for project-local installs or testing).
        When set with a single target, installs to ``<dest_dir>/flashmp/SKILL.md``.
        When set with ``target="all"``, installs into ``.claude/``, ``.cursor/``, and
        ``.codex/`` subdirectories under ``<dest_dir>``.
    skill_source:
        Optional explicit path to a ``SKILL.md`` file.

    Returns
    -------
    list[Path]
        Absolute paths of all installed ``SKILL.md`` files.
    """
    tgt = str(target).strip().lower()
    valid_targets = (*_SKILL_TARGETS, "all")
    if tgt not in valid_targets:
        raise ValueError(
            f"Unsupported skill target {target!r}; expected one of {', '.join(valid_targets)}"
        )

    src_path = _locate_skill_md(skill_source)
    content = src_path.read_bytes()

    destinations: list[Path] = []
    if dest_dir is None:
        home = Path.home()
        targets_to_install = _SKILL_TARGETS if tgt == "all" else (tgt,)
        for name in targets_to_install:
            destinations.append(home / f".{name}" / "skills" / "flashmp" / "SKILL.md")
    else:
        base = Path(dest_dir).expanduser().resolve()
        if tgt == "all":
            for name in _SKILL_TARGETS:
                destinations.append(base / f".{name}" / "skills" / "flashmp" / "SKILL.md")
        else:
            destinations.append(base / "flashmp" / "SKILL.md")

    installed: list[Path] = []
    for dest in destinations:
        _atomic_write_bytes(dest, content)
        installed.append(dest.resolve())
    return installed


# ---------------------------------------------------------------------------
# 4. Duck-Typed DataFrame / Series / Array Extraction (Zero pandas/polars Import)
# ---------------------------------------------------------------------------


def _get_loaded_numpy() -> Any | None:
    return sys.modules.get("numpy")


def _coerce_real_float(val: Any) -> float:
    if isinstance(val, (bool, complex, str, bytes, bytearray)) or val is None:
        raise TypeError(f"Expected real numeric value, got {type(val).__name__}: {val!r}")
    return float(val)


def _check_numpy_real_1d(arr: Any) -> Any:
    np = _get_loaded_numpy()
    kind = arr.dtype.kind
    if kind in ("c", "S", "U", "V", "M", "m"):
        raise TypeError(f"Unsupported non-real dtype {arr.dtype}; expected real numeric values")
    if kind == "b":
        raise TypeError("Boolean arrays are not supported as numeric time series")
    if kind == "O":
        for v in arr.flat:
            if isinstance(v, (bool, complex, str, bytes, bytearray)) or v is None:
                raise TypeError(f"Object array contains non-real element {v!r}")
    if arr.dtype == np.float64 and arr.flags.c_contiguous and arr.flags.aligned:
        return arr
    out = np.require(arr, dtype=np.float64, requirements=("C", "A"))
    if not out.flags.aligned:
        out = np.array(out, dtype=np.float64, order="C", copy=True)
    return out


def _is_numeric_dtype(dtype: Any) -> bool | None:
    """Duck-type numeric check for NumPy/Pandas/Polars dtype objects. Returns None if inconclusive."""
    if dtype is None:
        return None
    kind = getattr(dtype, "kind", None)
    if isinstance(kind, str) and len(kind) == 1:
        if kind == "O":
            return None  # object column could still hold floats or strings; inspect sample
        return kind in ("i", "u", "f")

    is_numeric_fn = getattr(dtype, "is_numeric", None)
    if callable(is_numeric_fn):
        try:
            return bool(is_numeric_fn())
        except Exception:
            pass

    dname = str(dtype).strip().lower()
    if not dname:
        return None
    for non_num in ("bool", "str", "utf", "date", "time", "dur", "cat", "bin", "complex", "list", "struct", "null"):
        if non_num in dname:
            return False
    if "object" in dname:
        return None
    for num_prefix in ("float", "int", "uint", "decimal", "f4", "f8", "i4", "i8", "u4", "u8"):
        if dname.startswith(num_prefix):
            return True
    return None


def _sample_is_numeric(col_obj: Any, limit: int = 16) -> bool:
    """Check in O(1) whether up to `limit` items from a column/sequence are real numbers."""
    if isinstance(col_obj, (str, bytes, bytearray, bool)):
        return False
    if isinstance(col_obj, array.array):
        return col_obj.typecode in ("d", "f", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q")

    it = None
    try:
        it = iter(col_obj)
    except TypeError:
        seq: Any = None
        if hasattr(col_obj, "to_list") and callable(col_obj.to_list):
            seq = col_obj.to_list()
        elif hasattr(col_obj, "tolist") and callable(col_obj.tolist):
            seq = col_obj.tolist()
        elif hasattr(col_obj, "values") and not callable(col_obj.values):
            seq = col_obj.values
        if seq is None or isinstance(seq, (str, bytes, bytearray, bool)):
            return False
        try:
            it = iter(seq)
        except TypeError:
            return False

    count = 0
    for item in it:
        if isinstance(item, (bool, complex, str, bytes, bytearray)) or item is None:
            return False
        if not isinstance(item, (int, float)):
            try:
                float(item)
            except (TypeError, ValueError):
                return False
        count += 1
        if count >= limit:
            break
    return count > 0


def _is_numeric_column(col_obj: Any) -> bool:
    dtype = getattr(col_obj, "dtype", None)
    verdict = _is_numeric_dtype(dtype)
    if verdict is not None:
        return verdict
    return _sample_is_numeric(col_obj)


def _df_get_column(df: Any, col_name: Any) -> Any:
    getter = getattr(df, "get_column", None)
    if callable(getter):
        return getter(col_name)
    return df[col_name]


def _is_dataframe_like(obj: Any) -> bool:
    if isinstance(obj, (dict, Mapping, list, tuple, str, bytes, bytearray, array.array)):
        return False
    np = _get_loaded_numpy()
    if np is not None and isinstance(obj, np.ndarray):
        return False
    return hasattr(obj, "columns") and (
        hasattr(obj, "__getitem__") or callable(getattr(obj, "get_column", None))
    )


def _extract_from_dataframe(df: Any, column: str | int | None) -> Any:
    cols = list(df.columns)
    if not cols:
        raise ValueError("DataFrame has no columns")

    if column is not None:
        if isinstance(column, bool):
            raise TypeError("column must be a column name or integer index, not bool")
        if column in cols:
            return _df_get_column(df, column)
        if isinstance(column, int):
            if -len(cols) <= column < len(cols):
                return _df_get_column(df, cols[column])
            raise IndexError(
                f"Column index {column} out of bounds for DataFrame with {len(cols)} columns"
            )
        raise KeyError(f"Column {column!r} not found in DataFrame")

    numeric_cols = [c for c in cols if _is_numeric_column(_df_get_column(df, c))]
    if not numeric_cols:
        raise ValueError("No numeric column found in DataFrame")

    chosen = numeric_cols[0]
    if len(numeric_cols) >= 2 and str(chosen).strip().lower() in _TIME_LIKE_COLUMNS:
        chosen = numeric_cols[1]
    return _df_get_column(df, chosen)


def _extract_1d_numeric(obj: Any, *, _depth: int = 0) -> Any:
    if _depth > 4:
        raise TypeError(f"Could not extract 1D numeric series from {type(obj).__name__}")

    if isinstance(obj, (str, bytes, bytearray, bool, int, float, complex)) or obj is None:
        raise TypeError(f"Expected 1D numeric series, got {type(obj).__name__}")

    # Fast zero-copy path for stdlib array.array
    if isinstance(obj, array.array):
        if obj.typecode == "d":
            return obj
        if obj.typecode in ("f", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"):
            return array.array("d", obj)
        raise TypeError(f"Unsupported array.array typecode {obj.typecode!r}")

    # Fast zero-copy path for already-loaded NumPy arrays
    np = _get_loaded_numpy()
    if np is not None and isinstance(obj, np.ndarray):
        if obj.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {obj.shape}")
        return _check_numpy_real_1d(obj)

    # Fast path for plain Python lists/tuples
    if isinstance(obj, (list, tuple)):
        if any(isinstance(x, (bool, complex, str, bytes, bytearray)) or x is None for x in obj):
            raise TypeError("Time series must contain only real numeric values")
        return array.array("d", obj)

    # Duck-typed Series objects (pandas.Series, polars.Series, or custom mocks)
    dtype = getattr(obj, "dtype", None)
    verdict = _is_numeric_dtype(dtype)
    if verdict is False:
        raise TypeError(f"Unsupported non-numeric Series dtype {dtype!r}")

    to_numpy = getattr(obj, "to_numpy", None)
    if callable(to_numpy):
        try:
            arr = to_numpy()
            if arr is not obj:
                return _extract_1d_numeric(arr, _depth=_depth + 1)
        except ImportError:
            # E.g. Polars Series.to_numpy() when numpy is not installed: fall through to to_list().
            pass

    to_list = getattr(obj, "to_list", None)
    if callable(to_list):
        vals = to_list()
        if vals is not obj:
            return _extract_1d_numeric(vals, _depth=_depth + 1)

    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        vals = tolist()
        if vals is not obj:
            return _extract_1d_numeric(vals, _depth=_depth + 1)

    if hasattr(obj, "values") and not callable(obj.values):
        vals = obj.values
        if vals is not obj:
            return _extract_1d_numeric(vals, _depth=_depth + 1)

    if np is not None and hasattr(obj, "__array__"):
        arr = np.asarray(obj)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {arr.shape}")
        return _check_numpy_real_1d(arr)

    return array.array("d", (_coerce_real_float(x) for x in obj))


def extract_series(obj: Any, column: str | int | None = None) -> Any:
    """Extract a contiguous, aligned 1D float64 series from common Python/DataFrame containers.

    Zero third-party imports: handles ``pandas.Series``, ``pandas.DataFrame``,
    ``polars.Series``, ``polars.DataFrame``, ``numpy.ndarray``, ``dict``, ``list``,
    ``tuple``, and ``array.array`` strictly via duck-typing.

    Returns a zero-copy C-contiguous aligned ``float64`` ``numpy.ndarray`` when backed by
    NumPy, or a zero-dependency ``array.array('d')`` otherwise.
    """
    if isinstance(obj, (str, bytes, bytearray, bool, int, float, complex)) or obj is None:
        raise TypeError(f"Expected a time-series or DataFrame container, got {type(obj).__name__}")

    # 1. Duck-typed DataFrame (pandas.DataFrame, polars.DataFrame, or compatible mock)
    if _is_dataframe_like(obj):
        col_series = _extract_from_dataframe(obj, column)
        return _extract_1d_numeric(col_series)

    # 2. Mapping / dict of named series
    if isinstance(obj, Mapping):
        if not obj:
            raise ValueError("Mapping is empty")
        if column is not None:
            if isinstance(column, bool):
                raise TypeError("column must be a key name or integer, not bool")
            if column not in obj:
                raise KeyError(f"Column/key {column!r} not found in mapping")
            return _extract_1d_numeric(obj[column])

        numeric_keys = [k for k, v in obj.items() if _is_numeric_column(v)]
        if not numeric_keys:
            raise ValueError("Mapping contains no numeric series")
        chosen = numeric_keys[0]
        if len(numeric_keys) >= 2 and str(chosen).strip().lower() in _TIME_LIKE_COLUMNS:
            chosen = numeric_keys[1]
        return _extract_1d_numeric(obj[chosen])

    # 3. Already-loaded NumPy array (1D or 2D with column selection)
    np = _get_loaded_numpy()
    if np is not None and isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            if column is not None:
                raise ValueError("column cannot be specified for a 1D numpy.ndarray")
            return _check_numpy_real_1d(obj)
        if obj.ndim == 2:
            if column is None:
                if obj.shape[1] == 1:
                    return _check_numpy_real_1d(obj[:, 0])
                if obj.shape[0] == 1:
                    return _check_numpy_real_1d(obj[0, :])
                raise ValueError(
                    f"2D numpy.ndarray with shape {obj.shape} requires an integer column index"
                )
            if isinstance(column, bool) or not isinstance(column, int):
                raise TypeError("column must be an integer index for a 2D numpy.ndarray")
            return _check_numpy_real_1d(obj[:, column])
        raise ValueError(f"Expected 1D or 2D numpy.ndarray, got ndim={obj.ndim}")

    # 4. 1D Series (pandas.Series, polars.Series), array.array, list, tuple, iterable
    if column is not None:
        series_name = getattr(obj, "name", None)
        if isinstance(column, bool):
            raise TypeError("column must be a column name or integer, not bool")
        if series_name is not None and column != series_name and column != 0:
            raise KeyError(f"Requested column {column!r} does not match Series name {series_name!r}")
        if series_name is None and column != 0:
            raise ValueError(
                f"Cannot select column {column!r} from 1D {type(obj).__name__} without column names"
            )
    return _extract_1d_numeric(obj)
