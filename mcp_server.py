#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-tools", json_response=True)

ROOT = Path(os.environ.get("MCP_ROOT", ".")).resolve()

MAX_READ_BYTES = int(os.environ.get("MCP_MAX_READ_BYTES", "262144"))  # 256 KiB
MAX_WRITE_BYTES = int(os.environ.get("MCP_MAX_WRITE_BYTES", "1048576"))  # 1 MiB
MAX_CMD_CHARS = int(os.environ.get("MCP_MAX_CMD_CHARS", "4000"))
ALLOW_DANGEROUS = os.environ.get("MCP_ALLOW_DANGEROUS", "0") == "1"


def _root() -> Path:
    return Path(os.environ.get("MCP_ROOT", ".")).resolve()

def _safe_relpath(path: str) -> Path:
    root = _root()

    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")

    p = Path(path)
    if path.startswith(("/", "~")):
        return p

    if any(part == ".." for part in p.parts):
        raise ValueError(".. is not allowed in path")

    full = (root / p).resolve()
    if not str(full).startswith(str(root)):
        raise ValueError("path escapes MCP_ROOT")
    return full

def _dangerous_cmd(cmd: str) -> Optional[str]:
    s = cmd.strip()
    if re.search(r"\brm\s+-rf\b", s):
        return "rm -rf"
    if re.search(r"\bmkfs\.", s) or re.search(r"\bdd\b\s+if=", s):
        return "mkfs/dd"
    if re.search(r":\(\)\s*\{\s*:\s*\|\s*:\s*;\s*\}\s*;\s*:", s):
        return "fork bomb"
    return None


@mcp.tool()
def read_text(path: str) -> Dict[str, Any]:
    """Read a UTF-8 text file. Size-limited."""
    full = _safe_relpath(path)
    data = full.read_bytes()
    if len(data) > MAX_READ_BYTES:
        raise ValueError(f"file too large ({len(data)} bytes > {MAX_READ_BYTES})")
    return {"path": path, "content": data.decode("utf-8", errors="replace")}


@mcp.tool()
def write_text(path: str, content: str, overwrite: bool = True) -> Dict[str, Any]:
    """Write a UTF-8 text file. Creates parent dirs. Size-limited."""
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    raw = content.encode("utf-8")
    if len(raw) > MAX_WRITE_BYTES:
        raise ValueError(f"content too large ({len(raw)} bytes > {MAX_WRITE_BYTES})")

    full = _safe_relpath(path)
    full.parent.mkdir(parents=True, exist_ok=True)

    if full.exists() and not overwrite:
        raise ValueError("file exists and overwrite=False")

    full.write_bytes(raw)
    return {"ok": True, "path": path, "bytes_written": len(raw)}


@mcp.tool()
def apply_unified_patch(diff: str, strip: int = 0) -> Dict[str, Any]:
    """Apply a unified diff using patch(1). diff must only reference files."""
    if not isinstance(diff, str) or not diff.strip():
        raise ValueError("diff must be a non-empty string")
    if strip not in (0, 1):
        raise ValueError("strip must be 0 or 1")

    try:
        subprocess.run(["patch", "--version"], capture_output=True, text=True, timeout=5)
    except Exception:
        raise RuntimeError("patch(1) not found; install 'patch'")

    r = subprocess.run(
        ["patch", f"-p{strip}", "--forward", "--batch"],
        cwd=str(ROOT),
        input=diff,
        capture_output=True,
        text=True,
        timeout=30,
    )
    ret = {
        "ok": r.returncode == 0,
        "returncode": r.returncode,
        "stdout": r.stdout,
        "stderr": r.stderr,
    }
    return ret


@mcp.tool()
def run_bash(command: str, timeout_s: int = 300) -> Dict[str, Any]:
    """Run a bash command in MCP_ROOT."""
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    if len(command) > MAX_CMD_CHARS:
        raise ValueError(f"command too long ({len(command)} chars > {MAX_CMD_CHARS})")

    bad = _dangerous_cmd(command)
    if bad and not ALLOW_DANGEROUS:
        raise ValueError(f"dangerous command blocked: {bad} (set MCP_ALLOW_DANGEROUS=1 to override)")

    r = subprocess.run(
        ["bash", "-lc", command],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return {"command": command, "cwd": str(ROOT), "stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode}


def main() -> None:
    # Direct execution: stdio server (this is the “simple run it as a script” mode). :contentReference[oaicite:1]{index=1}
    mcp.run()


if __name__ == "__main__":
    main()