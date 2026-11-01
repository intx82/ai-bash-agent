#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import fnmatch
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-tools", json_response=True)

MAX_READ_BYTES = int(os.environ.get("MCP_MAX_READ_BYTES", "262144"))  # 256 KiB
MAX_WRITE_BYTES = int(os.environ.get("MCP_MAX_WRITE_BYTES", "1048576"))  # 1 MiB per text write/append
MAX_MERGE_BYTES = int(os.environ.get("MCP_MAX_MERGE_BYTES", "268435456"))  # 256 MiB per merged file
MAX_GREP_FILE_BYTES = int(os.environ.get("MCP_MAX_GREP_FILE_BYTES", "0"))  # 0: no per-file limit
MAX_GREP_TOTAL_BYTES = int(os.environ.get("MCP_MAX_GREP_TOTAL_BYTES", "0"))  # 0: no aggregate limit
GREP_TIMEOUT_S = max(1, min(240, int(os.environ.get("MCP_GREP_TIMEOUT_S", "30"))))
GREP_MAX_CONCURRENT = max(1, int(os.environ.get("MCP_GREP_MAX_CONCURRENT", "2")))
_grep_slots = asyncio.Semaphore(GREP_MAX_CONCURRENT)
MAX_CMD_CHARS = int(os.environ.get("MCP_MAX_CMD_CHARS", "8000"))
ALLOW_DANGEROUS = os.environ.get("MCP_ALLOW_DANGEROUS", "0") == "1"


def _root() -> Path:
    return Path(os.environ.get("MCP_ROOT", ".")).resolve()

def _safe_relpath(path: str) -> Path:
    root = _root()

    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")

    p = Path(path)
    # Preserve existing support for absolute paths; expand ~/... correctly.
    if p.is_absolute() or path.startswith("~"):
        return p.expanduser().resolve()

    if any(part == ".." for part in p.parts):
        raise ValueError(".. is not allowed in path")

    full = (root / p).resolve()
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
def read_text(
    path: str, start_line: int = 1, end_line: Optional[int] = None,
    count: Optional[int] = None,
) -> Dict[str, Any]:
    """Read UTF-8 text, optionally selecting 1-based lines (end_line inclusive).

    Use either end_line or count, not both. Without a range, read the whole file
    as before. MCP_MAX_READ_BYTES limits the RETURNED data, so small ranges can
    be read even when the file itself is much larger. Newline bytes are preserved.
    Ranged responses provide next_start_line for convenient paging.
    """
    if not isinstance(start_line, int) or isinstance(start_line, bool) or start_line < 1:
        raise ValueError("start_line must be a positive 1-based integer")
    if end_line is not None:
        if not isinstance(end_line, int) or isinstance(end_line, bool) or end_line < start_line:
            raise ValueError("end_line must be an integer >= start_line (inclusive)")
    if count is not None:
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("count must be a positive integer")
    if end_line is not None and count is not None:
        raise ValueError("specify end_line or count, not both")

    full = _safe_relpath(path)
    ranged = start_line != 1 or end_line is not None or count is not None
    with full.open("rb") as f:
        if not ranged:
            data = f.read(MAX_READ_BYTES + 1)
            if len(data) > MAX_READ_BYTES:
                raise ValueError(
                    f"file exceeds MCP_MAX_READ_BYTES ({MAX_READ_BYTES} bytes); "
                    "use start_line with end_line or count to read a smaller range"
                )
            return {"path": path, "content": data.decode("utf-8", errors="replace")}

        def empty_result() -> Dict[str, Any]:
            return {
                "path": path, "content": "", "start_line": start_line,
                "end_line": None, "lines_read": 0,
                "has_more": False, "next_start_line": None,
            }

        # Skip lines using bounded reads; an unusually long line must not
        # consume unbounded memory merely because it precedes the selected range.
        for _ in range(start_line - 1):
            while True:
                part = f.readline(65536)
                if not part:
                    return empty_result()
                if part.endswith(b"\n"):
                    break

        limit = count if count is not None else (
            end_line - start_line + 1 if end_line is not None else None
        )
        pieces = []
        bytes_read = 0
        lines_read = 0
        while limit is None or lines_read < limit:
            line = f.readline(MAX_READ_BYTES - bytes_read + 1)
            if not line:
                break
            if bytes_read + len(line) > MAX_READ_BYTES:
                raise ValueError(
                    f"requested range exceeds MCP_MAX_READ_BYTES ({MAX_READ_BYTES} bytes); "
                    "request fewer lines or increase MCP_MAX_READ_BYTES"
                )
            pieces.append(line)
            bytes_read += len(line)
            lines_read += 1

        has_more = bool(f.read(1)) if limit is not None and lines_read == limit else False
        return {
            "path": path,
            "content": b"".join(pieces).decode("utf-8", errors="replace"),
            "start_line": start_line,
            "end_line": start_line + lines_read - 1 if lines_read else None,
            "lines_read": lines_read,
            "has_more": has_more,
            "next_start_line": start_line + lines_read if has_more else None,
        }


@mcp.tool()
def list_files(
    path: str = ".", recursive: bool = False, include_hidden: bool = False,
    max_entries: int = 200,
) -> Dict[str, Any]:
    """List directory entries (like ls), with types and sizes. No shell required.

    Absolute paths and ~/ paths work. Directories are not followed through
    symlinks; results are bounded to avoid filling the LLM context window.
    """
    if not isinstance(max_entries, int) or isinstance(max_entries, bool) or not 1 <= max_entries <= 2000:
        raise ValueError("max_entries must be between 1 and 2000")
    full = _safe_relpath(path)
    if not full.exists() and not full.is_symlink():
        raise FileNotFoundError(str(full))

    entries: list[Dict[str, Any]] = []
    scanned = 0
    truncated = False

    def add_entry(entry: Path) -> None:
        try:
            info = entry.lstat()
        except OSError:
            return
        if stat.S_ISLNK(info.st_mode):
            kind = "symlink"
        elif stat.S_ISDIR(info.st_mode):
            kind = "directory"
        elif stat.S_ISREG(info.st_mode):
            kind = "file"
        else:
            kind = "other"
        entries.append({"name": entry.name, "path": str(entry), "type": kind,
                        "size": info.st_size if kind == "file" else None})

    if not full.is_dir():
        add_entry(full)
    else:
        # Limit traversal, even if recursive directories contain millions of entries.
        pending = [full]
        while pending and len(entries) < max_entries and scanned < 20000:
            directory = pending.pop()
            with os.scandir(directory) as iterator:
                candidates = []
                for item in iterator:
                    scanned += 1
                    if scanned > 20000:
                        truncated = True
                        break
                    if include_hidden or not item.name.startswith("."):
                        candidates.append(item)
            candidates.sort(key=lambda item: item.name)
            subdirs = []
            for item in candidates:
                if len(entries) >= max_entries:
                    truncated = True
                    break
                add_entry(Path(item.path))
                if recursive and item.is_dir(follow_symlinks=False):
                    subdirs.append(Path(item.path))
            if recursive:
                pending.extend(reversed(subdirs))
            if pending and len(entries) >= max_entries:
                truncated = True
        if scanned >= 20000:
            truncated = True
    return {"ok": True, "path": path, "entries": entries,
            "count": len(entries), "truncated": truncated}


def _grep_files_impl(
    pattern: str, path: str = ".", recursive: bool = True,
    ignore_case: bool = False, regex: bool = False, file_glob: str = "*",
    max_matches: int = 100, max_files: int = 0, include_hidden: bool = False,
    timeout_s: int = GREP_TIMEOUT_S,
) -> Dict[str, Any]:
    """Search text files recursively for a SUBSTRING (or a regex with regex=True).

    Like grep -r for source trees: search all eligible files by default, including
    files after the first 500.  max_files=0 means unlimited.  Text is processed
    line by line, not loaded into memory.  The optional environment limits
    MCP_MAX_GREP_FILE_BYTES and MCP_MAX_GREP_TOTAL_BYTES default to 0 (unlimited).

    max_matches caps the returned lines, not the number of files examined by
    default.  If a limit/error stops coverage, complete=False, truncated=True,
    and truncation_reasons explicitly describe why.  Hidden directories/files
    and binary files are skipped unless configured otherwise.
    """
    if not isinstance(pattern, str) or not pattern or len(pattern) > 512:
        raise ValueError("pattern must be a non-empty string of at most 512 characters")
    if not isinstance(file_glob, str) or not file_glob:
        raise ValueError("file_glob must be non-empty")
    if not isinstance(max_matches, int) or isinstance(max_matches, bool) or not 1 <= max_matches <= 1000:
        raise ValueError("max_matches must be between 1 and 1000")
    if not isinstance(max_files, int) or isinstance(max_files, bool) or not 0 <= max_files <= 1000000:
        raise ValueError("max_files must be between 0 (unlimited) and 1000000")
    if not isinstance(include_hidden, bool):
        raise ValueError("include_hidden must be a boolean")
    if (not isinstance(timeout_s, int) or isinstance(timeout_s, bool)
            or not 1 <= timeout_s <= 240):
        raise ValueError("timeout_s must be between 1 and 240 seconds")

    started = time.monotonic()
    deadline = started + timeout_s

    full = _safe_relpath(path)
    if not full.exists():
        raise FileNotFoundError(str(full))
    if not (full.is_file() or full.is_dir()):
        raise ValueError(f"path is not a regular file or directory: {full}")

    compiled = re.compile(pattern, re.IGNORECASE) if regex else None
    needle = pattern.casefold() if ignore_case else pattern
    matches: list[Dict[str, Any]] = []
    files_searched = 0
    files_matched = 0
    files_skipped_binary = 0
    files_skipped_large = 0
    files_skipped_error = 0
    bytes_searched = 0
    reasons: set[str] = set()

    def on_walk_error(error: OSError) -> None:
        nonlocal files_skipped_error
        files_skipped_error += 1
        reasons.add("directory_error")

    def candidate_files():
        if time.monotonic() >= deadline:
            reasons.add("timeout")
            return
        if full.is_file():
            # Explicit file searches do not require a matching file_glob.
            yield full
            return
        if not recursive:
            with os.scandir(full) as entries:
                for ent in sorted(entries, key=lambda item: item.name):
                    if time.monotonic() >= deadline:
                        reasons.add("timeout")
                        return
                    if not include_hidden and ent.name.startswith("."):
                        continue
                    if ent.is_file(follow_symlinks=False) and fnmatch.fnmatchcase(ent.name, file_glob):
                        yield Path(ent.path)
            return
        for directory, subdirs, filenames in os.walk(
            full, topdown=True, followlinks=False, onerror=on_walk_error
        ):
            if time.monotonic() >= deadline:
                reasons.add("timeout")
                return
            # Sorted traversal is reproducible; never recurse via symlink dirs.
            subdirs[:] = sorted(
                d for d in subdirs
                if (include_hidden or not d.startswith("."))
                and not Path(directory, d).is_symlink()
            )
            for name in sorted(filenames):
                if time.monotonic() >= deadline:
                    reasons.add("timeout")
                    return
                if not include_hidden and name.startswith("."):
                    continue
                if not fnmatch.fnmatchcase(name, file_glob):
                    continue
                candidate = Path(directory, name)
                if candidate.is_symlink():
                    continue
                yield candidate

    for current in candidate_files():
        if time.monotonic() >= deadline:
            reasons.add("timeout")
            break
        if max_files and files_searched >= max_files:
            reasons.add("max_files")
            break
        try:
            size = current.stat().st_size
            if MAX_GREP_FILE_BYTES and size > MAX_GREP_FILE_BYTES:
                files_skipped_large += 1
                reasons.add("max_file_bytes")
                continue
            if MAX_GREP_TOTAL_BYTES and bytes_searched + size > MAX_GREP_TOTAL_BYTES:
                reasons.add("max_total_bytes")
                break

            with current.open("rb") as stream:
                # Do not interpret binary/compiled objects as text source files.
                sample = stream.read(8192)
                if b"\0" in sample:
                    files_skipped_binary += 1
                    continue
                stream.seek(0)
                files_searched += 1
                matched_this_file = False
                number = 0
                while True:
                    if time.monotonic() >= deadline:
                        reasons.add("timeout")
                        break
                    # Bounded readline prevents an immense/minified single line
                    # from exhausting memory in the worker subprocess.
                    raw_line = stream.readline(1024 * 1024 + 1)
                    if not raw_line:
                        break
                    number += 1
                    bytes_searched += len(raw_line)
                    if len(raw_line) > 1024 * 1024:
                        # The line is too long to check completely. Drain it in
                        # bounded chunks and report incomplete coverage.
                        reasons.add("max_line_bytes")
                        while raw_line and not raw_line.endswith(b"\n"):
                            if time.monotonic() >= deadline:
                                reasons.add("timeout")
                                break
                            raw_line = stream.readline(65536)
                            bytes_searched += len(raw_line)
                        if "timeout" in reasons:
                            break
                        continue
                    # rstrip only line terminators, preserving whitespace.
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if compiled is not None:
                        match = compiled.search(line)
                        match_pos = match.start() if match else -1
                    else:
                        match_pos = (line.casefold() if ignore_case else line).find(needle)
                    if match_pos >= 0:
                        if not matched_this_file:
                            files_matched += 1
                            matched_this_file = True
                        # For exceptionally long source lines, include the actual
                        # matching region instead of returning only the first 500
                        # characters (which could hide the matching text).
                        if len(line) > 500:
                            start = max(0, min(match_pos - 100, len(line) - 500))
                            excerpt = ("…" if start else "") + line[start:start + 500]
                            if start + 500 < len(line):
                                excerpt += "…"
                        else:
                            excerpt = line
                        matches.append({"path": str(current), "line": number, "text": excerpt})
                        if len(matches) >= max_matches:
                            reasons.add("max_matches")
                            break
        except OSError:
            files_skipped_error += 1
            reasons.add("file_error")
            continue
        if len(matches) >= max_matches or "timeout" in reasons:
            break

    complete = not reasons
    result = {
        "ok": True, "pattern": pattern, "path": path,
        "matches": matches, "count": len(matches),
        "files_searched": files_searched, "files_matched": files_matched,
        "files_skipped": files_skipped_binary + files_skipped_large + files_skipped_error,
        "files_skipped_binary": files_skipped_binary,
        "files_skipped_large": files_skipped_large,
        "files_skipped_error": files_skipped_error,
        "bytes_searched": bytes_searched,
        "complete": complete, "truncated": not complete,
        "truncation_reasons": sorted(reasons),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "timeout_s": timeout_s,
    }
    if not complete:
        result["warning"] = (
            "Incomplete grep: count counts only reported matches, not all matches "
            "in the search path. Narrow the path/file_glob or adjust limits."
        )
    return result


@mcp.tool()
async def grep_files(
    pattern: str, path: str = ".", recursive: bool = True,
    ignore_case: bool = False, regex: bool = False, file_glob: str = "*",
    max_matches: int = 100, max_files: int = 0, include_hidden: bool = False,
    timeout_s: int = GREP_TIMEOUT_S,
) -> Dict[str, Any]:
    """Search file contents, with a per-search deadline (default 30 seconds).

    Uses substring matching unless regex=True. Runs in an isolated subprocess
    so expensive regexes, slow filesystems and cancelled searches cannot wedge
    the MCP server. timeout_s is 1..240 seconds, always less than the bridge's
    300-second timeout. When incomplete, complete=False and truncation_reasons
    describe why. If the subprocess hangs inside I/O, it is killed and the
    result explicitly reports a hard timeout (no negative match conclusion).
    """
    arguments = {
        "pattern": pattern, "path": path, "recursive": recursive,
        "ignore_case": ignore_case, "regex": regex, "file_glob": file_glob,
        "max_matches": max_matches, "max_files": max_files,
        "include_hidden": include_hidden, "timeout_s": timeout_s,
    }
    if (not isinstance(timeout_s, int) or isinstance(timeout_s, bool)
            or not 1 <= timeout_s <= 240):
        raise ValueError("timeout_s must be between 1 and 240 seconds")

    # Reject immediately when the worker pool is saturated. Waiting for a grep
    # slot must not consume the 300s bridge timeout either.
    try:
        await asyncio.wait_for(_grep_slots.acquire(), timeout=3)
    except asyncio.TimeoutError:
        return {
            "ok": False, "pattern": pattern, "path": path,
            "complete": False, "truncated": True,
            "truncation_reasons": ["busy"],
            "error": "Too many concurrent grep searches; try a narrower path",
        }

    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).resolve()), "--grep-worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(json.dumps(arguments).encode("utf-8")),
                timeout=timeout_s + 3,
            )
        except asyncio.TimeoutError:
            return {
                "ok": False, "pattern": pattern, "path": path,
                "complete": False, "truncated": True,
                "truncation_reasons": ["hard_timeout"],
                "timeout_s": timeout_s,
                "error": "grep subprocess was stopped after exceeding its deadline; "
                         "narrow the directory or file_glob",
            }
        if proc.returncode != 0:
            return {
                "ok": False, "pattern": pattern, "path": path,
                "complete": False, "truncated": True,
                "truncation_reasons": ["worker_error"],
                "error": stderr.decode("utf-8", errors="replace")[-2000:]
                         or "grep worker exited unsuccessfully",
            }
        return json.loads(stdout)
    finally:
        if proc is not None and proc.returncode is None:
            proc.kill()
            # Reap the killed process without delaying the MCP event loop.
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                pass
        _grep_slots.release()


def _grep_worker_main() -> None:
    """Private subprocess entry point; never emits progress on MCP stdout."""
    try:
        raw = sys.stdin.buffer.read(32769)
        if len(raw) > 32768:
            raise ValueError("grep request is too large")
        args = json.loads(raw)
        if not isinstance(args, dict):
            raise ValueError("grep request must be a JSON object")
        result = _grep_files_impl(**args)
        sys.stdout.write(json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        print(f"grep worker: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)


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

    # Exclusive creation prevents overwriting a concurrently created file.
    try:
        with full.open("wb" if overwrite else "xb") as f:
            f.write(raw)
    except FileExistsError as e:
        raise ValueError("file exists and overwrite=False") from e
    return {"ok": True, "path": path, "bytes_written": len(raw)}

@mcp.tool()
def append_text(path: str, content: str) -> Dict[str, Any]:
    """Append UTF-8 text to the end of a file (like >>). Create the file if missing.

    MCP_MAX_WRITE_BYTES limits the size of the appended text, not the existing file.
    No newline is added automatically.
    """
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    raw = content.encode("utf-8")
    if len(raw) > MAX_WRITE_BYTES:
        raise ValueError(f"content too large ({len(raw)} bytes > {MAX_WRITE_BYTES})")

    full = _safe_relpath(path)
    full.parent.mkdir(parents=True, exist_ok=True)

    # O_APPEND positions every write at EOF even if another writer is active.
    fd = os.open(full, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
    with os.fdopen(fd, "wb", buffering=0) as f:
        remaining = memoryview(raw)
        while remaining:
            written = f.write(remaining)
            if written is None or written <= 0:
                raise OSError("failed to append file contents")
            remaining = remaining[written:]
    return {"ok": True, "path": path, "bytes_written": len(raw)}


@mcp.tool()
def merge_files(paths: list[str], output_path: str, overwrite: bool = True) -> Dict[str, Any]:
    """Concatenate two or more files, in order, preserving all bytes (like cat).

    Binary-safe, streamed, and bounded by MCP_MAX_MERGE_BYTES. The output may
    not be any of the inputs. A temporary file prevents corrupt partial output.
    """
    if not isinstance(paths, list) or len(paths) < 2:
        raise ValueError("paths must be a list containing at least two files")

    sources = [_safe_relpath(path) for path in paths]
    output = _safe_relpath(output_path)

    for source in sources:
        if not source.is_file():
            raise ValueError(f"input is not a regular file: {source}")
        if source == output or (output.exists() and os.path.samefile(source, output)):
            raise ValueError("output_path must not refer to an input file")

    if output.exists():
        if not output.is_file():
            raise ValueError("output_path is not a regular file")
        if not overwrite:
            raise ValueError("file exists and overwrite=False")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    total = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output.name}.", suffix=".tmp",
            dir=output.parent, delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            for source in sources:
                with source.open("rb") as inp:
                    while chunk := inp.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_MERGE_BYTES:
                            raise ValueError(
                                f"merged file too large ({total} bytes > {MAX_MERGE_BYTES})"
                            )
                        tmp.write(chunk)

        # Preserve existing output permission bits on overwrite.
        if overwrite and output.exists():
            tmp_path.chmod(stat.S_IMODE(output.stat().st_mode))

        if overwrite:
            os.replace(tmp_path, output)
        else:
            # Atomic no-overwrite creation; cannot race with another creator.
            os.link(tmp_path, output)

        return {"ok": True, "paths": paths, "output_path": output_path,
                "bytes_written": total, "files_merged": len(paths)}
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@mcp.tool()
def replace_in_file(
    path: str, old: str, new: str, count: int = 1,
    occurrence: Optional[int] = None, line_number: Optional[int] = None,
    replace_all: bool = False,
) -> Dict[str, Any]:
    """Safely replace literal text in a UTF-8 file (not a regex).

    By default, exactly one unmodified occurrence must match; ambiguous edits
    are refused. Use occurrence=N (1-based), line_number=N (1-based starting
    line), or replace_all=True to explicitly identify the intended target(s).
    Legacy count=N>1 requires exactly N unmodified occurrences.

    If `old` occurs inside `new` (e.g. f( -> f(NULL,), the selected occurrence
    is recognized as already updated if it is inside an existing `new` string.
    This checks the selected location, NOT a stored edit history. For an
    explicitly selected line where `old` has disappeared, a unique `new`
    occurrence on that line is also recognized as the desired final state.

    Returns replaced_line / replaced_lines for successful edits. For a no-op
    on one identifiable location returns already_applied_line. It never reports
    unrelated lines that happen to contain the replacement text.
    """
    if not isinstance(old, str) or not old:
        raise ValueError("old must be a non-empty string")
    if not isinstance(new, str):
        raise ValueError("new must be a string")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    if occurrence is not None and (
        not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 1
    ):
        raise ValueError("occurrence must be a positive 1-based integer")
    if line_number is not None and (
        not isinstance(line_number, int) or isinstance(line_number, bool) or line_number < 1
    ):
        raise ValueError("line_number must be a positive 1-based integer")
    if not isinstance(replace_all, bool):
        raise ValueError("replace_all must be a boolean")
    if sum((occurrence is not None, line_number is not None, replace_all, count != 1)) > 1:
        raise ValueError("Choose only one selector: occurrence, line_number, replace_all, or count>1")

    full = _safe_relpath(path)
    # Decode bytes directly; keep CRLF intact (Path.read_text normalizes it).
    raw = full.read_bytes()
    text = raw.decode("utf-8")
    positions = []
    cursor = 0
    while (pos := text.find(old, cursor)) >= 0:
        positions.append(pos)
        cursor = pos + len(old)  # non-overlapping, like str.replace
    # Index start-line numbers once, avoiding O(file size * occurrences) scans.
    import bisect
    newline_positions = [i for i, ch in enumerate(text) if ch == "\n"]
    def at_line(pos: int) -> int:
        return bisect.bisect_left(newline_positions, pos) + 1

    def no_change(status: str, target_line: Optional[int] = None) -> Dict[str, Any]:
        return {
            "ok": True, "path": path, "replaced": 0, "status": status,
            "replaced_line": None, "replaced_lines": [], "replacements": [],
            "already_applied_line": target_line if status == "already_applied" else None,
            "lines": [],
        }

    if new == old:
        return no_change("unchanged")
    if not positions and occurrence is None and line_number is None:
        raise ValueError("old text not found in file; no changes made")

    # `old` can remain at the beginning of `new`. Prevent unintended replay
    # *at the selected occurrence*, e.g. foo( -> foo(NULL,.
    old_offsets = []
    if old in new:
        offset = 0
        while (found := new.find(old, offset)) >= 0:
            old_offsets.append(found)
            offset = found + len(old)

    def already_modified(pos: int) -> bool:
        return any(
            pos >= offset and text.startswith(new, pos - offset)
            for offset in old_offsets
        )

    eligible = [pos for pos in positions if not already_modified(pos)]
    selected: list[int]
    already_applied_line: Optional[int] = None

    if occurrence is not None:
        if occurrence > len(positions):
            raise ValueError(
                f"occurrence {occurrence} out of range: {len(positions)} "
                "current occurrences; use line_number or more context if an "
                "earlier replacement removed the old text"
            )
        pos = positions[occurrence - 1]
        if already_modified(pos):
            already_applied_line = at_line(pos)
            selected = []
        else:
            selected = [pos]
    elif line_number is not None:
        matches_on_line = [pos for pos in positions if at_line(pos) == line_number]
        if len(matches_on_line) > 1:
            raise ValueError(
                f"line {line_number} has {len(matches_on_line)} occurrences; "
                "use occurrence=N or include more surrounding text in old"
            )
        if matches_on_line:
            pos = matches_on_line[0]
            if already_modified(pos):
                already_applied_line = line_number
                selected = []
            else:
                selected = [pos]
        else:
            # `old` may have vanished entirely from this specific line. Only
            # recognize already_applied if `new` appears uniquely on that line;
            # this states the desired text is present, not that we edited it.
            hits = []
            if new:
                cursor = 0
                while (pos := text.find(new, cursor)) >= 0:
                    if at_line(pos) == line_number:
                        hits.append(pos)
                        if len(hits) > 1:
                            break
                    cursor = pos + len(new)
            if len(hits) == 1:
                return no_change("already_applied", line_number)
            raise ValueError(
                f"no unique old or already-updated text starting on line {line_number}; "
                "no changes made; re-read file and select the correct line"
            )
    elif replace_all:
        selected = eligible
        if not selected and positions:
            # replace_all explicitly specifies every occurrence as the target,
            # and all matches already contain `new`.
            return no_change("already_applied")
    else:
        if len(eligible) == count and eligible:
            selected = eligible
        elif count == 1 and not eligible and len(positions) == 1:
            # Only one identifiable location in this file could be the target.
            return no_change("already_applied", at_line(positions[0]))
        else:
            lines = [at_line(pos) for pos in eligible[:12]]
            raise ValueError(
                f"ambiguous replace: {len(eligible)} unmodified occurrence(s) "
                f"on lines {lines}{'...' if len(eligible)>12 else ''}, "
                f"{len(positions) - len(eligible)} occurrences already contain "
                f"the new text; count={count}. No changes made. "
                "Use a unique old string, occurrence=N, line_number=N, "
                "or replace_all=true."
            )

    if not selected:
        if already_applied_line is not None:
            return no_change("already_applied", already_applied_line)
        raise ValueError("old text not found at the selected location; no changes made")

    # Construct the complete replacement before touching the destination.
    chosen = set(selected)
    parts = []
    prev = 0
    for pos in positions:
        if pos not in chosen:
            continue
        parts.append(text[prev:pos])
        parts.append(new)
        prev = pos + len(old)
    parts.append(text[prev:])
    updated_text = "".join(parts)
    updated = updated_text.encode("utf-8")

    def preview_line(content: str, pos: int) -> str:
        """Short preview around the match, avoiding huge MCP result payloads."""
        beginning = content.rfind("\n", 0, pos) + 1
        ending = content.find("\n", pos)
        if ending == -1:
            ending = len(content)
        left = max(beginning, pos - 160)
        right = min(ending, pos + 240)
        return ("..." if left > beginning else "") + content[left:right].rstrip("\r") + ("..." if right < ending else "")

    # Keep the original line number so the LLM can identify which occurrence
    # was modified. Report the full resulting line as additional verification.
    # Track character offsets, since earlier edits may change subsequent positions.
    replacements = []
    delta = 0
    for idx, pos in enumerate(selected):
        new_pos = pos + delta
        if idx < 100:
            replacements.append({
                "line_number": at_line(pos),
                "after_line_number": updated_text.count("\n", 0, new_pos) + 1,
                "before": preview_line(text, pos),
                "after": preview_line(updated_text, new_pos),
            })
        delta += len(new) - len(old)
    replaced_lines = [r["line_number"] for r in replacements]

    # Atomic replacement reduces the chance of leaving half-written code.
    # Keep the original mode and refuse to overwrite an intervening edit.
    before = full.stat()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{full.name}.", suffix=".tmp",
            dir=full.parent, delete=False,
        ) as temp:
            temp_path = Path(temp.name)
            temp.write(updated)
        temp_path.chmod(stat.S_IMODE(before.st_mode))
        now = full.stat()
        if (now.st_ino, now.st_size, now.st_mtime_ns) != (
            before.st_ino, before.st_size, before.st_mtime_ns
        ):
            raise RuntimeError("file changed during replacement; no changes made")
        os.replace(temp_path, full)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return {
        "ok": True, "path": path, "replaced": len(selected), "status": "updated",
        "replaced_line": replaced_lines[0] if len(selected) == 1 else None,
        "replaced_lines": replaced_lines,
        "replacements": replacements,
        "replacements_truncated": len(selected) > len(replacements),
        "already_applied_line": None,
        # Backward compatibility for clients using `lines`.
        "lines": replaced_lines[:100],
    }

@mcp.tool()
def apply_unified_patch(diff: str, strip: int = 0) -> Dict[str, Any]:
    """Apply a unified diff using patch(1). diff must only reference files."""
    if not isinstance(diff, str) or not diff.strip():
        raise ValueError("diff must be a non-empty string")
    if strip not in (0, 1):
        raise ValueError("strip must be 0 or 1")

    if not diff.endswith("\n"):
        diff += "\n"

    cmd = ["patch", "--batch", "--forward", f"-p{strip}"]
    try:
        # Check all hunks before making any changes; reject invalid patches.
        check = subprocess.run(
            [*cmd, "--dry-run"], cwd=str(_root()), input=diff,
            capture_output=True, text=True, timeout=30,
        )
        if check.returncode != 0:
            return {"ok": False, "returncode": check.returncode,
                    "stdout": check.stdout, "stderr": check.stderr}
        r = subprocess.run(
            cmd, cwd=str(_root()), input=diff,
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError as e:
        raise RuntimeError("patch(1) not found; install 'patch'") from e
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
        cwd=str(_root()),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return {"command": command, "cwd": str(_root()), "stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode}


@mcp.tool()
def run_ssh(host: str, command: str, timeout_s: int = 300) -> Dict[str, Any]:
    """Run a command over SSH using the user's configured public key."""
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")
    if host != host.strip() or host.startswith("-") or any(c.isspace() for c in host):
        raise ValueError("host must be an SSH alias, hostname, or user@host without options")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    if len(command) > MAX_CMD_CHARS:
        raise ValueError(f"command too long ({len(command)} chars > {MAX_CMD_CHARS})")

    bad = _dangerous_cmd(command)
    if bad and not ALLOW_DANGEROUS:
        raise ValueError(f"dangerous command blocked: {bad} (set MCP_ALLOW_DANGEROUS=1 to override)")

    try:
        r = subprocess.run(
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "PubkeyAuthentication=yes",
                "-o", "PreferredAuthentications=publickey",
                "-o", "PasswordAuthentication=no",
                "-o", "KbdInteractiveAuthentication=no",
                host,
                command,
            ],
            cwd=str(_root()),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as e:
        raise RuntimeError("ssh client not found; install openssh-client") from e

    return {
        "host": host,
        "command": command,
        "stdout": r.stdout,
        "stderr": r.stderr,
        "returncode": r.returncode,
    }


@mcp.tool()
def ask_big_brother(question: str) -> Dict[str, Any]:
    """Ask OpenAI with web search enabled for current facts/manuals/docs."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var is not set")

    model = os.environ.get("BIG_BROTHER_MODEL", "gpt-6.1-terra")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("python package 'openai' is not installed") from e

    client = OpenAI(api_key=api_key, timeout=300)

    response = client.responses.create(
        model=model,
        input=question,
        tools=[{"type": "web_search_preview"}],
        include=["web_search_call.action.sources"],
        timeout=300
    )

    answer = getattr(response, "output_text", "") or ""

    # best-effort extraction of sources from the response payload
    sources = []
    try:
        payload = response.model_dump()
        for item in payload.get("output", []):
            if item.get("type") == "web_search_call":
                action = item.get("action", {})
                for src in action.get("sources", []):
                    if isinstance(src, dict):
                        sources.append({
                            "title": src.get("title"),
                            "url": src.get("url"),
                        })
    except Exception:
        pass

    return {
        "ok": True,
        "model": model,
        "question": question,
        "answer": answer,
        "sources": sources,
    }

def main() -> None:
    # Direct execution: stdio server.
    mcp.run()


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--grep-worker":
        _grep_worker_main()
    else:
        main()
