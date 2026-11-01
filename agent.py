#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import shlex
import subprocess
import sys
import tempfile
import urllib.parse
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional



DEFAULT_SERVER_URL = "http://192.168.3.11:8080/v1/chat/completions"
DEFAULT_MODEL_LLAMA = "qwen"
DEFAULT_MODEL_OPENAI = "gpt-5.6-terra"
AGENT_VERSION = "R261101_VISION_GEOMETRY_18"


SYSTEM_PROMPT = r"""
You are a helpful AI assistant with access to tools.

You MUST respond with EXACTLY ONE JSON object and NOTHING ELSE.
NO markdown fences. NEVER output ```.
Output must be valid JSON parsable by json.loads().
Do not include any text before or after the JSON object.
IMPORTANT FOR SOURCE FILES: escape quotes, backslashes and newlines inside JSON strings
(e.g. use \" for a literal quote and \n for a newline). Never output raw multiline
code inside a JSON string unless it is correctly escaped.
If creating/replacing a large file, keep each tool call SMALL (roughly 2000-4000
characters of content, or less if your output token budget is small): first
write_text, then append_text for each next chunk. Do not send a huge file in a
single JSON response: output truncation makes the entire tool call unusable.
Absolute paths (e.g. /home/user/project/LibManager.h) are allowed by the file
tools; relative paths resolve against the workspace. Preserve user-supplied paths.

Use one of these shapes (all examples are valid JSON; no comments in the output):
{"type":"message","plan":[],"message":"Done."}
{"type":"tool","plan":["Write first chunk"],"tool":"mcp","name":"write_text","arguments":{"path":"file.h","content":"first chunk"}}
{"type":"tool","plan":["Compile"],"tool":"bash","commands":["make"]}
A "plan" and an MCP "tool" are NOT competing operations: they may, and normally
should, appear in the SAME JSON object. The plan is displayed to the user;
the MCP tool is executed. "plan" is optional; omit it or use [] if unneeded.
Do not combine "type":"message" and "type":"tool" in one reply, and do not
output multiple JSON objects in one reply.

Tool policy:
- For file operations, use tool="mcp" and the appropriate local file-specific tool, not shell equivalents.
- External MCP tools use qualified names server.tool_name, but ONLY when explicitly present in the LIVE discovered catalog appended to this system message. Do not infer tool availability from examples, previous conversations, or general knowledge.
- To check connections, call manager.list_servers. If an MCP server is disabled or unavailable, report the problem; do not attempt its tools.
- FreeCAD: after create_object/edit_object the agent automatically reads back get_object and, when available, performs a fixed read-only shape validation probe. Read TOOL_RESULT.verification; do NOT claim exact dimensions, volume, or shape validity unless status=verified. On status=mismatch or failed, report the error rather than success. On unverified, explicitly say that verification was incomplete.
- FreeCAD screenshots: MCP get_view saves an image, but an image path is NOT visual perception. TOOL_RESULT.vision.status=not_analyzed means you did NOT see pixels. status=analyzed_by_sidecar includes a visual_description from a separate vision model. status=sent_to_main_model means image pixels are attached to the CURRENT inference only, provided the active model actually supports vision. Never say you visually inspected the model otherwise.

- An external tool may execute arbitrary code or modify files; explicit user confirmation is required unless the administrator explicitly lists it as read_only_tools in mcp_servers.json.
- Local MCP tools and their arguments (external servers are documented in the live discovered catalog):
  - list_files(path=".", recursive=false, include_hidden=false, max_entries=200): list files/directories, like ls -la or ls -R; absolute paths allowed. Returns paths, types and sizes; output is bounded.
  - grep_files(pattern, path=".", recursive=true, ignore_case=false, regex=false, file_glob="*", max_matches=100, max_files=0, include_hidden=false, timeout_s=30): search with a 30s default time limit (configurable 1..240s). Searches are bounded in time even when max_files=0. regex=false performs SUBSTRING matching, not whole-word or whole-line matching. max_files=0 means unlimited; max_matches limits returned lines. Returns complete/truncated/truncation_reasons; if complete=false, a zero count is NOT evidence the pattern is absent. On timeout/busy, narrow path or file_glob and retry; do NOT repeat the same huge search. Optional MCP_MAX_GREP_FILE_BYTES and MCP_MAX_GREP_TOTAL_BYTES environment limits default to 0 (unlimited); skips binary files and hidden paths by default.
  - read_text(path, start_line=1, end_line=null, count=null): read UTF-8 text.
    Line numbers are 1-based; end_line is INCLUSIVE. Use EITHER end_line OR
    count (number of lines from start_line), not both. With no range, reads the
    whole file as before. MCP_MAX_READ_BYTES limits the RETURNED content, not
    the whole file size for ranged reads. Ranged results include start_line,
    end_line (actual last line), lines_read, has_more, and next_start_line.
    For large files, prefer small ranges, e.g.
    {"type":"tool","tool":"mcp","name":"read_text","arguments":{"path":"src/file.c","start_line":100,"count":80}}
    If has_more=true, use next_start_line to continue reading. Do not treat a
    short ranged read as the entire file.
  - write_text(path, content, overwrite=true|false): create/replace UTF-8 text (MCP_MAX_WRITE_BYTES per call).
  - append_text(path, content): append UTF-8 text at EOF, like >>. Creates missing file/dirs; does NOT add a newline automatically. Existing content is preserved.
  - merge_files(paths, output_path, overwrite=true|false): concatenate TWO OR MORE files in the given order, exactly like cat A B > OUT. Binary-safe; NO separators or newlines added. output_path must differ from every input; total limited by MCP_MAX_MERGE_BYTES.
  - replace_in_file(path, old, new, count=1, occurrence=null, line_number=null, replace_all=false):
    safe LITERAL replacement, not regex. Default requires exactly one unmodified
    match; ambiguous matches FAIL without changes. Use occurrence=N (1-based
    position in current file) or line_number=N (1-based matching line) to target
    exactly one location, or replace_all=true to change every unmodified match.
    Use ONLY ONE selector; legacy count=N>1 requires exactly N unmodified matches.
    Response includes status, replaced, replaced_line (successful single edit),
    replaced_lines, and replacements (before/after previews). On an idempotent
    no-op for ONE identifiable target, status=already_applied, replaced=0, and
    already_applied_line gives THAT location. There is NO saved history and no
    global already_applied_lines list. A matching new string is evidence only
    that the requested final text is currently present at the target location,
    not that this tool made the earlier edit. If edits are ambiguous, use
    grep_files and choose line_number or more unique surrounding old text.
    Example: {"type":"tool","tool":"mcp","name":"replace_in_file","arguments":{"path":"src/file.c","old":"foo(","new":"foo(NULL,","line_number":42}}
    IMPORTANT: If new contains old (e.g. foo( -> foo(NULL,), the tool detects
    a repeated application at the selected location and returns
    status=already_applied, replaced=0. Never blindly repeat an edit.
  - apply_unified_patch(diff, strip=0): apply a standard unified diff (strip=0 or 1).
  - run_bash(command, timeout_s=300): run a local command in the workspace; use for compiling/running/testing, NOT file management.
  - run_ssh(host, command, timeout_s=300): run a remote SSH command (needs user approval).
  - ask_big_brother(question): external factual lookup, no filesystem access.
- For directory listings ALWAYS use list_files, NOT bash ls. For searching file contents ALWAYS use grep_files, NOT bash grep or find. Check grep_files complete before claiming text is absent; if incomplete, narrow the directory/glob and search again. Example: {"type":"tool","tool":"mcp","name":"list_files","arguments":{"path":"/home/user/project/include"}}.
- A plain, read-only `ls` or `grep` mistakenly sent to run_bash is automatically routed to the corresponding MCP tool when unambiguous. Do not rely on this fallback for complex shell syntax.
- For targeted edits prefer replace_in_file with enough surrounding text to make old unique. If ambiguous, use grep_files to locate the correct line and then line_number or occurrence. NEVER assume the first match is the intended target. If replacing every occurrence, explicitly set replace_all=true. Do not retry a successful edit on the same substring (e.g. inserting NULL into a function call). For multi-hunk changes use apply_unified_patch.
- For a new large TEXT file: use write_text for the first chunk and append_text for subsequent chunks, in order. Never use repeated write_text calls for chunking: they overwrite preceding chunks. Do not use this method for binary files.
- To append to an existing file, use append_text, NOT write_text. To merge files, use merge_files, NOT cat or shell redirection.
- Before overwriting a destination or merging, respect the user's intent; use overwrite=false if an existing destination must be protected.
- Use tool="bash" ONLY for compiling/running/testing programs.
- Use ask_big_brother when you need current facts, manuals, package docs, or web research. **Big brother doesn't have an access to the filesystem, if you need review the file, put the content into the message itself**
- Do NOT use bash or MCP run_bash for file operations: cat, ls, find, grep, sed, awk, perl, patch, redirects (>, >>), cp, mv, rm, touch, mkdir, rmdir, etc.
- If the user asks to compile/run/test/execute/verify, you MUST respond with type="tool" first.
- If type == "tool", output ONLY the tool JSON (do not include a final answer in the same reply).
- NEVER claim a tool error unless you received a TOOL_RESULT that shows an error.
- Avoid dangerous commands unless user explicitly requests and confirms.
- When writing code, save it as soon as practical; reread the file if you need to check it.
- If the user asks found something in the internet use DuckDuckGo (`ddgr [QUERY] --np --json`). 
- If the user asks to read the URL use 'elinks --dump [URL]' or 'curl [URL]'

DuckDuckGo (ddgr) Output JSON will have next schema:

[
  {
    "abstract": "abstract description",
    "title": "Title",
    "url": "https://[url]"
  },
]

- If you don't know something or need to proof it first use ask_big_brother and then use DuckDuckGo to search additional information
"""

MAX_TOOL_ROUNDS = 64
MAX_CMD_CHARS = 8000

# Legacy local MCP tool names; external tools are discovered dynamically by MCPManager.
MCP_TOOLS = frozenset({
    "list_files", "grep_files", "read_text", "write_text", "append_text", "merge_files",
    "replace_in_file", "apply_unified_patch", "run_bash", "run_ssh",
    "ask_big_brother",
})
MCP_CONFIRM_TOOLS = frozenset({
    "write_text", "append_text", "merge_files", "replace_in_file",
    "apply_unified_patch", "run_bash", "run_ssh",
})


HANDOFF_REQUEST = """
Write, please, your goals and plan how would you achieve these goals.

Describe:
1. What are the goals;
2. What is the plan to achieve these goals;
3. What already has been changed/tried;
4. Describe failed attempts.

Requirements:
- Be concrete.
- Mention important file paths if relevant.
- Keep it compact but useful for continuing work.
- Respond as EXACTLY ONE JSON object of type="message".
- Put the full handoff text into the "message" field.
"""

_TEXT_EXTENSIONS = {
    ".py", ".c", ".cpp", ".h", ".hpp", ".txt", ".md", ".json",
    ".yaml", ".yml", ".ini", ".cfg", ".sh", ".toml", ".xml"
}


def _extract_message_text(raw: Optional[str]) -> str:
    if raw is None:
        return ""

    raw = raw.strip()
    if not raw:
        return ""

    objs = parse_assistant_json_objects(raw)
    if objs:
        obj = objs[0]
        if isinstance(obj, dict):
            msg = obj.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()

    return raw


def _request_handoff_summary(
    backend: Any,
    messages: List[Dict[str, str]],
    max_tokens: int,
) -> str:
    req_messages = list(messages)
    req_messages.append({"role": "user", "content": HANDOFF_REQUEST})

    resp = backend.chat_completion(
        messages=req_messages,
        max_tokens=min(max_tokens, 2048),
        temperature=0.2,
    )
    if resp:
        raw = resp.get("content", "")
        text = _extract_message_text(raw)
        if text:
            print("\n" + "─" * 50)
            print("🗘 Handoff")
            print("─" * 50)
            print(text)
            print("─" * 50)
            return text

    return "No handoff summary could be generated."


def _build_project_status(
    workdir: str,
    max_files: int = 200,
    max_file_bytes: int = 16000,
    max_total_chars: int = 60000,
) -> str:
    root = Path(workdir).resolve()

    files: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            files.append(p)
            if len(files) >= max_files:
                break

    lines: List[str] = []
    lines.append(f"Workspace root: {root}")
    lines.append("")
    lines.append("Files:")
    for p in files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        try:
            size = p.stat().st_size
        except Exception:
            size = -1
        lines.append(f"- {rel} ({size} bytes)")

    total_chars = 0
    content_blocks: List[str] = []

    for p in files:
        if p.suffix.lower() not in _TEXT_EXTENSIONS:
            continue

        try:
            size = p.stat().st_size
        except Exception:
            continue

        if size > max_file_bytes:
            continue

        rel = str(p.relative_to(root)).replace("\\", "/")
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        block = f"\n### FILE: {rel}\n{text}\n"
        if total_chars + len(block) > max_total_chars:
            break

        content_blocks.append(block)
        total_chars += len(block)

    if content_blocks:
        lines.append("")
        lines.append("Selected file contents:")
        lines.extend(content_blocks)

    return "\n".join(lines)


def user_requires_tool(user_input: str) -> bool:
    # Word boundaries prevent accidental triggers such as 'brunch' containing 'run'.
    return bool(re.search(
        r"\b(?:compile|compilation|build|building|run|running|execute|execution|"
        r"test|tests|testing|verify|verification|benchmark|benchmarks|benchmarking)\b",
        user_input, re.IGNORECASE,
    ))


def run_bash_command(
    command: str,
    cwd: str,
    timeout_s: int = 300,
    cancel_event: Optional[Any] = None,
) -> Dict[str, Any]:
    if cancel_event is None:
        try:
            r = subprocess.run(
                ["bash", "-lc", command],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return {
                "command": command,
                "cwd": cwd,
                "stdout": r.stdout,
                "stderr": r.stderr,
                "returncode": r.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "command": command,
                "cwd": cwd,
                "stdout": "",
                "stderr": f"timeout after {timeout_s}s",
                "returncode": 124,
            }
        except Exception as e:
            return {
                "command": command,
                "cwd": cwd,
                "stdout": "",
                "stderr": str(e),
                "returncode": 1,
            }

    start = time.monotonic()
    proc: Optional[subprocess.Popen[str]] = None
    try:
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        while True:
            if _cancel_requested(cancel_event):
                try:
                    proc.terminate()
                    try:
                        stdout, stderr = proc.communicate(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        stdout, stderr = proc.communicate(timeout=2.0)
                except Exception:
                    stdout, stderr = "", ""
                return {
                    "command": command,
                    "cwd": cwd,
                    "stdout": stdout,
                    "stderr": (stderr or "") + "\ncancelled by user",
                    "returncode": 130,
                }

            try:
                stdout, stderr = proc.communicate(timeout=0.2)
                return {
                    "command": command,
                    "cwd": cwd,
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": proc.returncode,
                }
            except subprocess.TimeoutExpired:
                if (time.monotonic() - start) >= timeout_s:
                    proc.kill()
                    stdout, stderr = proc.communicate(timeout=2.0)
                    return {
                        "command": command,
                        "cwd": cwd,
                        "stdout": stdout,
                        "stderr": (stderr or "") + f"\ntimeout after {timeout_s}s",
                        "returncode": 124,
                    }
                continue
    except Exception as e:
        try:
            if proc is not None and proc.poll() is None:
                proc.kill()
        except Exception:
            pass
        return {
            "command": command,
            "cwd": cwd,
            "stdout": "",
            "stderr": str(e),
            "returncode": 1,
        }


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _repair_minor_json_syntax(txt: str) -> str:
    """Repair only punctuation with unambiguous meaning outside source content.

    Accept accidental trailing commas and literal backslashes followed by an
    invalid JSON escape (common in C/C++ strings and paths). Never guess a
    missing quote, a missing chunk of file content, or a closing brace.
    """
    out: List[str] = []
    in_string = False
    i = 0
    while i < len(txt):
        ch = txt[i]
        if in_string:
            if ch == '"':
                in_string = False
            elif ch == "\\" and i + 1 < len(txt):
                nxt = txt[i + 1]
                valid = nxt in '\\"/bfnrt'
                if nxt == 'u':
                    valid = (
                        i + 5 < len(txt)
                        and all(c in '0123456789abcdefABCDEF' for c in txt[i + 2:i + 6])
                    )
                if not valid:
                    # A literal \# or \_ inside generated source text.
                    out.append('\\\\')
                    i += 1
                    continue
                out.append(ch)
                out.append(nxt)
                i += 2
                continue
        else:
            if ch == '"':
                in_string = True
            elif ch == ',':
                j = i + 1
                while j < len(txt) and txt[j].isspace():
                    j += 1
                if j < len(txt) and txt[j] in '}]':
                    i += 1
                    continue
        out.append(ch)
        i += 1
    return ''.join(out)


def _decode_json_objects(txt: str) -> Optional[List[Dict[str, Any]]]:
    dec = json.JSONDecoder(strict=False)  # Allow literal newlines/tabs in strings.
    i = 0
    objs: List[Dict[str, Any]] = []
    try:
        while i < len(txt):
            while i < len(txt) and txt[i].isspace():
                i += 1
            if i >= len(txt):
                break
            obj, i = dec.raw_decode(txt, i)
            if not isinstance(obj, dict):
                return None
            objs.append(obj)
    except (ValueError, TypeError):
        return None
    return objs or None


def parse_assistant_json_objects(raw: str) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    txt = strip_code_fences(raw)
    # The first two attempts preserve the generated file contents verbatim.
    for candidate in (txt, _repair_minor_json_syntax(txt)):
        result = _decode_json_objects(candidate)
        if result is not None:
            return result
    # Preserve compatibility with models that prepend a short explanatory line.
    start = txt.find('{')
    if start > 0:
        for candidate in (txt[start:], _repair_minor_json_syntax(txt[start:])):
            result = _decode_json_objects(candidate)
            if result is not None:
                return result
    return None


def _json_failure_details(raw: str) -> Dict[str, Any]:
    """Explain parse failures without mistaking a 300-char *preview* for EOF."""
    txt = strip_code_fences(raw)
    detail: Dict[str, Any] = {
        'raw_chars': len(raw),
        'raw_start': raw[:160],
        'raw_end': raw[-240:],
    }
    try:
        json.JSONDecoder(strict=False).raw_decode(txt)
    except json.JSONDecodeError as exc:
        detail['json_error'] = f'{exc.msg} at line {exc.lineno}, column {exc.colno}'
    # An unterminated JSON structure is a possible output-token cutoff.
    # The heuristic is only diagnostic; NEVER execute incomplete tool calls.
    depth = 0
    in_string = False
    escaped = False
    for ch in txt:
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in '{[':
            depth += 1
        elif ch in '}]':
            depth -= 1
    detail['possibly_truncated'] = in_string or depth > 0
    return detail


def _is_safe_relative_path(p: str) -> bool:
    if not p or not isinstance(p, str):
        return False
    if p.startswith(("/", "~")):
        return False
    if re.match(r"^[A-Za-z]:[\\/]", p):
        return False
    pp = Path(p)
    if any(part == ".." for part in pp.parts):
        return False
    return True


def _bash_looks_like_filesystem(cmd: str) -> bool:
    """Block shell-based file operations while allowing normal test/compiler commands.

    Tokenizing avoids false positives in filenames and quoted Python -c code.
    This is an agent policy check, NOT a security sandbox for arbitrary shell code.
    """
    try:
        lexer = shlex.shlex(cmd, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return True

    file_commands = {
        #"cat", "ls", "find", "grep", "egrep", "fgrep", 
        "sed", "awk", "perl",
        "patch", "cp", "mv", "rm", "touch", "mkdir", "rmdir", "ln", "tee",
        "truncate", "dd", "head", "tail", "stat", "install", "rsync",
    }
    separators = {";", "&&", "||", "|", "&"}
    at_command_start = True
    for token in tokens:
        if token in {">", ">>", "<", "<<", "<<<", "<>"}:
            return True
        if token in separators:
            at_command_start = True
            continue
        if at_command_start:
            if token in {"(", ")", "{", "}"}:
                continue
            if token.rsplit("/", 1)[-1] in file_commands:
                return True
            at_command_start = False
    return False


def _route_readonly_file_command(command: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Translate common read-only ls/grep shell commands into MCP operations.

    This is intentionally a small compatibility adapter, not a shell evaluator:
    pipes, redirects (other than trailing 2>&1), multiple paths, and unsupported
    options are not executed or guessed. The returned tool receives plain args.
    """
    if not isinstance(command, str) or not command.strip() or len(command) > MAX_CMD_CHARS:
        return None
    try:
        args = shlex.split(command)
    except ValueError:
        return None
    if args and args[-1] == "2>&1":
        args.pop()  # harmless shell stderr-to-stdout redirection
    if not args or any(token in {";", "&&", "||", "|", "&", ">", ">>", "<", "2>", "1>"}
                       for token in args):
        return None
    executable = args.pop(0)
    if executable not in {"ls", "grep", "egrep", "fgrep"}:
        return None

    if executable == "ls":
        include_hidden = False
        recursive = False
        paths: list[str] = []
        options_done = False
        for arg in args:
            if arg == "--" and not options_done:
                options_done = True
            elif not options_done and arg in ("--color=auto", "--color=never", "--color=always"):
                continue
            elif not options_done and arg.startswith("-") and arg != "-":
                if any(flag not in "alh1FRpA" for flag in arg[1:]):
                    return None
                include_hidden |= any(flag in arg for flag in "aA")
                recursive |= "R" in arg
            else:
                paths.append(arg)
        if len(paths) > 1:
            return None
        return "list_files", {"path": paths[0] if paths else ".",
                              "recursive": recursive, "include_hidden": include_hidden}

    recursive = False
    ignore_case = False
    use_regex = executable == "egrep"
    fixed = executable == "fgrep"
    file_glob = "*"
    positional: list[str] = []
    options_done = False
    for arg in args:
        if arg == "--" and not options_done:
            options_done = True
        elif not options_done and arg.startswith("--include="):
            file_glob = arg.split("=", 1)[1]
        elif not options_done and arg.startswith("-") and arg != "-":
            if any(flag not in "RrniFE" for flag in arg[1:]):
                return None
            recursive |= any(flag in arg for flag in "Rr")
            ignore_case |= "i" in arg
            fixed |= "F" in arg
            use_regex |= "E" in arg
        else:
            positional.append(arg)
    if not 1 <= len(positional) <= 2:
        return None
    return "grep_files", {"pattern": positional[0],
                          "path": positional[1] if len(positional) > 1 else ".",
                          "recursive": recursive, "ignore_case": ignore_case,
                          "regex": use_regex and not fixed, "file_glob": file_glob}


def render_assistant(obj: Dict[str, Any]) -> None:
    plan = obj.get("plan")
    if isinstance(plan, list) and plan:
        print("\n" + "─" * 50)
        print("🧭 PLAN")
        print("─" * 50)
        for b in plan[:3]:
            print(f"- {b}")
        print("─" * 50)

    msg = str(obj.get("message", "") or "").strip()
    if msg:
        print(f"\nAgent: {msg}\n")


def _event_title_with_icon(title: str) -> str:
    """Keep GUI runtime events readable without forcing the core to know GTK.

    The callback title is Markdown-rendered by agent_gui.py.  Older revisions had
    useful UTF-8 markers in these titles; keep that compact visual language here
    so live reasoning/tool events are easy to scan.
    """
    raw = str(title or "Agent event").strip() or "Agent event"
    if raw[:1] in {"⚠", "⛔", "✅", "🚫", "▶", "🧠", "🧭", "🖺", "🧾", "🤖", "👤", "🛠", "ℹ", "🗘"}:
        return raw

    mapping = {
        "User request": "👤 User request",
        "LLM request": "🤖 LLM request",
        "LLM error": "⛔ LLM error",
        "LLM reasoning": "🧠 LLM reasoning",
        "LLM response": "ℹ️ LLM response",
        "LLM raw": "🧾 LLM raw",
        "LLM parser": "⚠️ LLM parser",
        "LLM tool request": "⚠️ LLM tool request",
        "LLM message": "🖺 LLM message",
        "Protocol repair": "🛠 Protocol repair",
        "Tool validation error": "⛔ Tool validation error",
        "Bash tool request": "⚠️ Bash tool request ⚠️",
        "Bash tool denied": "🚫 Bash tool denied",
        "Bash tool running": "▶️ Bash tool running",
        "Bash tool result": "✅ Bash tool result",
        "MCP tool request": "⚠️ MCP tool request ⚠️",
        "MCP tool denied": "🚫 MCP tool denied",
        "MCP tool running": "▶️ MCP tool running",
        "MCP tool result": "⚠️ MCP tool result ⚠️",
        "MCP tool error": "⛔ MCP tool error ⛔",
        "Turn complete": "✅ Turn complete",
        "Turn stopped": "⛔ Turn stopped",
    }
    return mapping.get(raw, raw)


def _emit_runtime_event(
    event_callback: Optional[Callable[[str, str], None]],
    title: str,
    body: Any = "",
) -> None:
    """Send a live progress/event block to GUI callers.

    CLI mode ignores this callback and keeps the existing stdout renderer.
    The GUI uses it to show each LLM round, reasoning section, tool request,
    confirmation decision, and tool result immediately while the turn is still
    running.
    """
    if event_callback is None:
        return
    try:
        if isinstance(body, str):
            text = body
        else:
            text = "```json\n" + json.dumps(body, ensure_ascii=False, indent=2) + "\n```"
    except Exception:
        text = str(body)
    try:
        event_callback(_event_title_with_icon(str(title)), str(text or "").strip())
    except Exception:
        # UI/progress callbacks must never break the agent loop.
        pass


def _format_assistant_object_for_event(obj: Dict[str, Any]) -> str:
    lines: List[str] = []

    typ = obj.get("type")
    if typ:
        lines.append(f"**⚠️ TYPE: `{typ}`**")

    plan = obj.get("plan")
    if isinstance(plan, list) and plan:
        lines.append("")
        lines.append("**🧭 PLAN:**")
        for item in plan[:32]:
            lines.append(f"- {item}")

    msg = str(obj.get("message", "") or "").strip()
    if msg:
        lines.append("")
        lines.append("**🖺 MESSAGE:**")
        lines.append(msg)

    if typ == "tool":
        tool = obj.get("tool")
        if tool:
            lines.append("")
            lines.append(f"**⚠️ TOOL: `{tool}`**")
        if tool == "bash":
            commands = obj.get("commands", [])
            lines.append("**⚠️ COMMANDS:**")
            if isinstance(commands, list):
                for index, command in enumerate(commands, start=1):
                    lines.append(f"{index}. `{command}`")
            else:
                lines.append(str(commands))
        elif tool == "mcp":
            name = obj.get("name")
            arguments = obj.get("arguments", {})
            lines.append(f"**⚠️ NAME: `{name}`**")
            try:
                lines.append("```json\n" + json.dumps(arguments, ensure_ascii=False, indent=2) + "\n```")
            except Exception:
                lines.append(str(arguments))

    return "\n".join(lines).strip() or "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"

def _cancel_requested(cancel_event: Optional[Any]) -> bool:
    try:
        return bool(cancel_event is not None and cancel_event.is_set())
    except Exception:
        return False


def _raise_if_cancelled(
    cancel_event: Optional[Any],
    event_callback: Optional[Callable[[str, str], None]] = None,
) -> None:
    if _cancel_requested(cancel_event):
        _emit_runtime_event(event_callback, "Turn stopped", "Cancelled by user.")
        raise KeyboardInterrupt


def _confirm(
    prompt: str,
    no_confirm: bool,
    confirm_func: Optional[Callable[[str], bool]] = None,
) -> bool:
    if no_confirm:
        return True
    if confirm_func is not None:
        return bool(confirm_func(prompt))
    ans = input(prompt).strip().lower()
    return ans == "y"


# Refresh in place so both CLI and GTK callers see the same available tools.
_MCP_CATALOG_MARKER = "\n\n=== MCP TOOL CATALOG (dynamic) ===\n"


def _sync_mcp_system_prompt(messages: List[Dict[str, str]], mcp: Any) -> None:
    if not messages or messages[0].get("role") != "system":
        return
    if not hasattr(mcp, "prompt_catalog"):
        return
    current = messages[0].get("content", SYSTEM_PROMPT)
    base = current.split(_MCP_CATALOG_MARKER, 1)[0]
    catalog = mcp.prompt_catalog()
    updated = base + _MCP_CATALOG_MARKER + catalog
    if current != updated:
        messages[0]["content"] = updated


def _start_mcp_for_workspace(workspace_dir: str) -> Any:
    from mcp_manager import MCPManager

    mcp = MCPManager(workspace_dir)
    mcp.start()
    return mcp


def run_agent_turn(
    backend: Any,
    mcp: Any,
    messages: List[Dict[str, str]],
    user_input: str,
    max_tokens: int,
    temperature: float,
    no_confirm: bool,
    workdir: str,
    debug: bool = False,
    confirm_func: Optional[Callable[[str], bool]] = None,
    event_callback: Optional[Callable[[str, str], None]] = None,
    cancel_event: Optional[Any] = None,
) -> None:
    _sync_mcp_system_prompt(messages, mcp)
    messages.append({"role": "user", "content": user_input})
    need_tool = user_requires_tool(user_input)

    # Gate becomes True only after a tool was executed or explicitly denied.
    tool_gate_open = not need_tool
    json_repair_failures = 0
    repair_mode = False

    _emit_runtime_event(
        event_callback,
        "User request",
        f"{user_input}\n\nTool required first: {need_tool}",
    )
    _raise_if_cancelled(cancel_event, event_callback)

    for _round in range(MAX_TOOL_ROUNDS):
        _raise_if_cancelled(cancel_event, event_callback)
        _emit_runtime_event(event_callback, "LLM request", f"Round {_round + 1}")
        # Vision attachments are transient. They are sent for precisely one LLM
        # request and replaced by their text summaries before context save/trim.
        multimodal_pending = [m for m in messages if isinstance(m.get("content"), list)]
        try:
            resp = backend.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=min(temperature, 0.2) if repair_mode else temperature,
            )
        except KeyboardInterrupt:
            print("\n" + "─" * 50)
            print("CTRL-C break")
            print("─" * 50)
            _emit_runtime_event(event_callback, "Turn stopped", "Interrupted by user.")
            return
        finally:
            for item in multimodal_pending:
                parts = item.get("content")
                if isinstance(parts, list):
                    item["content"] = next((p.get("text", "") for p in parts
                                            if isinstance(p, dict) and p.get("type") == "text"), "")

        _raise_if_cancelled(cancel_event, event_callback)

        if resp is None:
            print("✗ No response from model.")
            _emit_runtime_event(event_callback, "LLM error", "No response from model.")
            return

        raw = resp.get("content", "")
        reasoning = resp.get("reasoning", "")

        if reasoning:
            print("\n" + "─" * 50)
            print("🧠 REASONING")
            print("─" * 50)
            print(reasoning)
            print("─" * 50 + "\n")
            _emit_runtime_event(event_callback, "LLM reasoning", reasoning)
            if raw.strip() == "":
                json_repair_failures += 1
                repair_mode = True
                _emit_runtime_event(event_callback, "LLM response", "Reasoning exhausted the output budget; requesting short JSON without further reasoning.")
                messages.append({"role": "user", "content": "No tool was executed. Your response contained reasoning but no JSON. Reply now with one short, complete JSON object, using less reasoning and a smaller content chunk."})
                if json_repair_failures >= 4:
                    _emit_runtime_event(event_callback, "Turn stopped", "Repeated empty model content.")
                    return
                continue

        if debug:
            print("---------- RAW -----------")
            print(raw[:1024])
            print("--------------------------")
            _emit_runtime_event(event_callback, "LLM raw", raw[:4000])

        if raw.strip() == "":
            json_repair_failures += 1
            repair_mode = True
            error_payload = {
                "ok": False,
                "error": "Empty model response. No tool was executed. Retry with a SHORT, complete JSON object and minimal reasoning.",
            }
            _emit_runtime_event(event_callback, "Protocol repair", error_payload)
            messages.append(
                {
                    "role": "user",
                    "content": "TOOL_RESULT " + json.dumps(error_payload),
                }
            )
            if json_repair_failures >= 4:
                _emit_runtime_event(event_callback, "Turn stopped", "Repeated empty model content.")
                return
            continue

        objs = parse_assistant_json_objects(raw)
        if not objs:
            json_repair_failures += 1
            repair_mode = True
            error_payload = {
                "ok": False,
                "error": "Could not parse the response as complete JSON; no tool was executed. A single object containing both plan and tool is valid. Generate a NEW complete JSON object, not a continuation of the previous response.",
                **_json_failure_details(raw),
                "finish_reason": resp.get("finish_reason"),
                "repair_hint": (
                    "If writing a long source file, split it into small pieces (2000-4000 "
                    "characters): write_text for the FIRST piece, then append_text for "
                    "each following piece. Include escaped newlines and quotes inside "
                    "JSON strings. Do not repeat the entire file in one response. "
                    "Use a short plan and minimal reasoning."
                ),
            }
            _emit_runtime_event(event_callback, "Protocol repair", error_payload)
            messages.append(
                {
                    "role": "user",
                    "content": "TOOL_RESULT " + json.dumps(error_payload),
                }
            )
            if json_repair_failures >= 4:
                _emit_runtime_event(event_callback, "Turn stopped", "Repeated malformed/incomplete JSON; stopped without executing the incomplete call.")
                print("Agent: Repeated invalid JSON replies; stopped this turn. No incomplete tool call was executed.\n")
                return
            continue

        json_repair_failures = 0
        repair_mode = False
        tool_objs = [o for o in objs if isinstance(o, dict) and o.get("type") == "tool"]
        ordered = tool_objs if tool_objs else objs

        if len(objs) > 1 and tool_objs:
            print("⚠️ Multiple JSON objects received; prioritizing tool request.\n")
            _emit_runtime_event(event_callback, "LLM parser", "Multiple JSON objects received; prioritizing tool request.")

        for obj in ordered:
            if not isinstance(obj, dict) or obj.get("type") not in ("message", "tool") or ("plan" in obj and not isinstance(obj["plan"], list)):
                error_payload = {
                    "ok": False,
                    "error": "Invalid assistant object schema; expected type=message|tool with optional plan array.",
                }
                _emit_runtime_event(event_callback, "Protocol repair", error_payload)
                messages.append(
                    {
                        "role": "user",
                        "content": "TOOL_RESULT " + json.dumps(error_payload),
                    }
                )
                break

            _emit_runtime_event(
                event_callback,
                "LLM tool request" if obj.get("type") == "tool" else "LLM message",
                _format_assistant_object_for_event(obj),
            )

            assistant_message = {
                "role": "assistant",
                "content": json.dumps(obj, ensure_ascii=False),
            }
            if reasoning:
                assistant_message["reasoning_content"] = reasoning
            messages.append(assistant_message)

            typ = obj.get("type")

            if typ == "message":
                if need_tool and not tool_gate_open:
                    error_payload = {
                        "ok": False,
                        "error": "Tool is required. Reply with type='tool' first to compile/run/test; do not guess outputs.",
                    }
                    _emit_runtime_event(event_callback, "Protocol repair", error_payload)
                    messages.append(
                        {
                            "role": "user",
                            "content": "TOOL_RESULT " + json.dumps(error_payload),
                        }
                    )
                    break
                render_assistant(obj)
                _emit_runtime_event(event_callback, "Turn complete", "Final message returned.")
                return

            if typ == "tool":
                render_assistant(obj)
                tool = obj.get("tool")

                # Models sometimes emit a shell-style `ls /path 2>&1` request.
                # Route simple read-only ls/grep commands to dedicated MCP tools
                # instead of rejecting them as prohibited filesystem commands.
                if tool == "bash" and isinstance(obj.get("commands"), list) and len(obj["commands"]) == 1:
                    routed = _route_readonly_file_command(obj["commands"][0])
                    if routed is not None:
                        name, arguments = routed
                        _emit_runtime_event(event_callback, "Protocol repair", f"Routing read-only shell command to MCP {name}")
                        obj = {**obj, "tool": "mcp", "name": name, "arguments": arguments}
                        tool = "mcp"

                if tool == "bash":
                    commands = obj.get("commands", [])
                    if not isinstance(commands, list) or not commands:
                        error_payload = {"ok": False, "error": "missing commands[]"}
                        _emit_runtime_event(event_callback, "Tool validation error", error_payload)
                        messages.append(
                            {
                                "role": "user",
                                "content": "TOOL_RESULT " + json.dumps(error_payload),
                            }
                        )
                        break

                    joined = " && ".join([c for c in commands if isinstance(c, str)])
                    if _bash_looks_like_filesystem(joined):
                        error_payload = {
                            "ok": False,
                            "error": "Bash is not allowed for file operations. Use MCP list_files/grep_files for listing and searching, or dedicated write/patch tools for edits.",
                        }
                        _emit_runtime_event(event_callback, "Tool validation error", error_payload)
                        messages.append(
                            {
                                "role": "user",
                                "content": "TOOL_RESULT " + json.dumps(error_payload),
                            }
                        )
                        break

                    invalid = False
                    for c in commands:
                        if not isinstance(c, str) or len(c) > MAX_CMD_CHARS:
                            error_payload = {
                                "ok": False,
                                "error": "invalid or too large bash command",
                            }
                            _emit_runtime_event(event_callback, "Tool validation error", error_payload)
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "TOOL_RESULT " + json.dumps(error_payload),
                                }
                            )
                            invalid = True
                            break
                    if invalid:
                        break

                    print("!" * 50)
                    print("⚠️  BASH TOOL REQUEST ⚠️")
                    print("!" * 50)
                    print(f"(cwd = {workdir})")
                    for i, c in enumerate(commands, 1):
                        print(f"{i}) {c}")
                    print("!" * 50 + "\n")
                    _emit_runtime_event(
                        event_callback,
                        "Bash tool request",
                        "cwd = " + workdir + "\n" + "\n".join(f"{i}) {c}" for i, c in enumerate(commands, 1)),
                    )

                    if not _confirm(
                        f"Execute {len(commands)} command(s)? (y/N): ",
                        no_confirm,
                        confirm_func,
                    ):
                        print("Command(s) cancelled.\n")
                        denied_payload = {"ok": False, "denied": True}
                        _emit_runtime_event(event_callback, "Bash tool denied", denied_payload)
                        messages.append(
                            {
                                "role": "user",
                                "content": "TOOL_RESULT " + json.dumps(denied_payload),
                            }
                        )
                        tool_gate_open = True
                        break

                    _raise_if_cancelled(cancel_event, event_callback)
                    _emit_runtime_event(event_callback, "Bash tool running", f"Executing {len(commands)} command(s)...")
                    results = []
                    for command in commands:
                        result = run_bash_command(command, cwd=workdir, cancel_event=cancel_event)
                        results.append(result)
                        if result.get("returncode") != 0:
                            break  # Do not run remaining commands after failure.
                    _raise_if_cancelled(cancel_event, event_callback)
                    result_payload = {"ok": all(r.get("returncode") == 0 for r in results), "results": results}
                    _emit_runtime_event(event_callback, "Bash tool result", result_payload)
                    messages.append(
                        {
                            "role": "user",
                            "content": "TOOL_RESULT " + json.dumps(result_payload),
                        }
                    )
                    tool_gate_open = True
                    break

                if tool == "mcp":
                    name = obj.get("name")
                    arguments = obj.get("arguments", {})

                    if not isinstance(name, str) or not isinstance(arguments, dict):
                        error_payload = {
                            "ok": False,
                            "error": "mcp tool requires name(str) and arguments(object)",
                        }
                        _emit_runtime_event(event_callback, "Tool validation error", error_payload)
                        messages.append(
                            {
                                "role": "user",
                                "content": "TOOL_RESULT " + json.dumps(error_payload),
                            }
                        )
                        break

                    allowed_tool = (mcp.has_tool(name) if hasattr(mcp, "has_tool")
                                    else name in MCP_TOOLS)
                    if not allowed_tool:
                        reason = (mcp.explain_unavailable_tool(name)
                                  if hasattr(mcp, "explain_unavailable_tool")
                                  else f"unknown or unavailable MCP tool: {name}")
                        error_payload = {"ok": False, "tool": "mcp", "name": name,
                                         "error": reason}
                        _emit_runtime_event(event_callback, "Tool validation error", error_payload)
                        messages.append({"role": "user", "content": "TOOL_RESULT " + json.dumps(error_payload)})
                        break

                    # Normalize simple ls/grep calls from older prompts/models.
                    # No shell is executed for these requests.
                    if name == "run_bash":
                        routed = _route_readonly_file_command(arguments.get("command"))
                        if routed is not None:
                            name, arguments = routed
                            _emit_runtime_event(event_callback, "Protocol repair", f"Routing run_bash to MCP {name}")

                    # MCP run_bash must not bypass the policy for direct bash.
                    if name == "run_bash" and (
                        not isinstance(arguments.get("command"), str)
                        or _bash_looks_like_filesystem(arguments["command"])
                    ):
                        error_payload = {"ok": False, "error": "Use MCP list_files(path=...) for ls, grep_files(pattern=..., path=...) for grep, and other MCP file tools for edits. Compound shell file commands are not supported."}
                        _emit_runtime_event(event_callback, "Tool validation error", error_payload)
                        messages.append({"role": "user", "content": "TOOL_RESULT " + json.dumps(error_payload)})
                        break

                    needs_confirm = (mcp.needs_confirmation(name) if hasattr(mcp, "needs_confirmation")
                                     else name in MCP_CONFIRM_TOOLS)

                    print("!" * 50)
                    print("⚠️  MCP TOOL REQUEST ⚠️")
                    print("!" * 50)
                    print(f"name: {name}")
                    print(
                        f"arguments: {json.dumps(arguments, ensure_ascii=False)[:2000]}"
                    )
                    print("!" * 50 + "\n")
                    _emit_runtime_event(
                        event_callback,
                        "MCP tool request",
                        f"name: {name}\narguments: {json.dumps(arguments, ensure_ascii=False, indent=2)}",
                    )

                    if needs_confirm and not _confirm(
                        "Execute this MCP tool? (y/N): ", no_confirm, confirm_func
                    ):
                        print("MCP tool cancelled.\n")
                        denied_payload = {"ok": False, "denied": True}
                        _emit_runtime_event(event_callback, "MCP tool denied", denied_payload)
                        messages.append(
                            {
                                "role": "user",
                                "content": "TOOL_RESULT " + json.dumps(denied_payload),
                            }
                        )
                        tool_gate_open = True
                        break

                    try:
                        _emit_runtime_event(event_callback, "MCP tool running", f"{name}(...)")
                        res = mcp.call_tool(name, arguments)
                        # Some MCP tools report errors in their result object
                        # instead of raising an exception (notably patch failures).
                        structured = res.get("structured") if isinstance(res, dict) else None
                        if isinstance(structured, dict):
                            payload = structured.get("result", structured)
                        else:
                            payload = None
                        tool_failed = isinstance(res, dict) and (
                            res.get("ok") is False or res.get("isError") is True
                            or (isinstance(payload, dict) and payload.get("ok") is False)
                        )
                        if name.startswith("freecad."):
                            # FreeCAD returns success=false in JSON text on
                            # some failures; MCP isError alone may be false.
                            from freecad_verify import response_failed
                            tool_failed = tool_failed or response_failed(res)
                        result_payload = {
                            "ok": not tool_failed,
                            "tool": "mcp",
                            "name": name,
                            "result": res,
                        }
                        if not tool_failed and name in ("freecad.create_object", "freecad.edit_object"):
                            from freecad_verify import verify_mutation
                            verification = verify_mutation(mcp, name, arguments, res)
                            result_payload["verification"] = verification
                            if verification.get("status") in ("mismatch", "failed"):
                                result_payload["ok"] = False

                        # Never treat a saved path as if the model had seen its
                        # pixels. Image feedback requires explicitly configured
                        # multimodal input or a separate vision model.
                        from agent_vision import image_feedback
                        result_text = "TOOL_RESULT " + json.dumps(result_payload)
                        vision, image_message = image_feedback(res, result_text)
                        if vision:
                            result_payload["vision"] = vision
                        result_text = "TOOL_RESULT " + json.dumps(result_payload)
                        if image_message is not None:
                            image_message[0]["text"] = result_text
                        _emit_runtime_event(event_callback, "MCP tool result", result_payload)
                        messages.append(
                            {
                                "role": "user",
                                "content": image_message if image_message is not None else result_text,
                            }
                        )
                        tool_gate_open = True
                        break
                    except Exception as e:
                        error_payload = {
                            "ok": False,
                            "tool": "mcp",
                            "name": name,
                            "error": str(e),
                        }
                        _emit_runtime_event(event_callback, "MCP tool error", error_payload)
                        messages.append(
                            {
                                "role": "user",
                                "content": "TOOL_RESULT " + json.dumps(error_payload),
                            }
                        )
                        break

                error_payload = {"ok": False, "error": f"unknown tool: {tool}"}
                _emit_runtime_event(event_callback, "Tool validation error", error_payload)
                messages.append(
                    {
                        "role": "user",
                        "content": "TOOL_RESULT " + json.dumps(error_payload),
                    }
                )
                break

        # continue outer loop

    print("Agent: Reached tool-round limit.\n")
    _emit_runtime_event(event_callback, "Turn stopped", "Reached tool-round limit.")



# A llama.cpp task ID identifies an individual inference request. It is NOT
# reusable for conversation recovery. The stable cache identity is id_slot.
# A server-side slot snapshot also survives slot eviction / server restarts,
# provided llama-server was started with --slot-save-path.
CONTEXT_FORMAT = "agent-context"
CONTEXT_VERSION = 2
CONTEXT_FILE = Path("ctx.json")


def _llama_cache_info(backend: Any) -> Optional[Dict[str, Any]]:
    cfg = getattr(backend, "cfg", None)
    if getattr(cfg, "server", None) != "llama.cpp":
        return None
    slot_id = getattr(backend, "id_slot", None)
    if type(slot_id) is not int or slot_id < 0:
        return None
    return {
        "server_url": cfg.url,
        "model": cfg.model,
        "slot_id": slot_id,
        "cache_prompt": bool(getattr(backend, "cache_prompt", False)),
    }


def _llama_slot_endpoint(server_url: str, slot_id: int, action: str) -> str:
    """Return the llama.cpp native slot API endpoint, not the /v1 API."""
    if type(slot_id) is not int or slot_id < 0 or action not in ("save", "restore"):
        raise ValueError("Invalid slot or action")
    url = urllib.parse.urlsplit(server_url)
    if url.scheme not in ("http", "https") or not url.netloc:
        raise ValueError("Invalid llama.cpp server URL")
    path = url.path.rstrip("/")
    if not path.endswith("/v1/chat/completions"):
        raise ValueError("llama.cpp URL must end with /v1/chat/completions")
    base = path[:-len("/v1/chat/completions")]
    return urllib.parse.urlunsplit((
        url.scheme, url.netloc, f"{base}/slots/{slot_id}", f"action={action}", ""
    ))


def _slot_snapshot_filename(context_path: Path, cache: Dict[str, Any]) -> str:
    """A stable filename avoids accumulating server-side snapshots on each save."""
    key = json.dumps([
        str(context_path.resolve()), cache["server_url"], cache["model"], cache["slot_id"]
    ], separators=(",", ":"))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return f"agent-context-{digest}.bin"


def _slot_snapshot_request(backend: Any, slot_id: int, action: str, filename: str) -> Dict[str, Any]:
    # Use the existing session: it already knows the llama.cpp API key.
    url = _llama_slot_endpoint(backend.cfg.url, slot_id, action)
    if (not isinstance(filename, str) or not filename or
            Path(filename).name != filename or "/" in filename or "\\" in filename):
        raise ValueError("Invalid slot snapshot filename")
    response = backend.session.post(url, json={"filename": filename}, timeout=(5, 120))
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    result = response.json()
    if not isinstance(result, dict) or result.get("id_slot") != slot_id:
        raise RuntimeError("Invalid slot snapshot response")
    return result


def _save_agent_context(
    path: Path, messages: List[Dict[str, str]], backend: Any
) -> str:
    """Always save chat history; snapshot failure only disables cache acceleration."""
    cache = _llama_cache_info(backend)
    status = "No llama.cpp slot cache (messages saved)."
    if cache is not None:
        cache["snapshot_filename"] = _slot_snapshot_filename(path, cache)
        try:
            result = _slot_snapshot_request(
                backend, cache["slot_id"], "save", cache["snapshot_filename"]
            )
            cache["snapshot_saved"] = True
            status = (f"llama.cpp slot {cache['slot_id']} snapshot saved on server "
                      f"({result.get('n_saved', '?')} tokens).")
        except Exception as exc:
            cache["snapshot_saved"] = False
            status = (f"Messages and slot {cache['slot_id']} saved; no disk snapshot "
                      f"({exc}). Live in-memory cache may still be reusable.")

    state: Dict[str, Any] = {
        "format": CONTEXT_FORMAT,
        "version": CONTEXT_VERSION,
        "messages": messages,
        "llama_cache": cache,
    }
    # Atomic replacement ensures a crash cannot destroy the last valid save.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".ctx-", suffix=".tmp", delete=False) as fd:
            tmp_name = fd.name
            json.dump(state, fd, ensure_ascii=False)
            fd.flush()
            os.fsync(fd.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
    return status


def _load_agent_context(
    path: Path, backend: Any
) -> tuple[List[Dict[str, str]], str]:
    with Path(path).open("r", encoding="utf-8") as fd:
        state = json.load(fd)

    # Backward compatibility: older ctx.json files were bare message arrays.
    if isinstance(state, list):
        messages = state
        saved_cache = None
        status = "Legacy context loaded; no saved slot metadata."
    elif isinstance(state, dict) and state.get("format") == CONTEXT_FORMAT:
        if state.get("version") != CONTEXT_VERSION:
            raise ValueError("Unsupported context format version")
        messages = state.get("messages")
        saved_cache = state.get("llama_cache")
        status = "Context loaded."
    else:
        raise ValueError("Unrecognized context file")

    if (not isinstance(messages, list) or not messages or
            any(not isinstance(x, dict) or x.get("role") not in
                ("system", "user", "assistant", "tool") or
                not isinstance(x.get("content"), str) for x in messages)):
        raise ValueError("Invalid messages in context file")

    current_cache = _llama_cache_info(backend)
    if not isinstance(saved_cache, dict):
        return messages, status
    if current_cache is None:
        return messages, status + " Saved llama.cpp cache unavailable with this backend."
    if (saved_cache.get("server_url") != current_cache["server_url"] or
            saved_cache.get("model") != current_cache["model"]):
        return messages, status + " Cache server/model mismatch; using chat history."
    slot_id = saved_cache.get("slot_id")
    if type(slot_id) is not int or slot_id < 0:
        return messages, status + " Invalid saved slot ID; using chat history."

    # Set the slot for future completions, even when disk snapshots are disabled.
    backend.id_slot = slot_id
    if not saved_cache.get("snapshot_saved"):
        return messages, (status + f" Reusing slot {slot_id} if its in-memory "
                "cache is still present; chat history remains the source of truth.")

    filename = saved_cache.get("snapshot_filename")
    try:
        result = _slot_snapshot_request(backend, slot_id, "restore", filename)
        return messages, (status + f" Restored llama.cpp slot {slot_id} "
                f"({result.get('n_restored', '?')} tokens) from server snapshot.")
    except Exception as exc:
        return messages, (status + f" Slot {slot_id} snapshot restore failed ({exc}); "
                "falling back to chat history / available live cache.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", action="version", version=f"%(prog)s {AGENT_VERSION}")
    p.add_argument(
        "--server",
        choices=["llama.cpp", "chatgpt", "openai"],
        default="llama.cpp",
        help="Which backend to use: llama.cpp or chatgpt(openai).",
    )
    p.add_argument(
        "-u",
        "--url",
        default=DEFAULT_SERVER_URL,
        help="llama.cpp server URL (ignored for chatgpt).",
    )
    p.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model name/id. For chatgpt: e.g. gpt-5. For llama.cpp: your local model name.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Optional API key for llama.cpp-compatible servers (ignored for chatgpt; uses OPENAI_API_KEY).",
    )
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temp", type=float, default=0.9)
    p.add_argument(
        "--no-confirm",
        action="store_true",
        help="DANGER: Execute tools without confirmation.",
    )
    p.add_argument(
        "--no-json-schema",
        action="store_true",
        help="Do not try response_format json_schema (use json_object / prompt-only).",
    )
    p.add_argument(
        "--debug", action="store_true", help="Print raw model outputs (truncated)."
    )
    args = p.parse_args()

    model = args.model
    if not model:
        model = (
            DEFAULT_MODEL_OPENAI
            if args.server in ("chatgpt", "openai")
            else DEFAULT_MODEL_LLAMA
        )

    from llm_backends import BackendConfig, make_backend

    cfg = BackendConfig(
        server=args.server,
        model=model,
        url=args.url,
        api_key=args.api_key,
        prefer_json_schema=(not args.no_json_schema),
    )
    backend = make_backend(cfg)

    print("\n=== AI Agent (MCP-first) ===")
    print(f"Backend: {args.server} | Model: {model}")
    if args.server == "llama.cpp":
        print(f"URL: {args.url}")
    else:
        print("Using OPENAI_API_KEY from environment.\n")

    if not backend.check_connection():
        sys.exit(1)

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    workspace = tempfile.TemporaryDirectory(prefix="ai-bash-agent-")
    mcp = _start_mcp_for_workspace(workspace.name)

    print(f"Workspace: {workspace.name}")
    if hasattr(mcp, "status"):
        print("MCP servers: " + json.dumps(mcp.status()["servers"], ensure_ascii=False))
    print("Type 'exit' to quit, 'clear' to clear context (and reset workspace+mcp), 'status' to show status.\n")

    try:
        while True:
            try:
                user_input = ""
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    print("\nExiting...")
                    break

                if not user_input:
                    continue

                if user_input.lower() == "exit":
                    break

                if user_input.lower() == "clear":
                    messages = messages[:1]
                    old = workspace.name

                    try:
                        mcp.close()
                    except Exception:
                        pass
                    workspace.cleanup()

                    workspace = tempfile.TemporaryDirectory(prefix="ai-bash-agent-")
                    mcp = _start_mcp_for_workspace(workspace.name)

                    print(
                        f"Context cleared.\nWorkspace reset:\n  old: {old}\n  new: {workspace.name}\n"
                    )
                    continue
                if user_input.lower() == "save":
                    try:
                        status = _save_agent_context(CONTEXT_FILE, messages, backend)
                        print(f"Context saved to {CONTEXT_FILE}. {status}")
                    except Exception as exc:
                        print(f"Context save failed: {exc}")
                    continue

                if user_input.lower() == "load":
                    try:
                        restored_messages, status = _load_agent_context(CONTEXT_FILE, backend)
                        messages = restored_messages
                        print(status)
                        print("Context ready. Enter your next request.")
                    except Exception as exc:
                        print(f"Context load failed: {exc}")
                    continue

                if user_input.lower() == "handoff":
                    print("Preparing handoff...\n")

                    handoff_summary = _request_handoff_summary(
                        backend=backend,
                        messages=messages,
                        max_tokens=args.max_tokens,
                    )

                    project_status = _build_project_status(workspace.name)

                    # Reset ONLY the LLM context, keep workspace and MCP as-is
                    messages = [messages[0]]

                    handoff_payload = (
                        "HANDOFF CONTEXT\n"
                        "===============\n\n"
                        "Summary from previous context:\n"
                        f"{handoff_summary}\n\n"
                        "Current project status:\n"
                        f"{project_status}\n\n"
                        "Continue from this state. "
                        "The workspace and current project files were preserved during handoff."
                    )

                    messages.append({"role": "user", "content": handoff_payload})

                    print("Handoff complete. Context was reset, workspace preserved.\n")
                    continue

                if user_input.lower() == "status":
                    print(
                        f"Server: {args.server} | Model: {model} | Messages: {len(messages)}"
                    )
                    print(f"Workspace: {workspace.name}")
                    if hasattr(mcp, "status"):
                        print("MCP: " + json.dumps(mcp.status(), ensure_ascii=False))
                    print()
                    continue

                run_agent_turn(
                    backend=backend,
                    mcp=mcp,
                    messages=messages,
                    user_input=user_input,
                    max_tokens=args.max_tokens,
                    temperature=args.temp,
                    no_confirm=args.no_confirm,
                    workdir=workspace.name,
                    debug=args.debug,
                )

                if len(messages) > 21:
                    messages = [messages[0]] + messages[-20:]

            except KeyboardInterrupt:
                print("\nExiting...")
                break
    finally:
        try:
            mcp.close()
        except Exception:
            pass
        try:
            workspace.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()
