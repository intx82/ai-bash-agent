#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_backends import BackendConfig, make_backend


DEFAULT_SERVER_URL = "http://192.168.3.11:8080/v1/chat/completions"
DEFAULT_MODEL_LLAMA = "qwen"
DEFAULT_MODEL_OPENAI = "gpt-5.4"  # change as you like


SYSTEM_PROMPT = r"""
You are a helpful AI assistant with access to a local bash tool.

You MUST respond with EXACTLY ONE JSON object and NOTHING ELSE.
NO markdown fences. NEVER output ```.
Output must be valid JSON parsable by json.loads().
Do not include any text before or after the JSON object.

Allowed top-level keys are: type, plan, message, tool, commands, files, patches.

When emitting shell commands containing regex, ensure the JSON remains valid by escaping backslashes for JSON.
Example:
grep -oP '(\\d{1,2}:\\d{2})'

Schema:
{
  "type": "message" | "tool",
  "plan": ["short bullet", "short bullet"],
  "message": "user-facing text",

  "files": [
    {"path": "relative/path.ext", "content": "file contents as a JSON string"}
  ],

  "patches": [
    {"path": "relative/path.ext", "diff": "unified diff string"}
  ],

  "tool": "bash",
  "commands": ["cmd1", "cmd2"]
}

Rules:
- plan: 1-3 short bullets. Do NOT output hidden step-by-step reasoning.
- NEVER claim a tool error unless you received a TOOL_RESULT that shows an error.
- If the user asks to compile/run/test/execute/verify, you MUST respond with type="tool" first.
- If type == "tool", output ONLY the tool JSON (do not include a final answer in the same reply).
- If you need to publish code/markdown: put it ONLY in files[].content (NOT in message).
- For modifying existing files: use patches[] (NOT sed/perl -pi).
- For writing new files: use files[] (NOT sed/echo/python/cat heredocs).
- Bash commands run in a temporary working directory (empty sandbox). Use relative paths and create files there.
- Avoid dangerous commands unless user explicitly requests and confirms.
- If the user asks found something in the internet use DuckDuckGo (`ddgr [QUERY] --np --json`). 

DuckDuckGo (ddgr) Output JSON will have next schema:

[
  {
    "abstract": "abstract description",
    "title": "Title",
    "url": "https://[url]"
  },
]

- If there is a need to fetch page from internet use a 'curl' and 'wget' tools, and then parse the output.
- If you don't know something or need to proof it, use DuckDuckGo to search additional information
"""

MAX_TOOL_ROUNDS = 32
MAX_CMD_CHARS = 4000


def user_requires_tool(user_input: str) -> bool:
    s = user_input.lower()
    keywords = ("compile", "build", "run", "execute", "test", "verify", "benchmark")
    return any(k in s for k in keywords)


def run_bash_command(command: str, cwd: str, timeout_s: int = 300) -> Dict[str, Any]:
    try:
        r = subprocess.run(
            ["bash", "-lc", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {"command": command, "cwd": cwd, "stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode}
    except subprocess.TimeoutExpired:
        return {"command": command, "cwd": cwd, "stdout": "", "stderr": f"timeout after {timeout_s}s", "returncode": 124}
    except Exception as e:
        return {"command": command, "cwd": cwd, "stdout": "", "stderr": str(e), "returncode": 1}


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_assistant_json_objects(raw: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parses one or more JSON objects concatenated together (NDJSON-ish).
    Returns a list of dicts, or None if parsing fails.
    Never raises.
    """
    if raw is None:
        return None

    txt = strip_code_fences(raw)

    dec = json.JSONDecoder()
    i = 0
    objs: List[Dict[str, Any]] = []

    # raw-decode loop (best for {}{} or {}\n{} cases)
    try:
        while i < len(txt):
            while i < len(txt) and txt[i].isspace():
                i += 1
            if i >= len(txt):
                break
            obj, j = dec.raw_decode(txt, i)
            if isinstance(obj, dict):
                objs.append(obj)
            i = j
        if objs:
            return objs
    except Exception:
        pass

    # fallback: extract first {...} or whole string
    try:
        s = txt.find("{")
        e = txt.rfind("}")
        if s != -1 and e != -1 and e > s:
            obj = json.loads(txt[s : e + 1])
            if isinstance(obj, dict):
                return [obj]
        # Final fallback: whole string
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        return None

    return None


def _is_safe_relative_path(p: str) -> bool:
    if not p or not isinstance(p, str):
        return False
    # if p.startswith(("/", "~")):
    #     return False
    # if re.match(r"^[A-Za-z]:[\\/]", p):  # Windows drive letters
    #     return False
    pp = Path(p)
    if any(part == ".." for part in pp.parts):
        return False
    return True


def write_files_from_obj(obj: Dict[str, Any], workdir: str) -> Tuple[List[str], List[str]]:
    """
    Writes obj["files"] to workdir (safe relative paths only).
    Returns (written_paths, error_messages).
    """
    files = obj.get("files")
    if not files:
        return ([], [])

    if not isinstance(files, list):
        return ([], ["files must be an array"])

    written: List[str] = []
    errors: List[str] = []

    root = Path(workdir).resolve()

    for idx, item in enumerate(files):
        if not isinstance(item, dict):
            errors.append(f"files[{idx}] is not an object")
            continue

        path = item.get("path")
        content = item.get("content")

        if not isinstance(path, str) or not isinstance(content, str):
            errors.append(f"files[{idx}] missing/invalid path or content")
            continue
        if not _is_safe_relative_path(path):
            errors.append(f"files[{idx}] unsafe path: {path!r}")
            continue

        target = (root / Path(path)).resolve()
        try:
            rel = target.relative_to(root)
        except Exception:
            errors.append(f"files[{idx}] path escapes workspace: {path!r}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(content, encoding="utf-8")
            written.append(str(rel).replace("\\", "/"))
        except Exception as e:
            errors.append(f"files[{idx}] write failed for {path!r}: {e}")

    return (written, errors)


def _extract_paths_from_unified_diff(diff: str) -> List[str]:
    paths: List[str] = []
    for m in re.finditer(r"^diff --git a/(\S+) b/(\S+)\s*$", diff, re.MULTILINE):
        paths.extend([m.group(1), m.group(2)])
    for m in re.finditer(r"^(---|\+\+\+)\s+(\S+)\s*$", diff, re.MULTILINE):
        p = m.group(2)
        if p == "/dev/null":
            continue
        if p.startswith("a/") or p.startswith("b/"):
            p = p[2:]
        paths.append(p)

    out: List[str] = []
    seen = set()
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def apply_patches_from_obj(obj: Dict[str, Any], workdir: str) -> Tuple[List[str], List[str]]:
    patches = obj.get("patches")
    if not patches:
        return ([], [])
    if not isinstance(patches, list):
        return ([], ["patches must be an array"])

    root = Path(workdir).resolve()
    patched: List[str] = []
    errors: List[str] = []

    try:
        subprocess.run(["patch", "--version"], capture_output=True, text=True, timeout=5)
    except Exception:
        return ([], ["'patch' tool is not available. Install it (e.g. apt install patch)."])

    for idx, item in enumerate(patches):
        if not isinstance(item, dict):
            errors.append(f"patches[{idx}] is not an object")
            continue

        path = item.get("path")
        diff = item.get("diff")

        if not isinstance(path, str) or not isinstance(diff, str):
            errors.append(f"patches[{idx}] missing/invalid path or diff")
            continue
        if not _is_safe_relative_path(path):
            errors.append(f"patches[{idx}] unsafe path: {path!r}")
            continue

        target = (root / Path(path)).resolve()
        try:
            rel = target.relative_to(root)
        except Exception:
            errors.append(f"patches[{idx}] path escapes workspace: {path!r}")
            continue

        referenced = _extract_paths_from_unified_diff(diff)
        bad_refs = []
        for rp in referenced:
            if not _is_safe_relative_path(rp):
                bad_refs.append(rp)
            elif rp != path:
                bad_refs.append(rp)

        if bad_refs:
            errors.append(f"patches[{idx}] diff references other/unsafe paths: {bad_refs!r} (expected only {path!r})")
            continue

        applied = False
        last = ""
        for pstrip in (1, 0):
            try:
                r = subprocess.run(
                    ["patch", f"-p{pstrip}", "--forward", "--batch"],
                    cwd=workdir,
                    input=diff,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if r.returncode == 0:
                    applied = True
                    break
                last = (r.stdout + "\n" + r.stderr).strip()
            except Exception as e:
                last = str(e)

        if not applied:
            errors.append(f"patches[{idx}] failed to apply to {path!r}: {last[:400]}")
            continue

        patched.append(str(rel).replace("\\", "/"))

    return (patched, errors)


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


def _bad_edit_command(cmd: str) -> Optional[str]:
    if re.search(r"\bsed\s+-i\b", cmd):
        return "sed -i"
    if re.search(r"\bperl\s+-pi\b", cmd):
        return "perl -pi"
    return None


def run_agent_turn(
    backend: Any,
    messages: List[Dict[str, str]],
    user_input: str,
    max_tokens: int,
    temperature: float,
    no_confirm: bool,
    workdir: str,
    debug: bool = False,
) -> None:
    messages.append({"role": "user", "content": user_input})
    need_tool = user_requires_tool(user_input)

    # gate becomes True only after bash executed or explicitly denied
    tool_gate_open = not need_tool

    for _round in range(MAX_TOOL_ROUNDS):
        raw = backend.chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature)

        if raw is None:
            print("✗ No response from model.")
            return

        if debug:
            print(f"---- RAW (first 1000) ----\n{raw[:1000]}\n-------------------------")

        # Empty completion is a real case. Treat it as "retry", not protocol violation.
        if raw.strip() == "":
            messages.append({
                "role": "user",
                "content": "TOOL_RESULT " + json.dumps({
                    "ok": False,
                    "error": "Empty response. Reply again with exactly ONE valid JSON object.",
                }),
            })
            continue

        objs = parse_assistant_json_objects(raw)
        if not objs:
            # Do NOT add a system message (it grows prompts and hurts behavior).
            messages.append({
                "role": "user",
                "content": "TOOL_RESULT " + json.dumps({
                    "ok": False,
                    "error": "Your previous reply was not valid JSON. Reply again with exactly ONE JSON object and no extra text.",
                    "raw_snippet": raw[:300],
                }),
            })
            continue

        # If any tool object exists, prioritize tool objects and ignore message objects in this reply.
        tool_objs = [o for o in objs if isinstance(o, dict) and o.get("type") == "tool" and o.get("tool") == "bash"]
        any_tool_obj = bool(tool_objs)

        if len(objs) > 1 and any_tool_obj:
            print("⚠️ Multiple JSON objects received; prioritizing tool request.\n")

        ordered = tool_objs if any_tool_obj else objs

        for obj in ordered:
            if not isinstance(obj, dict) or "type" not in obj or "plan" not in obj:
                messages.append({
                    "role": "user",
                    "content": "TOOL_RESULT " + json.dumps({
                        "ok": False,
                        "error": "Invalid JSON schema; reply with one valid object.",
                    }),
                })
                break

            messages.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})

            written, w_err = write_files_from_obj(obj, workdir)
            patched, p_err = apply_patches_from_obj(obj, workdir)

            if w_err or p_err:
                messages.append({
                    "role": "user",
                    "content": "TOOL_RESULT " + json.dumps({
                        "ok": False,
                        "error": "Failed to write files or apply patches.",
                        "written": written,
                        "patched": patched,
                        "details": (w_err + p_err),
                    }),
                })
                break  # next round

            render_assistant(obj)
            typ = obj.get("type")

            if typ == "message":
                if need_tool and not tool_gate_open:
                    messages.append({
                        "role": "user",
                        "content": "TOOL_RESULT " + json.dumps({
                            "ok": False,
                            "error": "Tool is required. Reply with type='tool', tool='bash', and commands[] to compile/run/test. Do not guess outputs.",
                        }),
                    })
                    break
                return

            if typ == "tool" and obj.get("tool") == "bash":
                commands = obj.get("commands", [])
                if not isinstance(commands, list) or not commands:
                    messages.append({
                        "role": "user",
                        "content": "TOOL_RESULT " + json.dumps({"ok": False, "error": "missing commands[]"}),
                    })
                    break

                # runtime guards (do NOT open tool gate on these)
                guard_failed = False
                for c in commands:
                    if len(c) > MAX_CMD_CHARS:
                        messages.append({
                            "role": "user",
                            "content": "TOOL_RESULT " + json.dumps({
                                "ok": False,
                                "error": f"Command too large ({len(c)} chars). Put code into files[] and use short commands for build/run.",
                            }),
                        })
                        guard_failed = True
                        break
                    bad = _bad_edit_command(c)
                    if bad:
                        messages.append({
                            "role": "user",
                            "content": "TOOL_RESULT " + json.dumps({
                                "ok": False,
                                "error": f"Do not use {bad} to edit code. Use patches[] instead.",
                            }),
                        })
                        guard_failed = True
                        break
                if guard_failed:
                    break  # next round

                print("!" * 50)
                print("⚠️  BASH TOOL REQUEST ⚠️")
                print("!" * 50)
                print(f"(cwd = {workdir})")
                for i, c in enumerate(commands, 1):
                    print(f"{i}) {c}")
                print("!" * 50 + "\n")

                if no_confirm:
                    confirm = "y"
                else:
                    confirm = input(f"Execute {len(commands)} command(s)? (y/N): ").strip().lower()

                if confirm != "y":
                    print("Command(s) cancelled.\n")
                    messages.append({
                        "role": "user",
                        "content": "TOOL_RESULT " + json.dumps({"ok": False, "denied": True}),
                    })
                    tool_gate_open = True  # denied => allow model to answer without tool
                    break

                results = []
                for c in commands:
                    r = run_bash_command(c, cwd=workdir)
                    results.append(r)

                messages.append({
                    "role": "user",
                    "content": "TOOL_RESULT " + json.dumps({"ok": True, "results": results}),
                })
                tool_gate_open = True  # executed => allow message response
                break  # next round

            # unknown type/tool
            messages.append({
                "role": "user",
                "content": "TOOL_RESULT " + json.dumps({
                    "ok": False,
                    "error": "unknown tool/type; use type=message or type=tool(tool=bash)",
                }),
            })
            break

    print("Agent: Reached tool-round limit.\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", choices=["llama.cpp", "chatgpt", "openai"], default="llama.cpp",
                   help="Which backend to use: llama.cpp or chatgpt(openai).")
    p.add_argument("-u", "--url", default=DEFAULT_SERVER_URL,
                   help="llama.cpp server URL (ignored for chatgpt).")
    p.add_argument("-m", "--model", default=None,
                   help="Model name/id. For chatgpt: e.g. gpt-5. For llama.cpp: your local model name.")
    p.add_argument("--api-key", default=None,
                   help="Optional API key for llama.cpp-compatible servers (ignored for chatgpt; uses OPENAI_API_KEY).")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--no-confirm", action="store_true",
                   help="DANGER: Execute bash commands without confirmation.")
    p.add_argument("--no-json-schema", action="store_true",
                   help="Do not try response_format json_schema (use json_object / prompt-only).")
    p.add_argument("--debug", action="store_true", help="Print raw model outputs (truncated).")
    args = p.parse_args()

    model = args.model
    if not model:
        model = DEFAULT_MODEL_OPENAI if args.server in ("chatgpt", "openai") else DEFAULT_MODEL_LLAMA

    cfg = BackendConfig(
        server=args.server,
        model=model,
        url=args.url,
        api_key=args.api_key,
        prefer_json_schema=(not args.no_json_schema),
    )
    backend = make_backend(cfg)

    print("\n=== AI Bash Agent ===")
    print(f"Backend: {args.server} | Model: {model}")
    if args.server == "llama.cpp":
        print(f"URL: {args.url}")
    else:
        print("Using OPENAI_API_KEY from environment.\n")

    if not backend.check_connection():
        sys.exit(1)

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    workspace = tempfile.TemporaryDirectory(prefix="ai-bash-agent-")
    print(f"Workspace: {workspace.name}\n")
    print("Type 'exit' to quit, 'clear' to clear context (and reset workspace), 'status' to show status.\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

                if user_input.lower() == "exit":
                    break

                if user_input.lower() == "clear":
                    messages = messages[:1]
                    old = workspace.name
                    workspace.cleanup()
                    workspace = tempfile.TemporaryDirectory(prefix="ai-bash-agent-")
                    print(f"Context cleared.\nWorkspace reset:\n  old: {old}\n  new: {workspace.name}\n")
                    continue

                if user_input.lower() == "status":
                    print(f"Server: {args.server} | Model: {model} | Messages: {len(messages)}")
                    print(f"Workspace: {workspace.name}\n")
                    continue

                run_agent_turn(
                    backend=backend,
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
            workspace.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()