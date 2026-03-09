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
Do not include markdown.
Do not include code fences.
Do not include any text before or after the JSON object.
Do not output keys other than: type, plan, message, tool, commands.
When emitting shell commands containing regex, ensure the JSON remains valid by escaping backslashes for JSON.
Example:
grep -oP '(\\d{1,2}:\\d{2})'

Schema:
{
  "type": "message" | "tool",
  "plan": ["short bullet", "short bullet"],
  "message": "user-facing text",

  "files": [                       // optional, for code / long text / markdown
    {"path": "relative/path.ext", "content": "file contents as a JSON string"}
  ],

  "patches": [                     // optional, for small edits (prefer over sed)
    {"path": "relative/path.ext", "diff": "unified diff string"}
  ],

  "tool": "bash",                  // only when type == "tool"
  "commands": ["cmd1", "cmd2"]     // only when type == "tool"
} // always close the brackets

Rules:
- plan: 1-3 short bullets. Do NOT output hidden step-by-step reasoning.
- NEVER claim a tool error unless you received a TOOL_RESULT that shows an error.
- If the user asks to compile/run/test/execute/verify, you MUST respond with type="tool" first.
- If type == "tool", output ONLY the tool JSON (do not include a final answer in the same reply).
- If you need to publish code/markdown: put it ONLY in files[].content (NOT in message).
- For modifying existing files: use patches[] (NOT sed/perl -pi).
- Use ONLY relative paths in files[].path and patches[].path (no /abs, no .., no ~).
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
    """
    if not raw:
        return None

    txt = strip_code_fences(raw)
    print(raw)
    dec = json.JSONDecoder()
    i = 0
    objs: List[Dict[str, Any]] = []

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

    s = txt.find("{")
    e = txt.rfind("}")
    if s != -1 and e != -1 and e > s:
        obj = json.loads(txt[s : e + 1])
        if isinstance(obj, dict):
            return [obj]

    return None


def _is_safe_relative_path(p: str) -> bool:
    if not p or not isinstance(p, str):
        return False
    # if p.startswith(("/", "~")):
    #     return False
    # Windows drive letters
    if re.match(r"^[A-Za-z]:[\\/]", p):
        return False
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
    """
    Extract file paths referenced by a unified diff. Best-effort.
    Returns normalized relative paths with a/ and b/ stripped when present.
    """
    paths: List[str] = []

    # diff --git a/foo b/foo
    for m in re.finditer(r"^diff --git a/(\S+) b/(\S+)\s*$", diff, re.MULTILINE):
        paths.extend([m.group(1), m.group(2)])

    # --- a/foo / +++ b/foo or --- foo / +++ foo
    for m in re.finditer(r"^(---|\+\+\+)\s+(\S+)\s*$", diff, re.MULTILINE):
        p = m.group(2)
        if p == "/dev/null":
            continue
        # Strip a/ b/ prefixes
        if p.startswith("a/") or p.startswith("b/"):
            p = p[2:]
        paths.append(p)

    # Dedup preserving order
    out: List[str] = []
    seen = set()
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def apply_patches_from_obj(obj: Dict[str, Any], workdir: str) -> Tuple[List[str], List[str]]:
    """
    Applies obj["patches"] using the system 'patch' utility (runtime-controlled, not model-controlled).
    Safety:
      - requires patches[].path to be safe relative
      - rejects diffs that reference other file paths
    Returns (patched_paths, error_messages).
    """
    patches = obj.get("patches")
    if not patches:
        return ([], [])

    if not isinstance(patches, list):
        return ([], ["patches must be an array"])

    root = Path(workdir).resolve()
    patched: List[str] = []
    errors: List[str] = []

    # Ensure patch tool exists
    try:
        subprocess.run(["patch", "--version"], capture_output=True, text=True, timeout=5)
    except Exception:
        return ([], ["'patch' tool is not available on this system. Install it (e.g. apt install patch)."])

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

        # Ensure target path stays within workspace
        target = (root / Path(path)).resolve()
        try:
            rel = target.relative_to(root)
        except Exception:
            errors.append(f"patches[{idx}] path escapes workspace: {path!r}")
            continue

        # Validate diff only references this file (best-effort)
        referenced = _extract_paths_from_unified_diff(diff)
        bad_refs = []
        for rp in referenced:
            if not _is_safe_relative_path(rp):
                bad_refs.append(rp)
            elif rp != path:
                bad_refs.append(rp)

        if bad_refs:
            errors.append(
                f"patches[{idx}] diff references other/unsafe paths: {bad_refs!r} (expected only {path!r})"
            )
            continue

        # Apply diff. Try git-style first (-p1), then plain (-p0).
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


def run_agent_turn(
    backend: Any,
    messages: List[Dict[str, str]],
    user_input: str,
    max_tokens: int,
    temperature: float,
    no_confirm: bool,
    workdir: str,
) -> None:
    messages.append({"role": "user", "content": user_input})

    need_tool = user_requires_tool(user_input)
    tool_was_used = False

    for _round in range(MAX_TOOL_ROUNDS):
        raw = backend.chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature)
        if not raw:
            print("✗ No response from model.")
            return
        
        try:
            objs = parse_assistant_json_objects(raw)
        except Exception as ex:
            print(raw)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "system", "content": f"You're answer violates JSON structure. json.loads raise exception: {ex}"})
            continue


        if not objs:
            print("Agent: (protocol violation: expected JSON)\n")
            print(raw)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "system", "content": "You are violate JSON structure. Provided answer is not JSON. Correct previous request and try again."})
            continue

        if len(objs) > 1:
            print("⚠️ Model returned multiple JSON objects; tools will be executed first and extra messages ignored.\n")

        for obj in objs:
            if not isinstance(obj, dict) or "type" not in obj or "plan" not in obj:
                messages.append({
                    "role": "user",
                    "content": "TOOL_RESULT " + json.dumps({"ok": False, "error": "invalid JSON schema; reply with one valid object"})
                })
                break

            # Persist assistant JSON for continuity
            messages.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})

            # Apply files then patches (so patches can target freshly written files too)
            written, w_err = write_files_from_obj(obj, workdir)
            patched, p_err = apply_patches_from_obj(obj, workdir)

            if written:
                print("\n" + "=" * 50)
                print("📄 FILES WRITTEN")
                print("=" * 50)
                for p in written:
                    print(f"- {p}")
                print("=" * 50 + "\n")

            if patched:
                print("\n" + "=" * 50)
                print("🩹 PATCHES APPLIED")
                print("=" * 50)
                for p in patched:
                    print(f"- {p}")
                print("=" * 50 + "\n")

            if w_err or p_err:
                print("\n" + "!" * 50)
                print("⚠️ FILE/PATCH ERRORS")
                print("!" * 50)
                for e in (w_err + p_err):
                    print(f"- {e}")
                print("!" * 50 + "\n")

                # Tell the model it must fix paths/diff/etc.
                messages.append({
                    "role": "user",
                    "content": "TOOL_RESULT " + json.dumps({
                        "ok": False,
                        "error": "Failed to write files or apply patches.",
                        "written": written,
                        "patched": patched,
                        "details": (w_err + p_err),
                    })
                })
                break  # next tool round

            render_assistant(obj)

            typ = obj.get("type")

            if typ == "message" and need_tool and not tool_was_used:
                messages.append({
                    "role": "user",
                    "content": "TOOL_RESULT " + json.dumps({
                        "ok": False,
                        "error": "Tool is required for this request. Reply with type='tool', tool='bash', and commands[] to compile/run/test. Do not guess outputs."
                    })
                })
                break

            if typ == "message":
                return

            if typ == "tool" and obj.get("tool") == "bash":
                commands = obj.get("commands", [])
                if not isinstance(commands, list) or not commands:
                    messages.append({
                        "role": "user",
                        "content": "TOOL_RESULT " + json.dumps({"ok": False, "error": "missing commands[]"})
                    })
                    tool_was_used = True
                    break

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
                        "content": "TOOL_RESULT " + json.dumps({"ok": False, "denied": True})
                    })
                    tool_was_used = True
                    break

                results = []
                for c in commands:
                    print(f"Executing: {c}")
                    r = run_bash_command(c, cwd=workdir)
                    results.append(r)
                    if r["stdout"]:
                        print("STDOUT:\n" + r["stdout"])
                    if r["stderr"]:
                        print("STDERR:\n" + r["stderr"])
                    print(f"Return Code: {r['returncode']}\n")

                messages.append({
                    "role": "user",
                    "content": "TOOL_RESULT " + json.dumps({"ok": True, "results": results})
                })
                tool_was_used = True
                break

            messages.append({
                "role": "user",
                "content": "TOOL_RESULT " + json.dumps({"ok": False, "error": "unknown tool/type; use type=message or type=tool(tool=bash)"})
            })
            tool_was_used = True
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
                    # optionally list workspace files
                    try:
                        ws = Path(workspace.name)
                        files = [str(p.relative_to(ws)) for p in ws.rglob("*") if p.is_file()]
                        if files:
                            print("Workspace files:")
                            for f in sorted(files)[:50]:
                                print(f"- {f}")
                            if len(files) > 50:
                                print(f"... ({len(files)-50} more)")
                        else:
                            print("Workspace files: (none)")
                        print()
                    except Exception:
                        pass
                    continue

                run_agent_turn(
                    backend=backend,
                    messages=messages,
                    user_input=user_input,
                    max_tokens=args.max_tokens,
                    temperature=args.temp,
                    no_confirm=args.no_confirm,
                    workdir=workspace.name,
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