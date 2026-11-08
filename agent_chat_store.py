#!/usr/bin/env python3
"""Read and save chats named by their JSON files in save/.

A chat file is traditionally a bare JSON array of messages.  Do not convert it
into an agent-gui-chat wrapper merely because the GTK UI loaded and saved it.
Existing agent-context and agent-gui-chat envelopes also remain readable, and
are updated in their original format when saved.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CHAT_FORMAT = "agent-gui-chat"  # Backward compatibility with the first v24.
MAX_CHATS_LISTED = 2000
Signature = Tuple[int, int]


@dataclass(frozen=True)
class ChatEntry:
    path: Path
    title: str
    modified: float
    size_bytes: int = 0
    model: str = ""


def _plain_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(x.get("text", "") for x in content
                        if isinstance(x, dict) and x.get("type") == "text"
                        and isinstance(x.get("text"), str))
    return ""


def validate_messages(data: Any) -> List[Dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("Chat must contain a JSON array of messages")
    messages: List[Dict[str, Any]] = []
    for index, message in enumerate(data):
        if not isinstance(message, dict) or message.get("role") not in (
                "system", "user", "assistant", "tool"):
            raise ValueError(f"Invalid chat message at index {index}")
        if not isinstance(message.get("content"), (str, list)):
            raise ValueError(f"Invalid chat content at index {index}")
        messages.append(dict(message))
    return messages


def _legacy_transcript(messages: List[Dict[str, Any]]) -> str:
    """Recreate a readable transcript from the saved conversation messages.

    The older JSON arrays don't contain GTK's separate visual event log.
    Avoid printing huge TOOL_RESULT payloads into the GUI.
    """
    blocks: List[str] = []
    for message in messages:
        role = message.get("role")
        body = _plain_text(message.get("content", ""))
        if role == "user" and body and not body.startswith("TOOL_RESULT "):
            blocks.append("## You\n\n" + body[:12000])
        elif role == "assistant":
            try:
                obj = json.loads(body)
            except (ValueError, TypeError):
                obj = None
            if isinstance(obj, dict):
                if obj.get("type") == "message" and isinstance(obj.get("message"), str):
                    blocks.append("## Agent\n\n" + obj["message"][:12000])
                elif obj.get("type") == "tool":
                    tool_name = str(obj.get("name") or obj.get("tool") or "unknown")
                    blocks.append("### Tool: `" + tool_name[:180] + "`")
            elif body:
                blocks.append("## Agent\n\n" + body[:12000])
    return "\n\n".join(blocks) + ("\n\n" if blocks else "")


class ChatStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory).expanduser().resolve()

    def named_path(self, filename: str) -> Path:
        """Build a path for a new chat; reject traversal or accidental overwrite."""
        if not isinstance(filename, str):
            raise ValueError("Chat name must be text")
        name = filename.strip()
        if name and not name.lower().endswith(".json"):
            name += ".json"
        if (not name or name in (".", "..") or len(name) > 180
                or Path(name).name != name or "/" in name or "\\" in name
                or name.startswith(".")):
            raise ValueError("Use a filename such as libmanager.json (no directories)")
        return self.directory / name

    def new_path(self) -> Path:
        """Legacy API; UI intentionally uses named_path() instead."""
        return self.directory / f"chat-{uuid.uuid4().hex}.json"

    @staticmethod
    def signature(path: Path) -> Optional[Signature]:
        try:
            stat = Path(path).stat()
            return stat.st_mtime_ns, stat.st_size
        except FileNotFoundError:
            return None

    @staticmethod
    def load(path: Path) -> Dict[str, Any]:
        path = Path(path).expanduser().resolve()
        source = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(source, list):
            messages = validate_messages(source)
            transcript = _legacy_transcript(messages)
            fmt = "legacy-list"
            profile = {}
        elif isinstance(source, dict) and isinstance(source.get("messages"), list):
            messages = validate_messages(source["messages"])
            fmt = str(source.get("format", "legacy-object"))
            raw = source.get("transcript")
            transcript = raw if isinstance(raw, str) else _legacy_transcript(messages)
            profile = source.get("profile") if isinstance(source.get("profile"), dict) else {}
        else:
            raise ValueError("Not a saved conversation (expected message array)")
        return {
            "messages": messages,
            "title": path.name,
            "transcript": transcript,
            "profile": {k: v for k, v in profile.items()
                        if k in ("model", "server", "url") and isinstance(v, str)},
            "source_format": fmt,
            "signature": ChatStore.signature(path),
        }

    @staticmethod
    def _atomic_write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=path.parent,
                    prefix=".chat-", suffix=".tmp", delete=False) as fd:
                temp_path = fd.name
                json.dump(payload, fd, ensure_ascii=False, indent=2)
                fd.write("\n")
                fd.flush()
                os.fsync(fd.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

    def save(self, path: Path, messages: List[Dict[str, Any]], transcript: str = "",
             profile: Optional[Dict[str, Any]] = None, title: Optional[str] = None,
             *, expected_signature: Optional[Signature] = None,
             expect_missing: bool = False) -> Path:
        """Save using the existing file format; new chats use bare JSON arrays.

        expected_signature protects against another process changing a file
        after the sidebar loaded it.  Before first modification, save a .json.bak
        copy (not listed as another chat).
        """
        path = Path(path).expanduser().resolve()
        messages = validate_messages(messages)
        actual = self.signature(path)
        if expect_missing and actual is not None:
            raise FileExistsError(f"Chat already exists: {path.name}")
        if expected_signature is not None and actual != expected_signature:
            raise RuntimeError(f"Chat changed on disk: {path.name}. Refresh or load it again.")

        if actual is not None:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                payload: Any = messages
            elif isinstance(existing, dict) and isinstance(existing.get("messages"), list):
                # Preserve agent-context's llama_cache and v24's metadata. Do
                # not invent fields or modify their format.
                payload = dict(existing)
                payload["messages"] = messages
                if payload.get("format") == CHAT_FORMAT:
                    payload["transcript"] = str(transcript)
            else:
                raise ValueError(f"Refusing to overwrite non-chat JSON: {path.name}")
            backup = path.with_name(path.name + ".bak")
            if not backup.exists():
                # Exclusive creation prevents overwriting an older backup.
                with backup.open("xb") as out, path.open("rb") as inp:
                    shutil.copyfileobj(inp, out)
        else:
            payload = messages
        self._atomic_write(path, payload)
        return path

    def list_chats(self, limit: int = MAX_CHATS_LISTED) -> List[ChatEntry]:
        """List existing save/*.json by filename, without parsing each chat."""
        if not self.directory.is_dir():
            return []
        entries: List[ChatEntry] = []
        for path in self.directory.glob("*.json"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append(ChatEntry(path, path.name, stat.st_mtime,
                                     stat.st_size))
        entries.sort(key=lambda entry: (-entry.modified, entry.title.casefold()))
        return entries[:limit]


def format_timestamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")
