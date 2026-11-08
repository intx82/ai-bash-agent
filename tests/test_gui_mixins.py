#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_chat_store import ChatStore
from agent_gui_support import ChatUIV24Mixin


class Label:
    def __init__(self):
        self.text = ''
        self.tooltip = ''
    def set_text(self, text): self.text = text
    def set_tooltip_text(self, text): self.tooltip = text


class ChatStub(ChatUIV24Mixin):
    def __init__(self, directory):
        self.chat_store = ChatStore(directory)
        self.messages = [{"role": "system", "content": "SYSTEM"},
                         {"role": "user", "content": "Create a box"}]
        self._system_prompt_v24 = 'SYSTEM NEW'
        self.transcript_markdown = '## You\n\nCreate a box\n'
        self.backend = SimpleNamespace(session=None)
        self.mcp = object()
        self.model = 'qwen'
        self.args = SimpleNamespace(server='llama.cpp', url='http://127.0.0.1:8080/v1/chat/completions',
                                    api_key='not-for-disk', model='qwen', no_json_schema=False)
        self.current_chat_path = None
        self.busy = False
        self.connection_pending = False
        self.chat_title_label = Label()
        self.connection_status = Label()
        self._glib = SimpleNamespace(idle_add=lambda callback: callback())
        self._start_mcp_v24 = lambda _: object()
        self.workspace = SimpleNamespace(name='/tmp')
        self.visible_transcript = ''
        self.connect_button = SimpleNamespace(set_label=lambda *a: None, set_sensitive=lambda *a: None)
        self.sidebar_toggle = SimpleNamespace(set_sensitive=lambda *a: None)
        self.new_chat_button = SimpleNamespace(set_sensitive=lambda *a: None)
        self.refresh_chat_button = SimpleNamespace(set_sensitive=lambda *a: None)
        self.send_button = SimpleNamespace(set_sensitive=lambda *a: None)
        self.input_text_view = SimpleNamespace(set_sensitive=lambda *a: None)
    def _refresh_chats_v24(self): pass
    def render_markdown(self, text): self.visible_transcript = text
    def reset_transcript(self): self.transcript_markdown = 'NEW'; self.visible_transcript = 'NEW'
    def append_markdown(self, text): self.visible_transcript += text
    def set_busy(self, value): self.busy = value


class TestMixin(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.gui = ChatStub(Path(tmp.name) / 'save')

    def test_save_load_and_legacy_does_not_overwrite(self):
        self.gui._save_current_chat_v24()
        path = self.gui.current_chat_path
        self.assertTrue(path.exists())
        self.gui.messages = [{"role":"system", "content":"modified"}]
        self.gui._load_chat_v24(path)
        self.assertEqual(self.gui.messages[0]['content'], 'SYSTEM NEW')
        self.assertIn('Create a box', self.gui.visible_transcript)
        self.assertEqual(self.gui.current_chat_path, path)
        legacy = self.gui.chat_store.directory / 'old.json'
        legacy.write_text(json.dumps([{"role":"user", "content":"Old legacy chat"}]))
        self.gui._load_chat_v24(legacy)
        self.assertIsNone(self.gui.current_chat_path)
        self.assertEqual(json.loads(legacy.read_text())[0]['role'], 'user')

    def test_new_chat(self):
        self.gui._save_current_chat_v24()
        self.gui._new_chat_v24(None)
        self.assertEqual(len(self.gui.messages), 1)
        self.assertIsNone(self.gui.current_chat_path)
        self.assertEqual(self.gui.visible_transcript, 'NEW')

    def test_reconnect_failure_retains_previous(self):
        current = self.gui.backend
        with patch('llm_backends.make_backend', side_effect=RuntimeError('offline')):
            with patch('agent_gui_v24_support.threading.Thread') as thread:
                def start_now():
                    thread.call_args.kwargs['target']()
                thread.return_value.start.side_effect = start_now
                self.gui._connect_async_v24('openai-compatible',
                    'https://api.openai.com/v1/chat/completions', 'gpt-6.1-sol', 'sk-test')
        self.assertIs(self.gui.backend, current)
        self.assertEqual(self.gui.model, 'qwen')
        self.assertFalse(self.gui.connection_pending)
        self.assertIn('failed', self.gui.visible_transcript.lower())

    def test_connect_success_switches_without_restarting_mcp(self):
        old_mcp = self.gui.mcp
        new_backend = SimpleNamespace(check_connection=lambda: True)
        with patch('llm_backends.make_backend', return_value=new_backend):
            with patch('agent_gui_v24_support.threading.Thread') as thread:
                thread.return_value.start.side_effect = lambda: thread.call_args.kwargs['target']()
                self.gui._connect_async_v24('openai-compatible',
                    'https://api.openai.com/v1/chat/completions', 'gpt-6.1-sol', 'sk-test')
        self.assertIs(self.gui.backend, new_backend)
        self.assertIs(self.gui.mcp, old_mcp)
        self.assertEqual(self.gui.model, 'gpt-6.1-sol')
        self.assertEqual(self.gui.args.api_key, 'sk-test')
        self.assertEqual(self.gui.connection_status.text, 'Connected')
        self.gui._save_current_chat_v24()
        self.assertNotIn('sk-test', self.gui.current_chat_path.read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
