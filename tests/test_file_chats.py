#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_chat_store import ChatStore
from agent_gui_support import ChatUIV24Mixin
#from apply_gui_v24 import patch_gui, VERSION, PREVIOUS_VERSION


class Label:
    def __init__(self):
        self.text = ''
        self.tooltip = ''
    def set_text(self, text): self.text = text
    def set_tooltip_text(self, text): self.tooltip = text


class ChatStub(ChatUIV24Mixin):
    def __init__(self, directory):
        self.chat_store = ChatStore(directory)
        self.messages = [{"role": "system", "content": "SYSTEM"}]
        self._system_prompt_v24 = "SYSTEM NEW"
        self.transcript_markdown = ""
        self.model = 'qwen'
        self.args = SimpleNamespace(server='llama.cpp', url='http://localhost:8080/v1/chat/completions', api_key='sk-avoid-persisting')
        self.current_chat_path = None
        self.current_chat_signature = None
        self._saved_chat_fingerprint = None
        self.busy = False
        self.connection_pending = False
        self.chat_title_label = Label()
        self.connection_status = Label()
        self._gtk = None
        self.visible_transcript = ''
        self._next_path = None
        self.errors = []
    def _refresh_chats_v24(self): pass
    def _confirm_leave_chat_v24(self): return True
    def _ask_new_chat_filename_v24(self): return self._next_path
    def render_markdown(self, text): self.visible_transcript = text
    def reset_transcript(self): self.transcript_markdown = 'NEW'; self.visible_transcript = 'NEW'
    def append_markdown(self, text): self.errors.append(text)


class FileChatTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'save'
        self.root.mkdir()
        self.store = ChatStore(self.root)
        self.legacy = [
            {'role': 'system', 'content': 'Old instructions'},
            {'role': 'user', 'content': 'Repair aaa.c'},
            {'role': 'assistant', 'content': json.dumps({'type':'tool','tool':'mcp','name':'read_text','arguments':{'path':'aaa.c'}})},
            {'role': 'user', 'content': 'TOOL_RESULT {"ok": true}'},
            {'role': 'assistant', 'content': json.dumps({'type':'message','message':'Fixed.'})},
        ]

    def test_sidebar_lists_exact_filenames_not_generated_titles(self):
        paths = ['3d-device-box.json', 'aaa.json', 'libmanager.rebase.json', 'libstt.alsa.output.json']
        for name in paths:
            (self.root / name).write_text(json.dumps(self.legacy), encoding='utf-8')
        (self.root / 'broken.json').write_text('{not valid JSON')
        (self.root / 'backup.json.bak').write_text('[]')
        result = self.store.list_chats()
        self.assertEqual(sorted(entry.title for entry in result), sorted(paths + ['broken.json']))
        self.assertTrue(all(entry.title == entry.path.name for entry in result))
        self.assertEqual(len(list(self.root.glob('chat-*.json'))), 0)

    def test_legacy_array_retained_and_original_backed_up(self):
        path = self.root / 'libstt.filesource.json'
        before = json.dumps(self.legacy, ensure_ascii=False)
        path.write_text(before, encoding='utf-8')
        loaded = self.store.load(path)
        self.assertEqual(loaded['title'], 'libstt.filesource.json')
        self.assertEqual(loaded['source_format'], 'legacy-list')
        messages = loaded['messages'] + [{'role':'user','content':'Continue'}]
        self.store.save(path, messages, expected_signature=loaded['signature'])
        self.assertIsInstance(json.loads(path.read_text()), list)
        self.assertEqual(json.loads(path.read_text()), messages)
        self.assertEqual((self.root / 'libstt.filesource.json.bak').read_text(), before)
        self.store.save(path, messages, expected_signature=self.store.signature(path))
        self.assertEqual((self.root / 'libstt.filesource.json.bak').read_text(), before)

    def test_load_edit_autosave_same_legacy_filename(self):
        path = self.root / 'libmanager.lint.json'
        path.write_text(json.dumps(self.legacy), encoding='utf-8')
        gui = ChatStub(self.root)
        gui._load_chat_v24(path)
        self.assertEqual(gui.current_chat_path, path)
        self.assertEqual(gui.chat_title_label.text, 'libmanager.lint.json')
        self.assertEqual(len(gui.messages), len(self.legacy))
        gui.messages.append({'role':'user', 'content':'Run tests'})
        gui._save_chat_after_turn_v24()
        self.assertEqual(json.loads(path.read_text())[-1]['content'], 'Run tests')
        self.assertFalse(list(self.root.glob('chat-*.json')))
        self.assertFalse(gui.errors)

    def test_loading_unmodified_does_not_rewrite_or_back_up(self):
        path = self.root / 'ssh.json'
        path.write_text(json.dumps(self.legacy), encoding='utf-8')
        gui = ChatStub(self.root)
        gui._load_chat_v24(path)
        stamp = self.store.signature(path)
        gui._save_current_chat_v24(silent=True)
        self.assertEqual(self.store.signature(path), stamp)
        self.assertFalse((self.root / 'ssh.json.bak').exists())

    def test_naming_new_chat_no_random_name(self):
        gui = ChatStub(self.root)
        gui._next_path = self.store.named_path('device-case')
        gui._new_chat_v24(None)
        self.assertEqual(gui.current_chat_path.name, 'device-case.json')
        self.assertFalse(gui.current_chat_path.exists())
        gui.messages.append({'role':'user','content':'Make 55mm box'})
        gui._save_chat_after_turn_v24()
        self.assertIsInstance(json.loads((self.root / 'device-case.json').read_text()), list)
        self.assertEqual(gui.chat_title_label.text, 'device-case.json')
        self.assertFalse(list(self.root.glob('chat-*.json')))

    def test_external_edit_prevents_overwrite(self):
        path = self.root / 'radius.6.0.1.53.json'
        path.write_text(json.dumps(self.legacy))
        gui = ChatStub(self.root)
        gui._load_chat_v24(path)
        path.write_text(json.dumps(self.legacy + [{'role':'user','content':'External edit'}]))
        gui.messages.append({'role':'user','content':'Our edit'})
        gui._save_current_chat_v24()
        self.assertIn('External edit', path.read_text())
        self.assertNotIn('Our edit', path.read_text())
        self.assertTrue(gui.errors)

    def test_existing_envelopes_preserved(self):
        for fmt in ('agent-context', 'agent-gui-chat'):
            with self.subTest(fmt=fmt):
                path = self.root / (fmt + '.json')
                state = {'format':fmt,'version':2,'messages':self.legacy,
                         'llama_cache':{'slot_id':0},'transcript':'prior'}
                path.write_text(json.dumps(state))
                loaded = self.store.load(path)
                self.store.save(path, self.legacy + [{'role':'user','content':'Followup'}],
                                'new transcript', expected_signature=loaded['signature'])
                updated = json.loads(path.read_text())
                self.assertEqual(updated['format'],fmt)
                self.assertEqual(updated['llama_cache'],{'slot_id':0})
                self.assertEqual(updated['messages'][-1]['content'],'Followup')

    def test_unrelated_json_refused(self):
        path = self.root/'unrelated.json'
        path.write_text('{"random":true}')
        with self.assertRaises(ValueError):
            self.store.load(path)
        with self.assertRaises(ValueError):
            self.store.save(path, self.legacy)

    def test_invalid_names_not_accepted(self):
        for invalid in ('../other.json', '/tmp/other.json', '.hidden.json', '..', 'a/b.json'):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.store.named_path(invalid)


    # def test_previous_v24_gui_patch_upgrade(self):
    #     source = ('GTKUI_VERSION = "' + PREVIOUS_VERSION + '"\n'
    #               'def mock():\n'
    #               '    if cmd == "save":\n'
    #               '                GLib.idle_add(self._save_chat_after_turn_v24)\n'
    #               '                return "## Chat save requested"\n')
    #     out = patch_gui(source)
    #     self.assertIn(VERSION,out)
    #     self.assertIn('GLib.idle_add(self._save_command_v24)',out)
    #     self.assertEqual(out,patch_gui(out))


if __name__ == '__main__':
    unittest.main(verbosity=2)
