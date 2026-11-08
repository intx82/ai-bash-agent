#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from apply_gui_v24 import patch_gui, install, VERSION, PREVIOUS_VERSION
from llm_backends import BackendConfig, OpenAICompatibleBackend, make_backend
from _synthetic_source import SYNTHETIC_GUI


class ConnectTests(unittest.TestCase):
    def test_openai_compatible_url_and_model(self):
        cfg = BackendConfig(server='openai-compatible', model='gpt-6.1-sol',
                            url='https://api.openai.com/v1/chat/completions',
                            api_key='fake-test-secret')
        backend = make_backend(cfg)
        self.assertIsInstance(backend, OpenAICompatibleBackend)
        self.assertEqual(backend.session.headers['Authorization'], 'Bearer fake-test-secret')
        response = MagicMock(ok=True)
        response.json.return_value = {'choices':[{'message':{'content':'{"type":"message"}'}}]}
        with patch.object(backend.session,'post',return_value=response) as mock:
            self.assertIsNotNone(backend.chat_completion([],256,0.4))
        request_data = mock.call_args.kwargs['json']
        self.assertNotIn('id_slot',request_data)
        self.assertNotIn('cache_prompt',request_data)


class InstallTests(unittest.TestCase):
    def test_original_gui_patch(self):
        patched = patch_gui(SYNTHETIC_GUI)
        self.assertIn(VERSION, patched)
        self.assertIn('ChatUIV24Mixin', patched)
        self.assertIn('self._install_chat_layout', patched)
        self.assertIn('self._save_command_v24', patched)
        self.assertEqual(patched, patch_gui(patched))
        compile(patched, 'agent_gui.py', 'exec')

    def test_upgrade_from_v24_and_backup(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root)/'agent_gui.py'
            old_v24 = patch_gui(SYNTHETIC_GUI).replace(VERSION,PREVIOUS_VERSION)
            # Original v24 used the previous save command.
            old_v24 = old_v24.replace('GLib.idle_add(self._save_command_v24)',
                                      'GLib.idle_add(self._save_chat_after_turn_v24)')
            source.write_text(old_v24,encoding='utf-8')
            install(source)
            self.assertIn(VERSION,source.read_text())
            self.assertIn('self._save_command_v24',source.read_text())
            self.assertEqual((Path(root)/'agent_gui.py.pre-v24_1.bak').read_text(),old_v24)

    def test_refuse_unrecognized_revision_without_change(self):
        with self.assertRaises(RuntimeError):
            patch_gui('def main(): pass\n')


if __name__ == '__main__':
    unittest.main(verbosity=2)
