"""Loopback-only Streamable HTTP integration; no paid APIs."""
import os
import socket
import threading
import time
import unittest
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp_bridge import MCPBridge, MCPBridgeConfig


class HTTPIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sdk = MCPServer('http-test')
        @cls.sdk.tool()
        def echo(value: str) -> dict:
            return {'value': value}
        inner = cls.sdk.streamable_http_app()
        async def app(scope, receive, send):
            if scope['type'] == 'http':
                headers = dict(scope['headers'])
                if headers.get(b'authorization') != b'Bearer test-token':
                    await send({'type': 'http.response.start', 'status': 401,
                                'headers': []})
                    await send({'type': 'http.response.body', 'body': b'unauthorized'})
                    return
            await inner(scope, receive, send)
        cls.sock = socket.socket()
        cls.sock.bind(('127.0.0.1', 0))
        cls.url = f'http://127.0.0.1:{cls.sock.getsockname()[1]}/mcp'
        cls.server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
        cls.thread = threading.Thread(target=cls.server.run,
                                      kwargs={'sockets': [cls.sock]}, daemon=True)
        cls.thread.start()
        deadline = time.monotonic() + 10
        while not cls.server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        if not cls.server.started:
            raise RuntimeError('HTTP fixture did not start')

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(10)
        cls.sock.close()

    def bridge(self, headers):
        return MCPBridge(MCPBridgeConfig(transport='streamable_http',
            url=self.url, headers=headers, startup_timeout=5, tool_timeout=5))

    def test_headers_discovery_call_and_reconnect(self):
        bridge = self.bridge({'Authorization': 'Bearer test-token'})
        try:
            for _ in range(2):
                bridge.start()
                self.assertIn('echo', {t['name'] for t in bridge.list_tools()})
                result = bridge.call_tool('echo', {'value': 'hello'})
                self.assertFalse(result['isError'], result)
                data = result['structured']
                if data is None:
                    import json
                    data = json.loads(result['text'][0])
                self.assertEqual(data.get('result', data)['value'], 'hello')
                bridge.close()
        finally:
            bridge.close()

    def test_auth_failure_is_reported(self):
        bridge = self.bridge({})
        try:
            with self.assertRaises(RuntimeError):
                bridge.start()
        finally:
            bridge.close()
