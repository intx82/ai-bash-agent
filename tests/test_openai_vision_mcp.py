#!/usr/bin/env python3
"""Offline tests for OpenAI vision MCP server; no network/API key needed."""
import asyncio
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

from PIL import Image

# Load a private server module with a scoped SDK stub, without poisoning
# sys.modules for other MCP integration tests in the same test process.
import importlib.util

class StubMCPServer:
    def __init__(self, name):
        self.name = name
        self.registry = {}
    def tool(self):
        def dec(fn):
            self.registry[fn.__name__] = fn
            return fn
        return dec
    def run(self): pass

server_stub = types.ModuleType('mcp.server.mcpserver')
server_stub.MCPServer = StubMCPServer
spec = importlib.util.spec_from_file_location(
    'offline_vision_server', Path(__file__).resolve().parents[1] / 'mcp_openai_vision.py')
vision = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'mcp.server.mcpserver': server_stub}):
    spec.loader.exec_module(vision)

with tempfile.TemporaryDirectory() as temp:
    root=Path(temp)
    p=root/'mcp_images'/'test.png'
    p.parent.mkdir()
    Image.new('RGB',(200,100),(225,225,225)).save(p)
    os.environ['MCP_VISION_ROOT']=str(root)
    os.environ.pop('MCP_VISION_EXTRA_ROOTS',None)
    info=vision.image_info(str(p))
    assert info['width_px']==200 and info['height_px']==100, info
    assert info['mime_type']=='image/png'
    assert vision.image_info('/etc/passwd')['ok'] is False
    assert vision.image_info('../etc/passwd')['ok'] is False
    assert set(vision.mcp.registry)=={'describe_image','image_info','compare_regions'}

    def fake_response(img, question, detail, max_objects):
        assert img['_data_uri'].startswith('data:image/png;base64,')
        return {'description':'Two boxes', 'objects':[
            {'name':'left','description':'large box','bbox_1000':[100,200,500,800],'confidence':'medium'},
            {'name':'right','description':'small box','bbox_1000':[600,200,800,600],'confidence':'low'}],
            'observations':['right of left'], 'limitations':['perspective']}
    with patch.object(vision,'_openai_describe',side_effect=fake_response):
        result=asyncio.run(vision.describe_image(str(p)))
    assert result['ok'] and result['objects'][0]['bbox_px']==[20,20,100,80],result
    assert result['objects'][0]['width_px']==80
    assert result['objects'][1]['width_px']==40
    assert result['relative_sizes'][0]['width_ratio_a_to_b']==2
    assert '_data_uri' not in result
    assert 'model_estimate' == result['objects'][0]['bbox_source']
    assert 'model estimates' in result['measurement_note']

    regions=[{'name':'a','bbox_px':[0,0,100,40]}, {'name':'b','bbox_px':[100,40,150,60]}]
    compared=vision.compare_regions(str(p),regions)
    assert compared['relative_sizes'][0]['width_ratio_a_to_b']==2
    assert compared['relative_sizes'][0]['bbox_area_ratio_a_to_b']==4
    assert not vision.compare_regions(str(p),[{'name':'broken','bbox_px':[1,1,400,600]}])['ok']
    try:
        vision.describe_image_impl(str(p),detail='invalid')
    except ValueError: pass
    else: raise AssertionError('bad detail accepted')

    # Mock OpenAI SDK client, verify that image goes through Responses API
    # with image data, schema and no response storage.
    class MockResponses:
        def create(self, **kwargs):
            assert kwargs['store'] is False
            assert kwargs['text']['format']['type']=='json_schema'
            assert kwargs['input'][0]['content'][1]['type']=='input_image'
            assert kwargs['input'][0]['content'][1]['detail']=='high'
            assert len(kwargs['input'][0]['content'][1]['image_url'])>150
            return types.SimpleNamespace(output_text=json.dumps(fake_response({'_data_uri':'data:image/png;base64,AA'},'', 'high',8)))
    class MockOpenAI:
        def __init__(self, **kwargs):
            assert kwargs['api_key']=='TESTONLY'
            self.responses=MockResponses()
    stub=types.SimpleNamespace(OpenAI=MockOpenAI)
    with patch.dict(sys.modules,{'openai':stub}),patch.dict(os.environ,{'OPENAI_API_KEY':'TESTONLY'}):
        result=vision.describe_image_impl(str(p))
    assert result['ok'] and result['relative_sizes'][0]['width_ratio_a_to_b']==2
    # Verify server returns a safe failure on missing key
    with patch.dict(os.environ, {'OPENAI_API_KEY':''}):
        fail=asyncio.run(vision.describe_image(str(p)))
    assert fail['ok'] is False and 'OPENAI_API_KEY' in fail['error']

print('OpenAI vision MCP offline tests: PASS')
