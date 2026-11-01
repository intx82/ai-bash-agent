#!/usr/bin/env python3
"""Offline regression tests: no FreeCAD GUI, SDK or vision model required."""
import base64
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

# Stub MCP SDK for agent + manager imports.
mcp=types.ModuleType('mcp'); client=types.ModuleType('mcp.client')
stdio=types.ModuleType('mcp.client.stdio'); mt=types.ModuleType('mcp.types')
class TextContent:
    def __init__(self,text):self.text=text
class ImageContent:
    def __init__(self,data,mimeType):self.data=data;self.mimeType=mimeType
mcp.ClientSession=lambda *a,**kw:None
mcp.StdioServerParameters=lambda **kw:kw
stdio.stdio_client=lambda a:None
mt.TextContent=TextContent
mt.ImageContent=ImageContent
mt.PaginatedRequestParams=lambda **kw:kw
mcp.types=mt
sys.modules.update({'mcp':mcp,'mcp.client':client,'mcp.client.stdio':stdio,'mcp.types':mt})
sys.path.insert(0,'/mnt/data')
import agent, agent_vision, freecad_verify as f, llm_backends

properties={'Length':50, 'Width':50, 'Height':10}
geometry={'is_null':False,'is_valid':True,'volume_mm3':25000.,'solids':1,'bbox_mm':{'x':50,'y':50,'z':10}}
class FakeMCP:
    def __init__(self, props=None, geometry_=None, image=None):
        self.props=properties if props is None else props
        self.geometry=geometry if geometry_ is None else geometry_
        self.image=image; self.calls=[]
    def prompt_catalog(self):return 'freecad.create_object freecad.get_object freecad.execute_code freecad.get_view'
    def has_tool(self,name):return name in ('freecad.create_object','freecad.get_object','freecad.execute_code','freecad.get_view')
    def needs_confirmation(self,name):return name=='freecad.create_object'
    def call_tool(self,name,arguments):
        self.calls.append((name,arguments))
        if name=='freecad.create_object':
            return {'structured':{'result':{'success':True,'object_name':'Box'}}, 'text':[], 'images':[], 'isError':False}
        if name=='freecad.get_object':
            return {'structured':{'result':{'success':True,'object':{'properties':self.props}}}, 'text':[],'images':[],'isError':False}
        if name=='freecad.execute_code':
            assert 'include_screenshot' in arguments and arguments['include_screenshot'] is False
            # Ensure probe is static / read-only and doc/obj names are quoted.
            assert 'App.getDocument(' in arguments['code']
            probe={'exists':True,'properties':self.props,'shape':self.geometry,'type_id':'Part::Box'}
            b=base64.b64encode(json.dumps(probe).encode()).decode()
            return {'structured':{'result':{'output':f._MARKER+b}},'text':[],'isError':False}
        if name=='freecad.get_view':
            return {'structured':{'result':{'success':True}}, 'text':[], 'images':[self.image],'isError':False}
        raise AssertionError(name)

args={'doc_name':'Box','obj_name':'Box','obj_type':'Part::Box','obj_properties':properties}
success={'structured':{'result':{'success':True}},'text':[],'isError':False}
mcp=FakeMCP()
rep=f.verify_mutation(mcp,'freecad.create_object',args,success)
assert rep['status']=='verified',rep
assert rep['expected_volume_mm3']==25000
assert [x[0] for x in mcp.calls]==['freecad.get_object','freecad.execute_code']
assert rep['geometry']['is_valid'] is True
mismatch=f.verify_mutation(FakeMCP(props={'Length':40,'Width':50,'Height':10}), 'freecad.create_object',args,success)
assert mismatch['status']=='mismatch',mismatch
invalid=f.verify_mutation(FakeMCP(geometry_={**geometry,'is_valid':False}),'freecad.create_object',args,success)
assert invalid['status']=='failed',invalid
unver=f.verify_mutation(FakeMCP(geometry_={}), 'freecad.create_object',args,success)
assert unver['status']=='unverified',unver
assert f.verify_mutation(FakeMCP(),'freecad.create_object',args,{'isError':True})['status']=='not_run'

class Model:
    def __init__(self,tool='freecad.create_object'):
        self.n=0;self.states=[];self.tool=tool
    def chat_completion(self,messages,max_tokens,temperature):
        self.n+=1
        self.states.append(json.loads(json.dumps(messages)))
        if self.n==1:
            return {'content':json.dumps({'type':'tool','plan':[], 'tool':'mcp','name':self.tool,
               'arguments': args if self.tool=='freecad.create_object' else {'view_name':'Isometric'}}), 'reasoning':''}
        return {'content':'{"type":"message","plan":[],"message":"Done"}','reasoning':''}

messages=[{'role':'system','content':agent.SYSTEM_PROMPT}]
model=Model()
agent.run_agent_turn(model,FakeMCP(),messages,'Create a box',2048,0.3,False,'/tmp',confirm_func=lambda _:True)
assert model.n==2
assert '"verification": {"status": "verified"' in messages[-2]['content'],messages[-2]

# Actual pixel delivery only with explicit opt-in. No persisted base64.
with tempfile.TemporaryDirectory() as tempdir:
    p=Path(tempdir)/'view.png'
    p.write_bytes(b'\x89PNG\r\n\x1a\n' + b'fake-png-data')
    image={'path':str(p),'mime_type':'image/png','bytes':p.stat().st_size}
    with patch.dict(os.environ,{'AGENT_VISION_MODE':'off'}):
        status,payload=agent_vision.image_feedback({'images':[image]},'TOOL_RESULT {}')
        assert status['status']=='not_analyzed' and payload is None
    with patch.dict(os.environ,{'AGENT_VISION_MODE':'main'}):
        status,payload=agent_vision.image_feedback({'images':[image]},'TOOL_RESULT {}')
        assert status['status']=='sent_to_main_model'
        assert payload[1]['image_url']['url'].startswith('data:image/png;base64,')
        model=Model('freecad.get_view')
        messages=[{'role':'system','content':agent.SYSTEM_PROMPT}]
        agent.run_agent_turn(model, FakeMCP(image=image), messages,'View it',2048,0.3,True,tempdir)
        assert isinstance(model.states[1][-1]['content'],list)
        assert model.states[1][-1]['content'][1]['type']=='image_url'
        assert isinstance(messages[-2]['content'],str)
        assert 'data:image/png;base64,' not in json.dumps(messages)
    class FakeResp:
        def raise_for_status(self):pass
        def json(self):return {'choices':[{'message':{'content':'A flat rectangular box is visible.'}}]}
    with patch.dict(os.environ,{'AGENT_VISION_MODE':'sidecar','AGENT_VISION_URL':'http://localhost:8000/v1/chat/completions','AGENT_VISION_MODEL':'vision-test'}):
        with patch('requests.post', return_value=FakeResp()) as request:
            status,payload=agent_vision.image_feedback({'images':[image]},'TOOL_RESULT {}')
            assert status['status']=='analyzed_by_sidecar' and payload is None
            assert 'flat rectangular' in status['visual_description']
            assert request.call_args.kwargs['json']['messages'][0]['content'][1]['type']=='image_url'

# Llama backend forwards content arrays without coerced string conversion.
class Resp:
    status_code=200
    def json(self):return {'choices':[{'message':{'content':'{"type":"message","plan":[],"message":"ok"}'}}]}
class Session:
    headers={}
    def post(self,url,json,timeout):
        assert isinstance(json['messages'][-1]['content'],list)
        assert json['messages'][-1]['content'][1]['type']=='image_url'
        return Resp()
    def get(self,*a,**kw):return Resp()
backend=llm_backends.LlamaCppBackend(llm_backends.BackendConfig(server='llama.cpp',model='vision',url='http://localhost:8080/v1/chat/completions'))
backend.session=Session()
assert backend.chat_completion([{'role':'user','content':[{'type':'text','text':'view'}, {'type':'image_url','image_url':{'url':'data:image/png;base64,aA=='}}]}],100,0.2)['content']
print('PASS geometry verification, numeric mismatch, shape failure, unverified, off/main/sidecar vision and ephemeral image, backend forwarding')

# Agent must not acknowledge a mismatched creation as a successful tool result.
model=Model()
messages=[{'role':'system','content':agent.SYSTEM_PROMPT}]
agent.run_agent_turn(model,FakeMCP(props={'Length':45,'Width':50,'Height':10}),messages,'Create 50x50x10',2048,0.2,False,'/tmp',confirm_func=lambda _:True)
assert '"status": "mismatch"' in messages[-2]['content']
assert '"ok": false' in messages[-2]['content']
assert f.response_failed({'text':['{"success": false, "error": "failed"}'], 'isError':False})
print('PASS mismatched geometry blocks successful tool completion and FreeCAD success:false is detected')
