"""V14: NHTSA status is not recall count; stop viewer is not fake completion."""
import ast
import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'agent/runtime'

def _method(file, name):
    t = ast.parse(file.read_text(encoding='utf-8'))
    fn = next(n for n in t.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    fn.decorator_list = []
    return compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(file), 'exec')


def _search(status, records):
    output = {}
    async def mock_fetch(*args):
        return records, status
    env = dict(ToolRuntime=object, Command=object, _vehicle_authorized=lambda *_: True,
               fetch_recalls=mock_fetch,
               _tool_response=lambda runtime, result, updates, name, summary: dict(result=result, updates=updates, summary=summary))
    exec(_method(SRC / 'tools.py', 'search_nhtsa_recalls'), env)
    return asyncio.run(env['search_nhtsa_recalls']('Honda','Accord',2018,SimpleNamespace(state={})))


def test_unavailable_lookup_has_unknown_count_and_no_rank_suggestion():
    event = _search('unavailable', [])
    assert event['result']['model_year_records'] is None
    assert event['updates']['recall_status'] == 'unavailable'
    assert 'Do not say no recalls' in event['result']['next_step']
    assert 'do not call retrieval' in event['result']['next_step']


def test_successful_empty_query_is_zero():
    event = _search('ok', [])
    assert event['result']['model_year_records'] == 0
    assert 'zero campaigns' in event['result']['next_step']


def test_successful_campaign_count_is_correct():
    event = _search('ok', [{'id': '20V771000'}]*6)
    assert event['result']['model_year_records'] == 6
    assert 'Call retrieve_recall_evidence' in event['result']['next_step']


def test_unavailable_answer_guard_and_status_prompt_present():
    src = (SRC / 'grounding.py').read_text()
    assert "recall_status == 'unavailable'" in src
    assert "_UNAVAILABLE_SAFE_ANSWER" in src
    prompt = (SRC / 'constants.py').read_text()
    assert "LOOKUP FAILED" in prompt and 'If the user writes "continue"' in prompt
    lifecycle = (SRC / 'lifecycle.py').read_text()
    assert "if state.get('recall_status') == 'ok' else None" in lifecycle


def test_viewer_does_not_claim_tool_waiting_after_result():
    viewer = (ROOT / 'frontend/src/components/ModelStreamViewer.tsx').read_text()
    assert 'completedTools.has(item.runId)' in viewer
    assert 'Result received' in viewer
    assert 'possibly interrupted' in viewer
    assert 'Stream ended' in viewer
