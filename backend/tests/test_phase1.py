"""Phase 1 regression tests: routing, provider errors, and chat history."""
import asyncio
import json
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from agent.intent import route_intent
from agent import generation, workflow
from backend.app import main
from backend.app.storage import SQLiteStore


VEHICLE = {'make': 'Honda', 'model': 'Accord', 'year': 2018}
CLASSIFICATION = {'category': 'Engine / Powertrain', 'method': 'keyword_demo_only'}


def test_intent_gating():
    assert route_intent('tell about the tigers') == 'out_of_scope'
    assert route_intent('what about the cars how many are present') == 'clarification'
    assert route_intent('My vehicle engine stalls while accelerating') == 'automotive'
    assert route_intent('Is it safe to drive?', [
        {'role': 'user', 'content': 'My car loses power on acceleration'}]) == 'automotive'


def test_non_automotive_does_not_call_recalls(monkeypatch):
    async def forbid_calls(**_kwargs):
        raise AssertionError('NHTSA must not be queried for unrelated questions')
    monkeypatch.setattr(workflow, 'fetch_recalls', forbid_calls)
    state = asyncio.run(workflow.SequentialFallback().ainvoke({
        'message': 'tell about the tigers', 'vehicle': VEHICLE}))
    assert state['generation_mode'] == 'intent_guardrail'
    assert state['recall_status'] == 'not_queried'
    assert state['documents'] == []
    assert 'automotive' in state['answer'].lower() or 'vehicle' in state['answer'].lower()


def mock_openrouter(monkeypatch, responder):
    real_client = httpx.AsyncClient
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        return responder(request)
    transport = httpx.MockTransport(handler)
    def fake_client(**kwargs):
        return real_client(transport=transport, **kwargs)
    monkeypatch.setattr(generation.httpx, 'AsyncClient', fake_client)
    monkeypatch.setattr(generation.settings, 'openrouter_api_key', 'fake-key')
    monkeypatch.setattr(generation.settings, 'openrouter_model', 'test/mock-model')
    return captured


def test_real_llm_response_and_history_used(monkeypatch):
    captured = mock_openrouter(monkeypatch, lambda request: httpx.Response(200, json={
        'choices': [{'message': {'content': 'This answer came from a mocked LLM.'}}]}))
    result = asyncio.run(generation.generate('What else should I check?', VEHICLE,
        CLASSIFICATION, [], 'ok', [
            {'role': 'user', 'content': 'My engine loses power.'},
            {'role': 'assistant', 'content': 'Please tell me about warning lights.'}]))
    assert result == ('This answer came from a mocked LLM.', 'llm', None)
    assert len(captured) == 1
    assert captured[0]['messages'][-2]['role'] == 'assistant'
    assert captured[0]['messages'][-1]['content'] == 'What else should I check?'


def test_provider_rate_limit_error_is_exposed_safely(monkeypatch):
    mock_openrouter(monkeypatch, lambda request: httpx.Response(429, json={
        'error': {'message': 'key-secret must not be echoed'}}))
    text, mode, error = asyncio.run(generation.generate('My car engine stalls', VEHICLE,
        CLASSIFICATION, [], 'ok'))
    assert mode == 'llm_unavailable'
    assert 'Non-LLM fallback' in text
    assert '429' in error
    assert 'key-secret' not in error


def test_history_is_passed_into_analysis_and_saved(monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'local_store', SQLiteStore(tmp_path / 'phase1.db'))
    assert main.settings.storage_mode == 'local'
    calls = []
    class FakeGraph:
        async def ainvoke(self, state):
            calls.append(state)
            return {'intent': 'automotive',
                    'classification': CLASSIFICATION | {'confidence': None},
                    'documents': [], 'answer': 'Mocked response', 'recall_status': 'ok',
                    'generation_mode': 'llm', 'generation_error': None}
    monkeypatch.setattr(main, 'analysis_graph', FakeGraph())
    client = TestClient(main.app)
    headers = {'X-Demo-User': str(uuid4())}
    payload = {'message': 'My car engine loses power', **VEHICLE}
    first = client.post('/api/analyze', json=payload, headers=headers)
    assert first.status_code == 200, first.text
    payload['conversation_id'] = first.json()['conversation_id']
    payload['message'] = 'Is it safe to drive?'
    second = client.post('/api/analyze', json=payload, headers=headers)
    assert second.status_code == 200, second.text
    assert calls[0]['history'] == []
    assert [row['role'] for row in calls[1]['history']] == ['user', 'assistant']
    assert second.json()['generation_error'] is None
