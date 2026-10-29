"""Offline tests; do not access OpenRouter or reveal credentials."""
import asyncio
import json
from pathlib import Path

import pytest

from agent.agentic_llm import AgentLLMError, _reasoning_text, stream_chat_events

ROOT = Path(__file__).resolve().parents[2]


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines
    def raise_for_status(self):
        pass
    async def aiter_lines(self):
        for line in self.lines:
            yield line


class FakeClient:
    lines = []
    captured = {}
    def __init__(self, **kwargs):
        pass
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass
    def stream(self, *args, **kwargs):
        FakeClient.captured = kwargs
        class Context:
            async def __aenter__(self):
                return FakeResponse(FakeClient.lines)
            async def __aexit__(self, *args):
                pass
        return Context()


@pytest.fixture(autouse=True)
def fake_http(monkeypatch):
    import agent.agentic_llm as m
    monkeypatch.setattr(m.settings, 'openrouter_model', 'offline-test-model')
    monkeypatch.setattr(m.settings, 'openrouter_api_key', 'fake-test-key')
    monkeypatch.setattr(m.httpx, 'AsyncClient', FakeClient)


def collect(include_reasoning=True):
    async def run():
        return [ev async for ev in stream_chat_events([{'role': 'user', 'content': 'test'}],
                                                      include_reasoning=include_reasoning)]
    return asyncio.run(run())


def event(delta):
    return 'data: ' + json.dumps({'choices': [{'delta': delta}]})


def test_separates_reasoning_from_final_answer():
    FakeClient.lines = [
        ': keepalive', event({'reasoning_details': [{'type': 'reasoning.text', 'text': 'Provider text'}],
                               'reasoning': 'Provider text'}),
        event({'reasoning_details': [{'type': 'reasoning.summary', 'summary': 'Explanation'}]}),
        event({'content': 'Final answer'}), 'data: [DONE]',
    ]
    assert collect() == [
        {'kind': 'reasoning', 'text': 'Provider text'},
        {'kind': 'reasoning', 'text': 'Explanation'},
        {'kind': 'answer', 'text': 'Final answer'},
    ]
    assert FakeClient.captured['json']['reasoning']['enabled'] is True
    assert FakeClient.captured['json']['stream'] is True


def test_default_ignores_reasoning_and_has_no_reasoning_request():
    FakeClient.lines = [event({'reasoning': 'not sent'}), event({'content': 'OK'}), 'data: [DONE]']
    assert collect(False) == [{'kind': 'answer', 'text': 'OK'}]
    assert 'reasoning' not in FakeClient.captured['json']


def test_encrypted_redacted_never_rendered():
    assert _reasoning_text({'reasoning_details': [{'type': 'reasoning.encrypted', 'data': 'secret'}]}) == ''
    assert _reasoning_text({'reasoning': '[REDACTED]'}) == ''
    assert _reasoning_text({'reasoning': 'reasoning text'}) == 'reasoning text'


def test_only_reasoning_without_answer_is_error():
    FakeClient.lines = [event({'reasoning': 'some text'}), 'data: [DONE]']
    with pytest.raises(AgentLLMError, match='no visible answer'):
        collect()


def test_ui_is_opt_in_and_separate_from_answer():
    source = (ROOT / 'frontend/src/components/StoryWorkspace.tsx').read_text()
    agent = (ROOT / 'agent/story_agent.py').read_text()
    assert 'onRawEvent:' in source and 'ChatOpenRouter(**model_kwargs)' in agent
    assert 'providerReasoning' in source
    assert 'AUTOASSIST_PROVIDER_REASONING' in agent
    assert 'Observed AG-UI node events' not in source
    assert 'Actual LangGraph node events, not hidden model reasoning.' not in source
    assert 'create_agent' in (ROOT / 'scripts/probe_create_agent_v8.py').read_text()
