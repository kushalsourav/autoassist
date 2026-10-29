"""Offline parser regression tests; no OpenRouter calls or credentials."""
import asyncio
import json

import pytest

from agent.agentic_llm import AgentLLMError, stream_chat


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class FakeStream:
    def __init__(self, lines):
        self.response = FakeResponse(lines)

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *args):
        pass


class FakeClient:
    lines = []
    captured = {}

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def stream(self, method, url, **kwargs):
        self.captured.update(kwargs)
        return FakeStream(self.lines)


@pytest.fixture(autouse=True)
def setup_transport(monkeypatch):
    import agent.agentic_llm as module
    monkeypatch.setattr(module.settings, 'openrouter_api_key', 'example-test-only')
    monkeypatch.setattr(module.settings, 'openrouter_model', 'test-model')
    monkeypatch.setattr(module.httpx, 'AsyncClient', FakeClient)


def run_stream():
    async def collect():
        return [part async for part in stream_chat([{'role': 'user', 'content': 'hello'}])]
    return asyncio.run(collect())


def data(v):
    return 'data: ' + json.dumps(v)


def test_stream_ignores_reasoning_and_comments():
    FakeClient.lines = [': OPENROUTER PROCESSING', '',
                        data({'choices': [{'delta': {'reasoning': 'HIDDEN INTERNAL'}}]}),
                        data({'choices': [{'delta': {'content': 'Hello '}}]}),
                        data({'choices': [{'delta': {'content': 'world'}}]}),
                        'data: [DONE]']
    assert run_stream() == ['Hello ', 'world']
    assert FakeClient.captured['json']['stream'] is True


def test_stream_midstream_failure_is_not_silent():
    FakeClient.lines = [data({'choices': [{'delta': {'content': 'Partial'}}]}),
                        data({'error': {'code': 'upstream_error'}})]
    with pytest.raises(AgentLLMError, match='stopped'):
        run_stream()


def test_stream_rejects_missing_done_termination():
    FakeClient.lines = [data({'choices': [{'delta': {'content': 'Part'}}]})]
    with pytest.raises(AgentLLMError, match='ended unexpectedly'):
        run_stream()


def test_stream_rejects_empty_text():
    FakeClient.lines = [data({'choices': [{'delta': {'reasoning': 'secret'}}]}), 'data: [DONE]']
    with pytest.raises(AgentLLMError, match='no visible answer'):
        run_stream()
