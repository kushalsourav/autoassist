"""Regression tests: previously blocked valid vehicle symptoms must reach ML/RAG."""
import asyncio
from agent.intent import route_intent
from agent import workflow

VEHICLE = {'make': 'Honda', 'model': 'Accord', 'year': 2018}
WIPER_QUERY = ('My windshield wipers intermittently stop working and the '
               'rearview camera sometimes shows a black screen.')


def test_user_reported_regression_is_automotive():
    assert route_intent(WIPER_QUERY, vehicle=VEHICLE) == 'automotive'
    assert route_intent(WIPER_QUERY) == 'automotive'


def test_detailed_unlisted_symptom_not_blocked():
    # The object and wording are not part of the automotive keyword list.
    text = 'The little latch keeps rattling loudly every time I go over bumps.'
    assert route_intent(text, vehicle=VEHICLE) == 'automotive'
    assert route_intent(text) == 'clarification'


def test_clear_unrelated_remains_blocked_even_with_vehicle():
    assert route_intent('tell me about tigers', vehicle=VEHICLE) == 'out_of_scope'
    assert route_intent('Please explain JavaScript programming', vehicle=VEHICLE) == 'out_of_scope'


def test_mixed_and_vague_requests_are_not_routed_to_recalls():
    assert route_intent('What about the cars, how many are present?', vehicle=VEHICLE) == 'clarification'
    assert route_intent('Is the weather good for my car today?', vehicle=VEHICLE) == 'clarification'
    assert route_intent('Hi', vehicle=VEHICLE) == 'clarification'


def test_followup_remains_automotive():
    assert route_intent('Is it safe to drive?', [
        {'role': 'user', 'content': 'My car loses power on acceleration'}],
        vehicle=VEHICLE) == 'automotive'


def test_workflow_reaches_classifier_and_recall_for_wiper_query(monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, 'predict_category', lambda text: (
        calls.append('classifier') or {'category': 'Electrical',
        'method': 'test', 'labels': [{'category': 'Electrical', 'score': 0.9}]}))
    async def fake_recalls(**kwargs):
        calls.append('nhtsa')
        return [], 'ok'
    monkeypatch.setattr(workflow, 'fetch_recalls', fake_recalls)
    monkeypatch.setattr(workflow, 'retrieve', lambda *a, **k: (calls.append('rag') or []))
    async def fake_generate(*args):
        calls.append('llm')
        return 'Mock answer', 'llm', None
    monkeypatch.setattr(workflow, 'generate', fake_generate)
    state = asyncio.run(workflow.SequentialFallback().ainvoke({
        'message': WIPER_QUERY, 'vehicle': VEHICLE, 'history': []}))
    assert state['intent'] == 'automotive'
    assert calls == ['classifier', 'nhtsa', 'rag', 'llm']


def test_unrelated_request_still_does_not_query_external_services(monkeypatch):
    async def fail(**kwargs):
        raise AssertionError('Must not query NHTSA for out-of-scope content')
    monkeypatch.setattr(workflow, 'fetch_recalls', fail)
    state = asyncio.run(workflow.SequentialFallback().ainvoke({
        'message': 'Tell me about elephants', 'vehicle': VEHICLE, 'history': []}))
    assert state['intent'] == 'out_of_scope'
    assert state['recall_status'] == 'not_queried'
