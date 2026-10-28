from uuid import uuid4
import pytest
import asyncio
from agent.rag import retrieve
from backend.app.classifier import predict_category
from backend.app.storage import SQLiteStore
from ml.prepare_nhtsa import categorize


def test_retrieval_matches_specific_document():
    docs = [{'id': 'BRAKE1', 'text': 'Brake caliper may fail with braking issues'},
            {'id': 'AIR1', 'text': 'Airbag cushion may not deploy during a collision'}]
    result = retrieve('brake failure when braking', docs)
    assert result
    assert result[0]['id'] == 'BRAKE1'
    assert result[0]['score'] > 0


def test_retrieval_empty_is_not_fabricated():
    assert retrieve('accelerating issue', []) == []


def test_no_untrained_confidence_claim():
    # On a clean checkout no model weights are committed.
    result = predict_category('The engine stalls and loses power while accelerating')
    assert result['category'] == 'Engine / Powertrain'
    if result['method'] == 'keyword_demo_only':
        assert result['confidence'] is None


def test_component_label_is_reported_not_diagnostic():
    assert categorize('SERVICE BRAKES, HYDRAULIC') == 'Brakes'
    assert categorize('STRUCTURE') is None


def test_dev_chat_history_scoped_to_user(tmp_path):
    async def check():
        store = SQLiteStore(tmp_path / 'test.db')
        alice, bob = str(uuid4()), str(uuid4())
        chat = await store.create_chat(alice, 'A vehicle complaint')
        await store.add_message(alice, chat['id'], 'user', 'The brakes vibrate')
        assert len(await store.list_messages(alice, chat['id'])) == 1
        assert await store.get_chat(bob, chat['id']) is None
        with pytest.raises(ValueError):
            await store.add_message(bob, chat['id'], 'assistant', 'Unauthorized')
    asyncio.run(check())
