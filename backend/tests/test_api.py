from uuid import uuid4
from fastapi.testclient import TestClient
import backend.app.main as main
from backend.app.storage import SQLiteStore


def test_health_and_first_chat_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'local_store', SQLiteStore(tmp_path / 'test.db'))
    assert main.settings.storage_mode == 'local'

    class FakeGraph:
        async def ainvoke(self, state):
            assert state['vehicle']['year'] == 2018
            return {
                'classification': {'category': 'Brakes', 'confidence': None, 'method': 'keyword_demo_only'},
                'documents': [{'id': 'TEST123', 'component': 'Brake parts',
                               'text': 'Officially sourced record from NHTSA in live mode',
                               'url': 'https://api.nhtsa.gov/recalls/recallsByVehicle?make=Honda&model=Accord&modelYear=2018'}],
                'answer': 'Test reply: not a confirmed diagnosis.',
                'recall_status': 'ok',
                'generation_mode': 'evidence_only',
            }

    monkeypatch.setattr(main, 'analysis_graph', FakeGraph())
    alice, bob = str(uuid4()), str(uuid4())
    client = TestClient(main.app)
    assert client.get('/api/health').status_code == 200
    unauthenticated = client.get('/api/chats')
    assert unauthenticated.status_code == 401
    user_headers = {'X-Demo-User': alice}
    result = client.post('/api/analyze', headers=user_headers,
                         json={'message': 'Brakes vibrate when pressing the pedal',
                               'make': 'Honda', 'model': 'Accord', 'year': 2018})
    assert result.status_code == 200, result.text
    payload = result.json()
    assert payload['sources'][0]['source_id'] == 'TEST123'
    cid = payload['conversation_id']
    assert len(client.get('/api/chats', headers=user_headers).json()) == 1
    saved = client.get(f'/api/chats/{cid}/messages', headers=user_headers)
    assert saved.status_code == 200
    assert [item['role'] for item in saved.json()] == ['user', 'assistant']
    assert saved.json()[1]['metadata']['classification']['category'] == 'Brakes'
    assert client.get(f'/api/chats/{cid}/messages', headers={'X-Demo-User': bob}).status_code == 404


def test_invalid_message_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'local_store', SQLiteStore(tmp_path / 'test.db'))
    response = TestClient(main.app).post('/api/analyze', headers={'X-Demo-User': str(uuid4())},
        json={'message': 'no', 'make': 'Honda', 'model': 'Accord', 'year': 2018})
    assert response.status_code == 422
