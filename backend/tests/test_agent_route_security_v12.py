"""Security policy and rate-limit tests, no external providers required."""
from __future__ import annotations
from types import SimpleNamespace
import pytest
from backend.app import agent_routes as mod


def test_production_requires_storage_and_secret(monkeypatch):
    from backend.app.config import settings
    monkeypatch.setattr(mod, '_production', lambda: True)
    monkeypatch.setattr(settings, 'storage_mode', 'local')
    with pytest.raises(RuntimeError, match='STORAGE_MODE=supabase'):
        mod._startup_policy()


def test_rejects_public_prod_reasoning_and_short_secret(monkeypatch):
    from backend.app.config import settings
    monkeypatch.setattr(mod, '_production', lambda: True)
    monkeypatch.setattr(settings, 'storage_mode', 'supabase')
    monkeypatch.setattr(settings, 'cors_origins', 'https://portfolio.example')
    values = {'AUTOASSIST_AGENT_TOKEN': 'short', 'AUTOASSIST_PROVIDER_REASONING': ''}
    monkeypatch.setattr(mod, '_env', lambda key, default='': values.get(key, default))
    with pytest.raises(RuntimeError, match='random AUTOASSIST_AGENT_TOKEN'):
        mod._startup_policy()
    values['AUTOASSIST_AGENT_TOKEN'] = 'a'*40
    values['AUTOASSIST_PROVIDER_REASONING'] = 'true'
    with pytest.raises(RuntimeError, match='Disable provider reasoning'):
        mod._startup_policy()


def test_rate_limit_is_bounded(monkeypatch):
    monkeypatch.setattr(mod, '_env', lambda key, default='':
                        '2' if key == 'AUTOASSIST_RATE_LIMIT_PER_MINUTE' else default)
    mod._hits.clear()
    request = SimpleNamespace(client=SimpleNamespace(host='127.0.0.1'),
                              url=SimpleNamespace(path='/agent'))
    assert mod._allow_request(request)
    assert mod._allow_request(request)
    assert not mod._allow_request(request)
