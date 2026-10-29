"""Mount the existing CopilotKit AG-UI agent on the SAME FastAPI application.

Production requests must come through the trusted Next.js server-side proxy.
For a public launch, add proper end-user auth and shared rate limiting too.
"""
from __future__ import annotations

import hmac
import os
from collections import defaultdict, deque
from threading import Lock
from time import monotonic
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import dotenv_values

_RATE_WINDOW = 60.0
_hits: dict[str, deque[float]] = defaultdict(deque)
_guard = Lock()


def _env(name: str, default: str = '') -> str:
    if name in os.environ:
        return os.environ[name]
    # Root .env is loaded by pydantic-settings, but not necessarily exported.
    root = Path(__file__).resolve().parents[2]
    return str(dotenv_values(root / '.env').get(name) or default)


def _production() -> bool:
    return _env('AUTOASSIST_ENV', 'development').lower() == 'production'


def _startup_policy() -> None:
    if not _production():
        return
    from backend.app.config import settings
    if settings.storage_mode != 'supabase':
        raise RuntimeError('Production requires STORAGE_MODE=supabase; X-Demo-User is not authentication')
    token = _env('AUTOASSIST_AGENT_TOKEN')
    if len(token) < 32:
        raise RuntimeError('Production requires a random AUTOASSIST_AGENT_TOKEN (32+ chars)')
    if _env('AUTOASSIST_PROVIDER_REASONING').lower() in ('1', 'yes', 'true'):
        raise RuntimeError('Disable provider reasoning before public deployment')
    origins = [x.strip() for x in settings.cors_origins.split(',')]
    if '*' in origins or any(origin.startswith('http://') and not origin.startswith('http://localhost') for origin in origins):
        raise RuntimeError('Production CORS origins must be specific HTTPS domains')


def _allow_request(request: Request) -> bool:
    """Basic in-process throttle; use a shared edge limiter in multi-worker prod."""
    limit = int(_env('AUTOASSIST_RATE_LIMIT_PER_MINUTE', '20'))
    limit = max(1, min(limit, 600))
    # Do not trust X-Forwarded-For sent by an untrusted client.
    address = request.client.host if request.client else 'unknown'
    key = f'{address}:{request.url.path.split("/")[1]}'
    now = monotonic()
    with _guard:
        # Avoid unbounded growth from high-cardinality IPs in long-lived servers.
        if len(_hits) > 4096:
            _hits.clear()
        visits = _hits[key]
        while visits and now - visits[0] > _RATE_WINDOW:
            visits.popleft()
        if len(visits) >= limit:
            return False
        visits.append(now)
    return True


def mount_agent(app: FastAPI) -> None:
    if getattr(app.state, 'autoassist_agent_mounted', False):
        return
    _startup_policy()

    @app.middleware('http')
    async def protect_agent(request: Request, call_next):
        path = request.url.path
        if path.startswith('/agent'):
            if _production():
                supplied = request.headers.get('x-autoassist-agent-token', '')
                expected = _env('AUTOASSIST_AGENT_TOKEN')
                if not hmac.compare_digest(supplied, expected):
                    return JSONResponse({'detail': 'Unauthorized agent request'}, status_code=401)
            if request.method == 'POST' and not _allow_request(request):
                return JSONResponse({'detail': 'Rate limit exceeded'}, status_code=429,
                                    headers={'Retry-After': '60'})
            if request.method == 'POST':
                length = request.headers.get('content-length')
                if length:
                    try:
                        too_large = int(length) > 262144
                    except ValueError:
                        return JSONResponse({'detail': 'Invalid Content-Length'}, status_code=400)
                    if too_large:
                        return JSONResponse({'detail': 'Agent request too large'}, status_code=413)
        elif path == '/api/analyze' and request.method == 'POST' and not _allow_request(request):
            return JSONResponse({'detail': 'Rate limit exceeded'}, status_code=429,
                                headers={'Retry-After': '60'})
        return await call_next(request)

    from ag_ui_langgraph import add_langgraph_fastapi_endpoint
    from copilotkit import LangGraphAGUIAgent
    from agent.story_agent import story_graph

    add_langgraph_fastapi_endpoint(
        app=app,
        agent=LangGraphAGUIAgent(
            name='autoassist',
            description='AutoAssist LangChain tool-using automotive assistant.',
            graph=story_graph,
        ),
        path='/agent',
    )
    app.state.autoassist_agent_mounted = True
