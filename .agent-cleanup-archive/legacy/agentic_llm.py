"""OpenRouter chat transport for AutoAssist's semantic agent.

No vehicle heuristics, silent keyword fallback, or provider-specific tool-call assumptions.
The agent planner validates model-authored JSON locally before tool execution.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from pydantic import ValidationError

from backend.app.config import settings

log = logging.getLogger(__name__)


class AgentLLMError(RuntimeError):
    """A user-safe indication of why an agent LLM operation failed."""


def _extract_text(payload: dict[str, Any]) -> str:
    try:
        text = payload['choices'][0]['message']['content']
        if isinstance(text, str):
            return text.strip()
        if isinstance(text, list):
            return '\n'.join(part.get('text', '') for part in text
                             if isinstance(part, dict) and isinstance(part.get('text'), str)).strip()
    except (KeyError, TypeError, IndexError):
        pass
    return ''


async def chat(messages: list[dict[str, str]], *, max_tokens: int = 900,
               temperature: float = 0.1) -> str:
    if not (settings.openrouter_api_key and settings.openrouter_model):
        raise AgentLLMError('OpenRouter is not configured for the agent server.')
    try:
        async with httpx.AsyncClient(timeout=settings.openrouter_timeout_seconds) as client:
            response = await client.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers={'Authorization': f'Bearer {settings.openrouter_api_key}',
                         'Content-Type': 'application/json', 'X-Title': 'AutoAssist agent'},
                json={'model': settings.openrouter_model, 'messages': messages,
                      'temperature': temperature, 'max_tokens': max_tokens},
            )
            response.raise_for_status()
            text = _extract_text(response.json())
            if not text:
                raise AgentLLMError('The model returned no text. Try a non-reasoning model or a higher token budget.')
            return text
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        log.warning('AutoAssist LLM HTTP %s model=%s', status, settings.openrouter_model)
        raise AgentLLMError(f'OpenRouter returned HTTP {status}; check the configured model or quota.') from exc
    except httpx.TimeoutException as exc:
        raise AgentLLMError('OpenRouter timed out. Please retry.') from exc
    except httpx.RequestError as exc:
        raise AgentLLMError('Unable to connect to OpenRouter.') from exc
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise AgentLLMError('The LLM provider returned an unexpected response.') from exc


def decode_json_object(raw: str) -> dict[str, Any]:
    """Decode one entire JSON object, including an optionally fenced response.

    Does not use eval or extract arbitrary snippets from untrusted markdown.
    """
    text = raw.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', text, flags=re.I)
    if fenced:
        text = fenced.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentLLMError('The planning model did not return valid JSON.') from exc
    if not isinstance(obj, dict):
        raise AgentLLMError('The planning model returned JSON that was not an object.')
    return obj


def _reasoning_text(delta: dict[str, Any]) -> str:
    """Extract readable provider-supplied reasoning, never encrypted/redacted fields.

    OpenRouter may send both reasoning_details and the legacy reasoning string.
    Prefer the structured field to avoid duplicating the same chunk.
    """
    details = delta.get('reasoning_details')
    readable: list[str] = []
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict):
                continue
            if item.get('type') not in ('reasoning.text', 'reasoning.summary'):
                continue
            value = item.get('text') if item['type'] == 'reasoning.text' else item.get('summary')
            if isinstance(value, str) and value.strip() and '[REDACTED]' not in value:
                readable.append(value)
    if readable:
        return ''.join(readable)
    if isinstance(details, list) and details:
        # Some providers send encrypted or redacted entries; do not fall back to a
        # possibly duplicated or redacted legacy field when details are present.
        return ''
    legacy = delta.get('reasoning')
    return legacy if isinstance(legacy, str) and '[REDACTED]' not in legacy else ''


async def stream_chat_events(messages: list[dict[str, str]], *, max_tokens: int = 1100,
                             temperature: float = 0.2, include_reasoning: bool = False):
    """Emit separate answer/reasoning SSE deltas from OpenRouter.

    Reasoning is opt-in. It is provider-exposed text, not guaranteed internal thought.
    Encrypted/redacted reasoning is never surfaced. Final response requires answer text.
    """
    if not (settings.openrouter_api_key and settings.openrouter_model):
        raise AgentLLMError('OpenRouter is not configured for the agent server.')
    body: dict[str, Any] = {
        'model': settings.openrouter_model, 'messages': messages,
        'temperature': temperature, 'max_tokens': max_tokens, 'stream': True,
    }
    if include_reasoning:
        body['reasoning'] = {'enabled': True, 'effort': 'low', 'exclude': False}
    answered = False
    done = False
    try:
        async with httpx.AsyncClient(timeout=settings.openrouter_timeout_seconds) as client:
            async with client.stream(
                'POST', 'https://openrouter.ai/api/v1/chat/completions',
                headers={'Authorization': f'Bearer {settings.openrouter_api_key}',
                         'Content-Type': 'application/json',
                         'Accept': 'text/event-stream', 'X-Title': 'AutoAssist agent'},
                json=body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(':') or not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise AgentLLMError('OpenRouter sent invalid streaming data.') from exc
                    if chunk.get('error'):
                        raise AgentLLMError('OpenRouter stopped while streaming the response.')
                    choices = chunk.get('choices') or []
                    if not choices:
                        continue
                    if choices[0].get('finish_reason') == 'error':
                        raise AgentLLMError('OpenRouter stopped while streaming the response.')
                    delta = choices[0].get('delta') or {}
                    if not isinstance(delta, dict):
                        continue
                    if include_reasoning:
                        explanation = _reasoning_text(delta)
                        if explanation:
                            yield {'kind': 'reasoning', 'text': explanation}
                    content = delta.get('content', '')
                    if isinstance(content, list):
                        content = ''.join(str(part.get('text', '')) for part in content
                                          if isinstance(part, dict) and part.get('type') == 'text')
                    if isinstance(content, str) and content:
                        answered = True
                        yield {'kind': 'answer', 'text': content}
                if not answered:
                    raise AgentLLMError('OpenRouter streamed no visible answer text.')
                if not done:
                    raise AgentLLMError('OpenRouter response stream ended unexpectedly.')
    except httpx.HTTPStatusError as exc:
        raise AgentLLMError(f'OpenRouter HTTP {exc.response.status_code} during streaming.') from exc
    except httpx.TimeoutException as exc:
        raise AgentLLMError('OpenRouter timed out during streaming.') from exc
    except httpx.RequestError as exc:
        raise AgentLLMError('Unable to stream from OpenRouter.') from exc


async def stream_chat(messages: list[dict[str, str]], *, max_tokens: int = 1100,
                      temperature: float = 0.2):
    """Backward-compatible answer-only interface (V3-V6 regression tests)."""
    async for event in stream_chat_events(messages, max_tokens=max_tokens, temperature=temperature):
        if event['kind'] == 'answer':
            yield event['text']
