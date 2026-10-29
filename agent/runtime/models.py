"""Isolated model client registry — Anthropic (default) and OpenRouter.

Selecting the active provider
------------------------------
Set ``AUTOASSIST_PROVIDER`` in your ``.env`` (or shell):

    AUTOASSIST_PROVIDER=anthropic   # default — claude-haiku-5-5 unless overridden
    AUTOASSIST_PROVIDER=openrouter  # uses OPENROUTER_API_KEY / OPENROUTER_MODEL

Only these two providers are compiled in.  The factory (factory.py) calls
:func:`make_model` and gets back a ready LangChain chat model.

Anthropic prompt caching
-------------------------
Two cache breakpoints are inserted on every invoke (Anthropic allows 4):
  1. End of the system message   → caches tools + system prompt, constant every round
  2. Last human / tool message   → caches the growing conversation prefix
This is done transparently via a :class:`CachedChatAnthropic` subclass so
the graph / tools do not need to know about it.

Usage tracking
--------------
:func:`usage_entry` returns a structured record for one model call.  Attach
it to your metrics store or log it as JSON.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
_CACHE_CONTROL = {"type": "ephemeral"}

# These Anthropic models reject forced ``any``/named tool choice.
_NO_FORCED_TOOL_CHOICE_PREFIXES = (
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
)


# ─────────────────────────────────────────── provider selection

def active_provider() -> str:
    """Return the normalised provider name (default: anthropic).

    Checks os.environ first (explicit shell export / Docker env), then falls
    back to the pydantic-settings model which reads the project's .env file.
    pydantic-settings intentionally does NOT re-export .env values to
    os.environ, so a plain os.getenv() call is not enough.
    """
    raw = (os.getenv("AUTOASSIST_PROVIDER") or _settings().autoassist_provider or "anthropic").strip().lower()
    if raw not in ("anthropic", "openrouter"):
        raise ValueError(
            f"AUTOASSIST_PROVIDER={raw!r} is not supported. "
            "Choose 'anthropic' (default) or 'openrouter'."
        )
    return raw


def _settings():
    """Lazy import to avoid circular imports at module load time."""
    from backend.app.config import settings as _s
    return _s


def anthropic_model_name() -> str:
    # os.environ is checked first (explicit shell export wins); pydantic-settings
    # reads .env into the Settings model but does NOT export to os.environ.
    return os.getenv("ANTHROPIC_MODEL") or _settings().anthropic_model or DEFAULT_ANTHROPIC_MODEL


# ─────────────────────────────────────────── tool-choice normalisation

def normalize_tool_choice(model: str, tool_choice: str | dict | None):
    """Translate the graph's required-tool request to a valid Anthropic choice."""
    if tool_choice != "required":
        return tool_choice
    if model.startswith(_NO_FORCED_TOOL_CHOICE_PREFIXES):
        # These models support tools but reject forced ``any``/named choices.
        return "auto"
    return "any"


# ─────────────────────────────────────────── prompt caching helpers

def _with_cache_breakpoint(msg):
    """Return a copy of *msg* with cache_control appended to its last text block."""
    content = msg.content
    if isinstance(content, str):
        if not content.strip():
            return msg
        blocks = [{"type": "text", "text": content, "cache_control": _CACHE_CONTROL}]
    elif isinstance(content, list) and content:
        blocks = list(content)
        last = blocks[-1]
        if isinstance(last, str):
            blocks[-1] = {"type": "text", "text": last, "cache_control": _CACHE_CONTROL}
        elif isinstance(last, dict) and last.get("type") == "text":
            blocks[-1] = {**last, "cache_control": _CACHE_CONTROL}
        else:
            return msg
    else:
        return msg
    return msg.model_copy(update={"content": blocks})


def mark_cache_breakpoints(messages: list) -> list:
    """Insert at most two cache breakpoints per turn (Anthropic allows 4 total).

    Breakpoint 1 — end of the system message (caches tools + system, same every round).
    Breakpoint 2 — last human/tool message  (caches the growing conversation prefix).
    """
    out = list(messages)
    targets = [i for i, m in enumerate(out) if isinstance(m, SystemMessage)][:1]
    if out and isinstance(out[-1], (HumanMessage, ToolMessage)):
        targets.append(len(out) - 1)
    for i in dict.fromkeys(targets):
        out[i] = _with_cache_breakpoint(out[i])
    return out


# ─────────────────────────────────────────── Anthropic client

@lru_cache(maxsize=1)
def _cached_anthropic_class():
    from langchain_anthropic import ChatAnthropic

    class CachedChatAnthropic(ChatAnthropic):
        """ChatAnthropic that silently injects prompt-cache breakpoints on every invoke."""

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            tool_choice = normalize_tool_choice(self.model, tool_choice)
            return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)

        def invoke(self, input, config=None, *, stop=None, **kwargs):
            if isinstance(input, list):
                input = mark_cache_breakpoints(input)
            return super().invoke(input, config, stop=stop, **kwargs)

    return CachedChatAnthropic


def _make_anthropic(max_tokens: int, timeout: float) -> object:
    # pydantic-settings does NOT export .env values to os.environ, so check
    # both sources: explicit env-var first, then the Settings model.
    api_key = (os.getenv("ANTHROPIC_API_KEY") or _settings().anthropic_api_key or "").strip()
    if not api_key:
        raise RuntimeError(
            "AUTOASSIST_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
            "Add it to your .env or environment."
        )
    return _cached_anthropic_class()(
        model=anthropic_model_name(),
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=0,
        api_key=api_key,
    )


# ─────────────────────────────────────────── OpenRouter client

def _make_openrouter(openrouter_model: str, max_tokens: int, timeout: float,
                     reasoning_enabled: bool = False) -> object:
    # Check os.environ first (explicit export), then the Settings model.
    api_key = (os.getenv("OPENROUTER_API_KEY") or _settings().openrouter_api_key or "").strip()
    if not api_key:
        raise RuntimeError(
            "AUTOASSIST_PROVIDER=openrouter but OPENROUTER_API_KEY is not set. "
            "Add it to your .env or environment."
        )
    if not openrouter_model:
        raise RuntimeError(
            "AUTOASSIST_PROVIDER=openrouter but OPENROUTER_MODEL is not set. "
            "Add it to your .env or environment."
        )
    from langchain_openrouter import ChatOpenRouter

    model_kwargs: dict = dict(
        model=openrouter_model,
        api_key=api_key,
        temperature=0,
        timeout=int(timeout * 1000),
        max_tokens=max_tokens,
        max_retries=0,
        app_title="AutoAssist",
    )
    if reasoning_enabled:
        model_kwargs["reasoning"] = {"effort": "low", "summary": "auto"}
    return ChatOpenRouter(**model_kwargs)


# ─────────────────────────────────────────── unified factory

def make_model(
    *,
    max_tokens: int = 4096,
    timeout: float = 90.0,
    openrouter_model: str = "",
    reasoning_enabled: bool = False,
) -> object:
    """Return a ready LangChain chat model for the configured provider.

    Parameters
    ----------
    max_tokens:
        Hard cap on output tokens (passed through to the provider).
    timeout:
        Request timeout in **seconds** for Anthropic; converted to ms for OpenRouter.
    openrouter_model:
        Model identifier used when ``AUTOASSIST_PROVIDER=openrouter``.
    reasoning_enabled:
        When True and provider is OpenRouter, request low-effort reasoning summaries.
    """
    provider = active_provider()
    if provider == "anthropic":
        return _make_anthropic(max_tokens=max_tokens, timeout=timeout)
    return _make_openrouter(
        openrouter_model=openrouter_model,
        max_tokens=max_tokens,
        timeout=timeout,
        reasoning_enabled=reasoning_enabled,
    )


# ─────────────────────────────────────────── usage tracking

def usage_entry(
    raw: AIMessage | None,
    stage: str,
    start: float,
    error: str | None = None,
) -> dict:
    """Structured token-usage, cache, and latency record for one model call.

    Drop the returned dict into your audit log / metrics store.
    """
    provider = active_provider()
    metadata = raw.response_metadata if raw else {}
    usage = raw.usage_metadata if raw else None
    details = (usage or {}).get("input_token_details") or {}

    if provider == "anthropic":
        requested_model = anthropic_model_name()
        actual_model = metadata.get("model_name", metadata.get("model"))
    else:
        requested_model = os.getenv("OPENROUTER_MODEL", "")
        actual_model = metadata.get("model_name", metadata.get("model"))

    return {
        "stage": stage,
        "provider": provider,
        "requested_model": requested_model,
        "actual_model": actual_model,
        "usage": usage,
        # Anthropic-specific cache fields (None for OpenRouter)
        "cache_read_tokens": details.get("cache_read"),
        "cache_creation_tokens": details.get("cache_creation"),
        "cost_usd": None,  # Neither Anthropic nor OpenRouter report cost in-response
        "latency_seconds": round(time.monotonic() - start, 3),
        "error": error,
    }

