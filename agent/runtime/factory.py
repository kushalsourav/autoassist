"""The single live LangChain agent graph — provider-agnostic model factory.

Active provider is selected by ``AUTOASSIST_PROVIDER`` in ``.env`` or env:

    AUTOASSIST_PROVIDER=anthropic   # default; claude-haiku-5-5
    AUTOASSIST_PROVIDER=openrouter  # requires OPENROUTER_API_KEY + OPENROUTER_MODEL
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import dotenv_values
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from backend.app.config import settings
from agent.runtime.constants import SYSTEM_PROMPT
from agent.runtime.state import StoryState
from agent.runtime.tools import classify_symptoms, search_nhtsa_recalls, retrieve_recall_evidence, get_project_metrics
from agent.runtime.lifecycle import begin_investigation, finish_investigation
from agent.runtime.models import make_model, active_provider


def _env(name: str, default: str = "") -> str:
    """Read from os.environ first, then from the root .env file (pydantic-settings
    loads .env into its model but may not export every key to os.environ)."""
    if name in os.environ:
        return os.environ[name]
    root = Path(__file__).resolve().parents[3]  # project root
    return str(dotenv_values(root / ".env").get(name) or default)


def build_story_graph():
    provider = active_provider()

    # ── reasoning flag (only meaningful for OpenRouter) ───────────────────────
    raw_reasoning = _env("AUTOASSIST_PROVIDER_REASONING")
    reasoning_enabled = str(raw_reasoning).lower() in ("1", "true", "yes")

    # ── token budget ──────────────────────────────────────────────────────────
    if provider == "openrouter":
        max_tokens = 4096 if reasoning_enabled else 1536
        openrouter_model = settings.openrouter_model
        timeout = float(settings.openrouter_timeout_seconds)
    else:
        # Anthropic: haiku-5-5 easily fits investigations in 4096 tokens.
        max_tokens = getattr(settings, "anthropic_max_tokens", 4096)
        openrouter_model = ""
        timeout = float(getattr(settings, "anthropic_timeout_seconds", 90))

    model = make_model(
        max_tokens=max_tokens,
        timeout=timeout,
        openrouter_model=openrouter_model,
        reasoning_enabled=reasoning_enabled,
    )

    return create_agent(
        model=model,
        tools=[classify_symptoms, search_nhtsa_recalls, retrieve_recall_evidence, get_project_metrics],
        system_prompt=SYSTEM_PROMPT,
        state_schema=StoryState,
        middleware=[begin_investigation, finish_investigation],
        checkpointer=InMemorySaver(),
        name="autoassist",
    )


story_graph = build_story_graph()
