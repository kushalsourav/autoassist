"""The single live LangChain agent graph and OpenRouter model factory."""
from __future__ import annotations
import os
from dotenv import dotenv_values
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from backend.app.config import settings
from agent.runtime.constants import SYSTEM_PROMPT
from agent.runtime.state import StoryState
from agent.runtime.tools import classify_symptoms, search_nhtsa_recalls, retrieve_recall_evidence, get_project_metrics
from agent.runtime.lifecycle import begin_investigation, finish_investigation

def build_story_graph():
    if not settings.openrouter_api_key or not settings.openrouter_model:
        raise RuntimeError('Set OPENROUTER_API_KEY and OPENROUTER_MODEL in the backend environment.')
    # pydantic-settings reads root .env WITHOUT exporting it to os.environ.
    # This fixes V7's missing-reasoning switch when the user edits root .env.
    raw_switch = os.getenv('AUTOASSIST_PROVIDER_REASONING')
    if raw_switch is None:
        raw_switch = dotenv_values(settings.model_config.get('env_file')).get('AUTOASSIST_PROVIDER_REASONING', '')
    reasoning_enabled = str(raw_switch).lower() in ('1', 'true', 'yes')
    model_kwargs = dict(model=settings.openrouter_model, api_key=settings.openrouter_api_key,
                        temperature=0, timeout=int(settings.openrouter_timeout_seconds * 1000),
                        max_tokens=4096 if reasoning_enabled else 1536, max_retries=0,
                        app_title='AutoAssist')
    if reasoning_enabled:
        model_kwargs['reasoning'] = {'effort': 'low', 'summary': 'auto'}
    model = ChatOpenRouter(**model_kwargs)
    # create_agent returns a compiled LangGraph; no hand-authored second workflow.
    return create_agent(model=model,
                        tools=[classify_symptoms, search_nhtsa_recalls,
                               retrieve_recall_evidence, get_project_metrics],
                        system_prompt=SYSTEM_PROMPT,
                        state_schema=StoryState,
                        middleware=[begin_investigation, finish_investigation],
                        checkpointer=InMemorySaver(), name='autoassist')


story_graph = build_story_graph()
