"""Public CopilotKit agent entry point (LangChain create_agent).

The runnable graph, tools and lifecycle live under agent.runtime. Import this
module for compatibility with the existing /agent AG-UI adapter. Legacy
/api/analyze still uses agent.workflow; it is NOT imported by this graph.
"""
from agent.runtime.factory import build_story_graph, story_graph
from agent.runtime.state import StoryState
from agent.runtime.constants import PUBLIC_PROJECT_SCOPE, SYSTEM_PROMPT
from agent.runtime.grounding import _human_text, _vehicle_authorized, _ground_answer
from agent.runtime.tools import (
    classify_symptoms, search_nhtsa_recalls, retrieve_recall_evidence,
    get_project_metrics, _tool_response,
)
from agent.runtime.lifecycle import begin_investigation, finish_investigation, _snapshot

__all__ = [
    'story_graph', 'build_story_graph', 'StoryState', 'PUBLIC_PROJECT_SCOPE',
    'SYSTEM_PROMPT', 'classify_symptoms', 'search_nhtsa_recalls',
    'retrieve_recall_evidence', 'get_project_metrics',
    'begin_investigation', 'finish_investigation',
]
