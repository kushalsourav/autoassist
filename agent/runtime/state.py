"""Agent state projected to AG-UI and remembered across turns."""
from __future__ import annotations
import operator
from typing import Annotated, NotRequired
from langchain.agents import AgentState

class StoryState(AgentState):
    """CopilotKit shared-state projections alongside create_agent message state."""
    vehicle: NotRequired[dict]
    classification: NotRequired[dict]
    recall_status: NotRequired[str]
    recalls: NotRequired[list[dict]]
    documents: NotRequired[list[dict]]
    sources: NotRequired[list[dict]]
    # Multiple tool calls in a single model step should not overwrite each other.
    tool_events: Annotated[list[dict], operator.add]
    result_history: NotRequired[list[dict]]
    followups: NotRequired[list[str]]
    stage: NotRequired[str]
    answer: NotRequired[str]
    generation_mode: NotRequired[str]
    generation_error: NotRequired[str | None]
