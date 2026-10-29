"""Agent state projected to AG-UI and remembered across turns."""
from __future__ import annotations

import operator
from typing import Annotated, NotRequired

from langchain.agents import AgentState


def _last(old, new):
    """Last-write-wins reducer — allows multiple nodes to write the same key
    in one superstep without raising InvalidUpdateError."""
    return new


class StoryState(AgentState):
    """CopilotKit shared-state projections alongside create_agent message state.

    Fields that are written by more than one node / middleware in the same
    superstep **must** carry an Annotated reducer; plain fields use LangGraph's
    default ``last_value`` channel which rejects concurrent writes.

    Two reducer strategies are used here:
    * ``_last``         — last-write-wins for scalar / dict fields
    * ``operator.add``  — append-only for list fields (tool_events)
    """

    vehicle: Annotated[NotRequired[dict], _last]
    classification: Annotated[NotRequired[dict], _last]
    recall_status: Annotated[NotRequired[str], _last]
    recalls: Annotated[NotRequired[list], _last]
    documents: Annotated[NotRequired[list], _last]
    sources: Annotated[NotRequired[list], _last]
    # Multiple tool calls in a single model step must append, not overwrite.
    tool_events: Annotated[list[dict], operator.add]
    result_history: Annotated[NotRequired[list[dict]], _last]
    followups: Annotated[NotRequired[list[str]], _last]
    # Written by begin_investigation, tools, and finish_investigation —
    # concurrent writes are expected; the last one wins.
    stage: Annotated[NotRequired[str], _last]
    answer: Annotated[NotRequired[str], _last]
    generation_mode: Annotated[NotRequired[str], _last]
    generation_error: Annotated[NotRequired[str | None], _last]
