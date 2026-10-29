"""Per-turn lifecycle hooks and grounded conversation audit snapshots."""
from __future__ import annotations
from langchain.agents.middleware import before_agent, after_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from agent.runtime.state import StoryState
from agent.runtime.grounding import _ground_answer

@before_agent(state_schema=StoryState)
def begin_investigation(state: StoryState, runtime: Runtime) -> dict:
    """Reset per-turn display fields, while preserving conversation and vehicle memory."""
    return {'classification': {}, 'recall_status': 'not_queried', 'recalls': [], 'documents': [],
            'sources': [], 'stage': 'running', 'generation_error': None, 'followups': []}


def _snapshot(state: StoryState, turn: int) -> dict:
    events = [e for e in state.get('tool_events', []) if e.get('turn') == turn]
    return {'turn': turn, 'skill': 'tool_using_agent', 'vehicle': state.get('vehicle', {}),
            'classification': state.get('classification', {}),
            'recall_status': state.get('recall_status', 'not_queried'),
            'sources': state.get('sources', []),
            'evidence_summary': {'lookup_status': state.get('recall_status', 'not_queried'),
                                 'model_year_records': len(state.get('recalls') or []),
                                 'retrieved_candidates': len(state.get('documents') or []),
                                 'lookup_scope': 'make_model_year', 'vin_eligibility_checked': False,
                                 'repair_status_checked': False},
            'tool_events': events, 'followups': [], 'generation_mode': 'langchain_agent',
            'generation_error': None}


@after_agent(state_schema=StoryState)
def finish_investigation(state: StoryState, runtime: Runtime) -> dict:
    """Attach grounded UI projections to one finished assistant turn."""
    final = next((m for m in reversed(state.get('messages', []))
                  if isinstance(m, AIMessage) and not m.tool_calls), None)
    result = {'stage': 'complete', 'generation_mode': 'langchain_agent'}
    if final:
        # Reasoning-capable providers may return mixed content blocks. Persist
        # only the user-facing answer, not provider reasoning, in chat history.
        answer_text = (final.content if isinstance(final.content, str) else
                       ''.join(str(b.get('text', '')) for b in final.content
                               if isinstance(b, dict) and b.get('type') == 'text'))
        grounded = _ground_answer(answer_text, state.get('documents', []))
        result['answer'] = grounded
        if grounded != final.content:
            result['messages'] = [final.model_copy(update={'content': grounded})]
    turn = sum(isinstance(m, HumanMessage) for m in state.get('messages', []))
    prior = [x for x in state.get('result_history', []) if x.get('turn') != turn]
    result['result_history'] = prior + [_snapshot(state, turn)]
    return result
