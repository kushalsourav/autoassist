"""Deterministic vehicle-lookup eligibility and answer-grounding rules."""
from __future__ import annotations
import re
from langchain_core.messages import HumanMessage
from agent.recall_guardrails import ensure_unverified_recall_status, VIN_NOTICE

def _human_text(state: dict) -> str:
    """Only user-authored text may authorize a vehicle-specific lookup."""
    return ' '.join(str(m.content) for m in state.get('messages', [])
                    if isinstance(m, HumanMessage)).casefold()


def _vehicle_authorized(make: str, model: str, year: int, state: dict) -> bool:
    """Conservative literal evidence gate; not a make/model lookup catalog."""
    if year < 1980 or year > 2027 or not make.strip() or not model.strip():
        return False
    user_text = _human_text(state)
    previous = state.get('vehicle') or {}
    # The model cannot invent a vehicle, though it may normalize spaces/hyphens.
    def mentioned(value: str, key: str) -> bool:
        normalized = lambda s: re.sub(r'[^\w]+', '', str(s).casefold())
        return (normalized(value) in normalized(user_text) or
                normalized(value) == normalized(previous.get(key, '')))
    return (mentioned(make, 'make') and mentioned(model, 'model') and
            (re.search(r'(?<!\d)' + str(year) + r'(?!\d)', user_text) is not None or
             year == previous.get('year')))


def _ground_answer(text: str, docs: list[dict]) -> str:
    allowed = {str(d.get('id', '')).upper() for d in docs}
    result = re.sub(r'\b\d{2}V\d{6}\b',
                    lambda m: m.group(0) if m.group(0).upper() in allowed else '[unverified campaign ID omitted]',
                    text, flags=re.I)
    if docs:
        guarded = ensure_unverified_recall_status(result, True)
        # Do not duplicate a model-written VIN caveat when the guardrail adds one.
        if re.search(r'(?:VIN|vehicle).{0,160}(?:not confirmed|not verified|unverified|unknown|'
                     r'not establish|not guarantee|not necessarily|not.*repaired)',
                     result, flags=re.I | re.S):
            guarded = guarded.replace('\n\n' + VIN_NOTICE, '')
        result = guarded
    return result
