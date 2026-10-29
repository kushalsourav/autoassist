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


# Negative recall assertions are not allowed when the lookup itself failed.
# This is a last-resort guard; the model prompt/tool result must also be correct
# because an AG-UI client may display streamed tokens before after_agent runs.
_UNAVAILABLE_NO_RECALL_CLAIM = re.compile(
    r'\b(?:no|zero|0)\s+(?:(?:open|relevant|nhtsa|safety|federal|model.year)\s+)*'
    r'(?:recalls?|campaigns?|model.year records?)\b|'
    r'\b(?:no|none)\s+(?:recalls?|campaigns?)\s+(?:were\s+)?(?:found|returned|listed|identified)\b|'
    r'\b(?:recalls?|campaigns?)\s+(?:were\s+)?not\s+found\b|'
    r'\b(?:recalls?|campaigns?)\s+(?:were\s+)?(?:not found|absent)\b|'
    r'\b(?:lookup|search)\s+(?:returned|found|showed|identified)\s+(?:no|zero|0)\b',
    flags=re.I,
)

_UNAVAILABLE_SAFE_ANSWER = (
    "**NHTSA recall lookup unavailable.** I couldn't complete the model-year recall "
    "search, so the number of campaigns is unknown. This does **not** mean there "
    "are no recalls or safety defects. You can retry or check your vehicle's VIN "
    "at https://www.nhtsa.gov/recalls.\n\n"
    "For potentially safety-related symptoms, arrange a qualified "
    "inspection; the symptom classifier is not a mechanical diagnosis."
)


def _ground_answer(text: str, docs: list[dict], recall_status: str = 'not_queried') -> str:
    if recall_status == 'unavailable' and _UNAVAILABLE_NO_RECALL_CLAIM.search(text):
        # Replace rather than regex-editing natural language: a misleading claim
        # might be restated in another sentence or heading.
        return _UNAVAILABLE_SAFE_ANSWER
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
