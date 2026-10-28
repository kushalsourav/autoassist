"""CopilotKit-facing LangGraph agent: story-first vehicle context and actual execution stages.

Keeps the existing classifier, NHTSA, RAG, and generation logic unchanged.
The AG-UI adapter streams state at graph-node boundaries; OpenRouter text is still
emitted when the existing generator finishes (not token-by-token in this slice).
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from backend.app.classifier import predict_category
from agent.nhtsa import fetch_recalls
from agent.rag import retrieve
from agent.generation import generate
from backend.app.config import settings

from agent.story_vehicle import extract_vehicle, missing_vehicle_fields, vehicle_question, _SYMPTOM


def message_text(msg: BaseMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ' '.join(part.get('text', '') for part in content if isinstance(part, dict))
    return ''


class StoryState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    vehicle: dict
    symptom: str
    stage: str
    intent: str
    classification: dict
    recall_status: str
    recalls: list[dict]
    documents: list[dict]
    sources: list[dict]
    answer: str
    generation_mode: str
    generation_error: str | None


def intake_node(state: StoryState) -> dict:
    user_messages = [message_text(m) for m in state.get('messages', []) if isinstance(m, HumanMessage)]
    latest = user_messages[-1] if user_messages else ''
    vehicle = dict(state.get('vehicle') or {})
    for user_text in user_messages:
        vehicle = extract_vehicle(user_text, vehicle)
    # A follow-up giving just vehicle details should use the earlier symptom report.
    symptom = latest
    if not _SYMPTOM.search(latest):
        previous = next((txt for txt in reversed(user_messages[:-1]) if _SYMPTOM.search(txt)), '')
        if previous:
            symptom = previous
    missing = missing_vehicle_fields(vehicle)
    return {'vehicle': vehicle, 'symptom': symptom, 'stage': 'vehicle_required' if missing else 'vehicle_identified'}


def after_intake(state: StoryState) -> str:
    return 'clarify' if missing_vehicle_fields(state.get('vehicle', {})) else 'classify'


def clarify_node(state: StoryState) -> dict:
    answer = vehicle_question(state.get('vehicle', {}))
    return {'messages': [AIMessage(content=answer)], 'answer': answer,
            'stage': 'awaiting_vehicle', 'sources': [], 'documents': [],
            'classification': {}, 'recall_status': 'not_queried'}


def classify_node(state: StoryState) -> dict:
    return {'classification': predict_category(state['symptom']), 'stage': 'classified'}


async def recalls_node(state: StoryState) -> dict:
    recalls, status = await fetch_recalls(**state['vehicle'])
    return {'recalls': recalls, 'recall_status': status, 'stage': 'recalls_checked'}


def retrieve_node(state: StoryState) -> dict:
    # Respect the existing production prototype feature flag. A verified cache
    # avoids re-embedding recall documents; failed snapshot checks fall back.
    query, recalls = state['symptom'], state.get('recalls', [])
    classification = state.get('classification')
    mode = getattr(settings, 'rag_corpus_mode', 'off').lower()
    docs = None
    if mode in ('shadow', 'prefer') and settings.rag_backend == 'faiss':
        try:
            from agent.corpus_retrieval import retrieve_cached
            cached, _reason = retrieve_cached(
                query, state['vehicle'], recalls, state['recall_status'],
                getattr(settings, 'rag_corpus_dir', 'ml/artifacts/recall_corpus_v2'),
                classification=classification,
                threshold=settings.rag_min_similarity)
            if mode == 'prefer':
                docs = cached
        except (ImportError, OSError, ValueError, RuntimeError, KeyError):
            # No stale evidence: only fall back to ranking the fresh NHTSA list.
            pass
    if docs is None:
        docs = retrieve(query, recalls, classification=classification)
    sources = [{'title': f'NHTSA recall {d["id"]} — {d["component"][:65]}',
                'url': d['url'], 'source_id': d['id'], 'excerpt': d['text'][:320]} for d in docs]
    return {'documents': docs, 'sources': sources, 'stage': 'evidence_retrieved'}


async def respond_node(state: StoryState) -> dict:
    history = [{'role': 'user' if isinstance(msg, HumanMessage) else 'assistant',
                'content': message_text(msg)}
               for msg in state.get('messages', [])[-9:]
               if isinstance(msg, (HumanMessage, AIMessage))]
    answer, mode, error = await generate(state['symptom'], state['vehicle'],
                                         state['classification'], state.get('documents', []),
                                         state['recall_status'], history[:-1])
    return {'messages': [AIMessage(content=answer)], 'answer': answer, 'stage': 'complete',
            'generation_mode': mode, 'generation_error': error}


def build_story_graph():
    graph = StateGraph(StoryState)
    graph.add_node('understand_vehicle', intake_node)
    graph.add_node('ask_vehicle_details', clarify_node)
    graph.add_node('classify_symptoms', classify_node)
    graph.add_node('search_nhtsa_recalls', recalls_node)
    graph.add_node('rank_recall_evidence', retrieve_node)
    graph.add_node('compose_grounded_answer', respond_node)
    graph.add_edge(START, 'understand_vehicle')
    graph.add_conditional_edges('understand_vehicle', after_intake, {
        'clarify': 'ask_vehicle_details', 'classify': 'classify_symptoms'})
    graph.add_edge('ask_vehicle_details', END)
    graph.add_edge('classify_symptoms', 'search_nhtsa_recalls')
    graph.add_edge('search_nhtsa_recalls', 'rank_recall_evidence')
    graph.add_edge('rank_recall_evidence', 'compose_grounded_answer')
    graph.add_edge('compose_grounded_answer', END)
    # Development-only memory; not durable across restarts and not user-authenticated.
    return graph.compile(checkpointer=MemorySaver())


story_graph = build_story_graph()
