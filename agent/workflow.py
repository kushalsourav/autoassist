"""Intent route -> (automotive analysis) OR (scope/clarification reply)."""
from typing import TypedDict
try:
    from langgraph.graph import StateGraph, START, END
    HAS_LANGGRAPH = True
except ImportError:
    # Used in offline unit tests when LangGraph is not installed.
    HAS_LANGGRAPH = False

from backend.app.classifier import predict_category
from agent.intent import route_intent, intent_reply
from agent.nhtsa import fetch_recalls
from agent.rag import retrieve
from agent.generation import generate


class AnalysisState(TypedDict, total=False):
    message: str
    vehicle: dict
    history: list[dict]
    intent: str
    classification: dict
    recalls: list[dict]
    recall_status: str
    documents: list[dict]
    answer: str
    generation_mode: str
    generation_error: str | None


def intent_node(state: AnalysisState) -> dict:
    return {'intent': route_intent(state['message'], state.get('history'), vehicle=state.get('vehicle'))}


def choose_path(state: AnalysisState) -> str:
    return 'automotive' if state['intent'] == 'automotive' else 'skip'


def skip_node(state: AnalysisState) -> dict:
    return {
        'answer': intent_reply(state['intent']),
        'classification': {'category': 'Not applicable', 'confidence': None,
                           'method': 'skipped_non_automotive'},
        'documents': [], 'recall_status': 'not_queried',
        'generation_mode': 'intent_guardrail', 'generation_error': None,
    }


def classify_node(state: AnalysisState) -> dict:
    return {'classification': predict_category(state['message'])}


async def recalls_node(state: AnalysisState) -> dict:
    recalls, status = await fetch_recalls(**state['vehicle'])
    return {'recalls': recalls, 'recall_status': status}


def retrieve_node(state: AnalysisState) -> dict:
    # The live NHTSA API is still queried by recalls_node. Cache use requires
    # identical campaign IDs and text so stale/incomplete snapshots fall back.
    from backend.app.config import settings
    from agent.corpus_retrieval import retrieve_cached
    from pathlib import Path
    import logging

    query = state['message']
    live_docs = state['recalls']
    classification = state.get('classification')
    mode = settings.rag_corpus_mode.lower()
    if mode not in ('off', 'shadow', 'prefer'):
        raise ValueError('RAG_CORPUS_MODE must be off, shadow, or prefer')
    if mode != 'off' and settings.rag_backend == 'faiss':
        corpus_dir = Path(settings.rag_corpus_dir)
        if not corpus_dir.is_absolute():
            corpus_dir = settings.model_file.parents[2] / corpus_dir
        try:
            cached, reason = retrieve_cached(
                query, state['vehicle'], live_docs, state['recall_status'],
                corpus_dir, classification=classification,
                threshold=settings.rag_min_similarity)
            if cached is not None and mode == 'prefer':
                return {'documents': cached}
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
            cached, reason = None, type(exc).__name__
        logging.getLogger(__name__).info('recall corpus %s: %s', mode, reason)
    else:
        cached = None
    live = retrieve(query, live_docs, classification=classification)
    if mode == 'shadow' and cached is not None:
        a, b = [d['id'] for d in cached], [d['id'] for d in live]
        logging.getLogger(__name__).info('recall corpus shadow matches=%s ids=%s live_ids=%s',
                                         a == b, a, b)
    return {'documents': live}


async def generate_node(state: AnalysisState) -> dict:
    answer, mode, error = await generate(
        state['message'], state['vehicle'], state['classification'],
        state['documents'], state['recall_status'], state.get('history'))
    return {'answer': answer, 'generation_mode': mode, 'generation_error': error}


class SequentialFallback:
    async def ainvoke(self, state: AnalysisState) -> AnalysisState:
        state.update(intent_node(state))
        if choose_path(state) == 'skip':
            state.update(skip_node(state))
            return state
        for node in (classify_node, recalls_node, retrieve_node, generate_node):
            state.update(await node(state) if node in (recalls_node, generate_node) else node(state))
        return state


def build_graph():
    if not HAS_LANGGRAPH:
        return SequentialFallback()
    graph = StateGraph(AnalysisState)
    graph.add_node('route_intent', intent_node)
    graph.add_node('out_of_scope_or_clarify', skip_node)
    graph.add_node('classify', classify_node)
    graph.add_node('recalls', recalls_node)
    graph.add_node('retrieve', retrieve_node)
    graph.add_node('generate', generate_node)
    graph.add_edge(START, 'route_intent')
    graph.add_conditional_edges('route_intent', choose_path, {
        'automotive': 'classify', 'skip': 'out_of_scope_or_clarify'})
    graph.add_edge('out_of_scope_or_clarify', END)
    graph.add_edge('classify', 'recalls')
    graph.add_edge('recalls', 'retrieve')
    graph.add_edge('retrieve', 'generate')
    graph.add_edge('generate', END)
    return graph.compile()


analysis_graph = build_graph()
