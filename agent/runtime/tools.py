"""Real callable tools used by LangChain's iterative create_agent loop."""
from __future__ import annotations
import json
import logging
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain_core.messages import ToolMessage, HumanMessage
from agent.nhtsa import fetch_recalls
from agent.rag import retrieve
from agent.runtime.constants import PUBLIC_PROJECT_SCOPE
from agent.runtime.grounding import _vehicle_authorized
from backend.app.classifier import predict_category
from backend.app.config import settings

log = logging.getLogger(__name__)

def _tool_response(runtime: ToolRuntime, result: dict, updates: dict, name: str, summary: str) -> Command:
    messages = [ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=runtime.tool_call_id)]
    return Command(update={**updates, 'messages': messages,
                           'tool_events': [{'tool': name, 'status': 'complete', 'summary': summary,
                                            'turn': sum(isinstance(m, HumanMessage) for m in runtime.state.get('messages', []))}]})


@tool
async def classify_symptoms(symptoms: str, runtime: ToolRuntime) -> Command:
    """Classify reported vehicle symptoms using the fine-tuned DistilBERT model. This is NOT a mechanical diagnosis."""
    if not symptoms.strip():
        return _tool_response(runtime, {'error': 'No symptom description provided'}, {},
                              'classify_symptoms', 'No symptoms supplied')
    result = predict_category(symptoms[:3000])
    return _tool_response(runtime,
                          {'category': result.get('category'), 'labels': result.get('labels', []),
                           'method': result.get('method'), 'limitation': 'Scores are not calibrated probabilities.'},
                          {'classification': result, 'stage': 'classified'},
                          'classify_symptoms', 'Categorized reported symptoms with the ML classifier')


@tool
async def search_nhtsa_recalls(make: str, model: str, year: int, runtime: ToolRuntime) -> Command:
    """Retrieve NHTSA model-year recall campaigns for an explicitly supplied make, model and year. No VIN status."""
    vehicle = {'make': make.strip(), 'model': model.strip(), 'year': year}
    if not _vehicle_authorized(make, model, year, runtime.state):
        return _tool_response(runtime,
                              {'status': 'vehicle_details_required', 'message': 'Ask the user for explicit make, model and model year. Do not guess.'},
                              {'recall_status': 'not_queried', 'recalls': []},
                              'search_nhtsa_recalls', 'Vehicle details were not verified in user messages')
    records, status = await fetch_recalls(vehicle['make'], vehicle['model'], vehicle['year'])
    # The full records remain in agent state for evidence retrieval; only a compact summary goes to LLM.
    return _tool_response(runtime,
                          {'status': status, 'model_year_records': len(records), 'vehicle': vehicle,
                           'scope': 'Make/model/year only; VIN and repair status unverified',
                           'next_step': 'Call retrieve_recall_evidence for symptom relevance.'},
                          {'vehicle': vehicle, 'recalls': records, 'recall_status': status,
                           'documents': [], 'sources': [], 'stage': 'recalls_checked'},
                          'search_nhtsa_recalls', f'NHTSA returned {len(records)} model-year records (status={status})')


@tool
def retrieve_recall_evidence(symptoms: str, runtime: ToolRuntime) -> Command:
    """Rank NHTSA records already fetched for this vehicle against the actual symptoms. Never fabricate recall evidence."""
    state = runtime.state
    records = state.get('recalls') or []
    if state.get('recall_status') != 'ok':
        return _tool_response(runtime,
                              {'status': 'no_successful_lookup', 'candidates': []},
                              {'documents': [], 'sources': []}, 'retrieve_recall_evidence',
                              'No successful NHTSA lookup available for ranking')
    classification = state.get('classification') or None
    docs = None
    if getattr(settings, 'rag_corpus_mode', 'off').lower() in ('shadow', 'prefer') and settings.rag_backend == 'faiss':
        try:
            from agent.corpus_retrieval import retrieve_cached
            cached, _ = retrieve_cached(
                symptoms, state['vehicle'], records, state['recall_status'],
                getattr(settings, 'rag_corpus_dir', 'ml/artifacts/recall_corpus_v2'),
                classification=classification, threshold=settings.rag_min_similarity)
            if settings.rag_corpus_mode.lower() == 'prefer':
                docs = cached
        except (ImportError, OSError, RuntimeError, ValueError, KeyError):
            log.exception('Verified retrieval cache failed; using live retrieval')
    if docs is None:
        docs = retrieve(symptoms, records, classification=classification)
    sources = [{'title': f'NHTSA recall {d["id"]} — {d["component"][:65]}',
                'url': d['url'], 'source_id': d['id'], 'excerpt': d['text'][:320]} for d in docs]
    compact = [{'campaign_id': d['id'], 'component': d['component'],
                'excerpt': d['text'][:900]} for d in docs]
    return _tool_response(runtime,
                          {'candidates': compact, 'candidate_count': len(compact),
                           'caveat': 'Retrieved candidates are not verified causes or VIN-specific open recalls.'},
                          {'documents': docs, 'sources': sources, 'stage': 'evidence_retrieved'},
                          'retrieve_recall_evidence', f'Selected {len(docs)} evidence candidates')


@tool
def get_project_metrics() -> dict:
    """Return verified AutoAssist training/evaluation counts, benchmarks, and scope. Use for questions about this project."""
    return dict(PUBLIC_PROJECT_SCOPE)
