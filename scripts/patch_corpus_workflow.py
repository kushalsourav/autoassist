"""Idempotent, narrow opt-in wiring for the saved recall corpus.

Use after applying the corpus-v2 patch. Backups are made when edits are needed.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch() -> None:
    cfg = ROOT / 'backend' / 'app' / 'config.py'
    workflow = ROOT / 'agent' / 'workflow.py'
    conf = cfg.read_text(encoding='utf-8')
    if 'rag_corpus_mode:' not in conf:
        anchor = "    rag_min_similarity: float = Field(default=0.35, ge=0, le=1)"
        if anchor not in conf:
            raise RuntimeError('Cannot safely patch config.py: RAG settings changed')
        revised = conf.replace(anchor, anchor + "\n    rag_corpus_mode: str = 'off'  # off | shadow | prefer\n    rag_corpus_dir: str = 'ml/artifacts/recall_corpus_v2'", 1)
        cfg.with_suffix('.py.before_corpus.bak').write_text(conf, encoding='utf-8')
        cfg.write_text(revised, encoding='utf-8')
    content = workflow.read_text(encoding='utf-8')
    if 'retrieve_cached(' not in content:
        original = '''def retrieve_node(state: AnalysisState) -> dict:
    return {'documents': retrieve(state['message'], state['recalls'],
                                  classification=state.get('classification'))}
'''
        replacement = '''def retrieve_node(state: AnalysisState) -> dict:
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
'''
        if original not in content:
            raise RuntimeError('Cannot safely patch workflow.py: retrieve_node changed')
        workflow.with_suffix('.py.before_corpus.bak').write_text(content, encoding='utf-8')
        workflow.write_text(content.replace(original, replacement, 1), encoding='utf-8')
    print('Corpus feature flag installed (default off). Restart FastAPI to load changes.')


if __name__ == '__main__':
    patch()
