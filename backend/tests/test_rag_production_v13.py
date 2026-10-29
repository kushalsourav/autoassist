"""Offline regression tests for the V13 retrieval tool fallback without network/LLM."""
import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / 'agent/runtime/tools.py'
DOCKER = ROOT / 'backend/Dockerfile.unified-v12'


def _extract_tool(retrieve_fn):
    source = ast.parse(TOOLS.read_text(encoding='utf-8'))
    fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'retrieve_recall_evidence')
    fn.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    response = lambda runtime, result, updates, name, summary: {'result': result, 'updates': updates, 'name': name, 'summary': summary}
    env = {'ToolRuntime': object, 'Command': object, '_tool_response': response,
           'settings': SimpleNamespace(rag_backend='faiss', rag_corpus_mode='off', rag_min_similarity=0.35),
           'retrieve': retrieve_fn,
           'log': SimpleNamespace(exception=lambda *_a, **_k: None)}
    exec(compile(module, str(TOOLS), 'exec'), env)
    return env['retrieve_recall_evidence']


def _runtime():
    return SimpleNamespace(state={
        'recalls': [{'id': '20V771000', 'text': 'camera display issue', 'component': 'BACK OVER PREVENTION', 'url': 'https://example.invalid/recall'}],
        'recall_status': 'ok', 'classification': None, 'vehicle': {'make': 'Honda', 'model': 'Accord', 'year': 2018}})


def test_faiss_success_uses_primary_path():
    called = []
    def retrieve(*args, **kwargs):
        called.append(kwargs.get('backend'))
        return _runtime().state['recalls']
    result = _extract_tool(retrieve)('rear camera black', _runtime())
    assert called == [None]
    assert result['result']['retrieval_status'] == 'ok'
    assert result['result']['candidate_count'] == 1


def test_faiss_error_recovers_with_tfidf():
    called = []
    def retrieve(*args, **kwargs):
        called.append(kwargs.get('backend'))
        if kwargs.get('backend') != 'tfidf':
            raise RuntimeError('FAISS offline')
        return _runtime().state['recalls']
    result = _extract_tool(retrieve)('rear camera black', _runtime())
    assert called == [None, 'tfidf']
    assert result['result']['retrieval_status'] == 'tfidf_fallback'
    assert result['result']['candidate_count'] == 1


def test_both_retrievers_fail_with_explicit_unavailable():
    def retrieve(*args, **kwargs):
        raise RuntimeError('unavailable')
    result = _extract_tool(retrieve)('rear camera black', _runtime())
    assert result['result']['retrieval_status'] == 'retrieval_unavailable'
    assert result['result']['candidate_count'] == 0
    assert 'do not interpret' in result['result']['caveat']


def test_docker_checks_corpus_and_preloads_minilm():
    content = DOCKER.read_text(encoding='utf-8')
    assert '/app/ml/artifacts/recall_corpus_v2/manifest.json' in content
    assert '/app/ml/artifacts/recall_corpus_v2/documents.jsonl' in content
    assert '/app/ml/artifacts/recall_corpus_v2/index.faiss' in content
    assert "SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'" in content
    assert 'HF_HOME=/app/.cache/huggingface' in content
