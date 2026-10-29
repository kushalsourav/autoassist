"""Recall-focused retrieval: TF-IDF default or sentence-transformer + FAISS.

This searches ONLY records returned for the selected make/model/year. It is not
an indexed repair-manual knowledge base, and a model-year match does not prove
VIN applicability or recall repair status.
"""
from functools import lru_cache
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.app.config import settings
from ml.taxonomy_v3 import components_to_labels_v3


@lru_cache(maxsize=1)
def _embedding_model():
    # Load lazily, on CPU; sharing the process with the trained classifier is safe.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError('FAISS RAG requires: pip install -r backend/requirements-rag.txt') from exc
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')


def _component_filter(query: str, documents: list[dict], classification: dict | None) -> list[dict]:
    """Prefer conservative relevance over showing unrelated recall campaigns.

    This is a metadata gate, not a claim about the true cause of a symptom.
    Unknown component groups are retained for semantic ranking. Explicit
    campaign-ID requests bypass the gate. Weak/uncertain classification bypasses.
    """
    classification = classification or {}
    labels = classification.get('labels') or []
    relevant = {item.get('category') for item in labels if isinstance(item, dict)}
    relevant.discard(None)
    if not relevant:
        return documents
    if classification.get('method') != 'finetuned_multilabel_distilbert_v3':
        return documents
    result = []
    upper_query = query.upper()
    for doc in documents:
        campaign = str(doc.get('id') or '').upper()
        # E.g. "20V771000" — do not filter an explicitly requested campaign.
        if campaign and campaign in upper_query:
            result.append(doc)
            continue
        groups = set(components_to_labels_v3(str(doc.get('component') or '')))
        if not groups or (groups & relevant):
            result.append(doc)
    return result


def _tfidf_retrieve(query: str, documents: list[dict], top_k: int) -> list[dict]:
    corpus = [str(d.get('text', '')) for d in documents]
    vectors = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit_transform(corpus + [query])
    similarities = cosine_similarity(vectors[-1], vectors[:-1]).ravel()
    indices = similarities.argsort()[::-1][:top_k]
    return [{**documents[int(i)], 'score': round(float(similarities[i]), 4),
             'retrieval_backend': 'tfidf'} for i in indices if similarities[i] > 0]


def _faiss_retrieve(query: str, documents: list[dict], top_k: int,
                    threshold: float, embedder=None) -> list[dict]:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('FAISS RAG requires: pip install -r backend/requirements-rag.txt') from exc
    model = embedder if embedder is not None else _embedding_model()
    corpus = [str(d.get('text') or '')[:3000] for d in documents]
    # Normalize both sides so FAISS inner product is cosine similarity.
    embeddings = np.asarray(model.encode(corpus + [query], normalize_embeddings=True,
                                        convert_to_numpy=True, show_progress_bar=False),
                            dtype=np.float32)
    vectors = np.ascontiguousarray(embeddings[:-1])
    qvector = np.ascontiguousarray(embeddings[-1:])
    if vectors.ndim != 2 or qvector.shape != (1, vectors.shape[1]):
        raise RuntimeError('Invalid sentence-embedding shape')
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    scores, indices = index.search(qvector, min(top_k, len(documents)))
    return [{**documents[int(i)], 'score': round(float(score), 4),
             'retrieval_backend': 'faiss_minilm_cosine'}
            for score, i in zip(scores[0], indices[0])
            if i >= 0 and float(score) >= threshold]


# A weak label can disagree with a recall's official component.  In particular,
# VISIBILITY/WIPER and BACK OVER PREVENTION are not currently mapped to Electrical,
# even when their recall narrative mentions BCM/software and camera failures.
# Do not let the component gate discard strong text evidence before ranking.
#
# Critically, do NOT put common symptom-description words here — terms like
# "engine", "power", "light", "fuel", "accelerat*" are exactly what NHTSA
# recall narratives share with user-reported symptoms. Removing them leaves
# zero shared terms between the query and every document, defeating the lexical
# rescue path entirely.
_LEXICAL_STOP = frozenset({
    'about', 'after', 'again', 'also', 'always', 'another', 'because', 'been',
    'before', 'could', 'does', 'during', 'each', 'every', 'from', 'have',
    'having', 'into', 'intermittent', 'intermittently', 'just',
    'many', 'might', 'more', 'most', 'often', 'once', 'only', 'other',
    'problem', 'problems', 'reported', 'same', 'some', 'sometimes',
    'stop', 'stopped', 'stops', 'such', 'than', 'that', 'their', 'them', 'there',
    'these', 'this', 'those', 'through', 'under', 'very',
    'were', 'what', 'when', 'where', 'which', 'while', 'with', 'work', 'working',
    'would', 'your', 'recall', 'recalls', 'campaign', 'number', 'issue', 'issues',
    'model', 'year', 'certain', 'honda', 'car',
    'cars', 'may', 'will', 'the', 'and', 'are', 'for', 'not', 'has', 'had', 'can',
    'certain', 'equipped', 'affected', 'vehicles', 'vehicle',
})


def _lexical_terms(text: str) -> set[str]:
    # Normalize singular/plural for common component words without heavy NLP.
    words = re.findall(r'[a-z]{3,}', (text or '').lower())
    return {(word[:-1] if len(word) > 4 and word.endswith('s') and not word.endswith('ss')
             else word) for word in words if word not in _LEXICAL_STOP}


def _matching_terms(query: str, document: dict) -> list[str]:
    return sorted(_lexical_terms(query) & _lexical_terms(str(document.get('text') or '')))


def _faiss_hybrid_retrieve(query: str, documents: list[dict], top_k: int,
                           threshold: float, classification: dict | None,
                           embedder=None) -> list[dict]:
    """FAISS semantic ranking with component-metadata and lexical rescue gates.

    Eligibility: All documents pass to FAISS unless the component filter
    explicitly *rejects* them AND they have no lexical overlap with the query.
    This is intentionally permissive at the eligibility stage — FAISS cosine
    similarity does the actual relevance ranking, and the threshold gate at the
    end removes low-quality matches.  Over-filtering at eligibility is worse
    than under-filtering because it silently drops genuinely relevant records.

    Keep policy: a candidate is kept if:
      * It was explicitly requested by campaign ID (always keep), OR
      * It has ≥2 shared non-stopword terms with the query (lexical rescue), OR
      * It passes the component metadata gate AND its cosine score ≥ threshold.

    No matching campaign is ever fabricated if NHTSA returned none.
    """
    # Cheap eligibility: run FAISS on all documents. The component filter is
    # used only at the keep stage, not as a pre-filter gate, so that FAISS can
    # find semantically relevant records even when weak-label categories disagree
    # with NHTSA's official component taxonomy (e.g. "Engine/Fuel" symptoms
    # matching a POWERTRAIN:TCM recall that the label model missed).
    if not documents:
        return []
    candidates = _faiss_retrieve(query, documents, top_k=len(documents),
                                 threshold=-2.0, embedder=embedder)
    keep = []
    for row in candidates:
        metadata_accepts = bool(_component_filter(query, [row], classification))
        matched_terms = _matching_terms(query, row)
        explicit_id = bool(row.get('id') and str(row['id']).upper() in query.upper())
        narrative_accepts = len(matched_terms) >= 2
        if explicit_id or narrative_accepts or (row['score'] >= threshold and metadata_accepts):
            keep.append((row, matched_terms, explicit_id))
        elif row['score'] >= threshold:
            # Component filter disagreed but semantic similarity is high — keep
            # with a slightly raised bar: require score above a stricter cutoff
            # rather than silently discarding. This rescues records where the
            # weak-label category simply does not cover the recall's component.
            if row['score'] >= max(threshold, 0.25):
                keep.append((row, matched_terms, explicit_id))

    # Deduplicate campaigns that appear under multiple official component names.
    keep.sort(key=lambda x: (bool(x[2]), min(len(x[1]), 5), x[0]['score']), reverse=True)
    output = []
    seen = set()
    for row, _, _ in keep:
        key = str(row.get('id') or '')
        if key and key in seen:
            continue
        seen.add(key)
        output.append(row)
        if len(output) >= top_k:
            break
    return output


def retrieve(query: str, documents: list[dict], top_k: int = 3,
             classification: dict | None = None, backend: str | None = None,
             min_similarity: float | None = None, embedder=None) -> list[dict]:
    """Return evidence with a relevance gate, or [] if none passes.

    min_similarity is a tunable development heuristic, not a calibrated
    probability. Use held-out relevance judgments before production deployment.
    """
    if not query.strip() or not documents or top_k < 1:
        return []
    backend = backend or settings.rag_backend
    if backend == 'faiss':
        return _faiss_hybrid_retrieve(
            query, documents, top_k,
            settings.rag_min_similarity if min_similarity is None else min_similarity,
            classification, embedder=embedder)
    if backend == 'tfidf':
        selected = _component_filter(query, documents, classification)
        return _tfidf_retrieve(query, selected, top_k) if selected else []
    raise ValueError(f'Unsupported RAG_BACKEND={backend!r}; use tfidf or faiss')
