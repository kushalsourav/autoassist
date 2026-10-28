"""Opt-in, snapshot-validated recall retrieval using cached MiniLM vectors.

The default live workflow is unchanged. The caller MUST fetch the current NHTSA
make/model/year results first; this cache is used only when both campaign IDs
and normalized source texts match that successful live response. A failed or
unlisted vehicle request is never interpreted as zero recalls.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from agent.rag import _component_filter, _embedding_model, _matching_terms
from scripts.build_recall_corpus_v2 import normalize_campaign, normalize_vehicle, vehicle_key


@lru_cache(maxsize=2)
def _load_corpus(directory: str):
    """Read one immutable snapshot per process. Restart the API after rebuilding."""
    import faiss
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    docs = [json.loads(line) for line in (root / 'documents.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()]
    index = faiss.read_index(str(root / 'index.faiss'))
    if not docs or index.ntotal != len(docs) or index.ntotal != manifest.get('indexed_documents'):
        raise ValueError('Corpus document/index count mismatch')
    if index.d != manifest.get('embedding_dimension'):
        raise ValueError('Corpus embedding dimension mismatch')
    if manifest.get('version') != 'v2':
        raise ValueError('Unexpected corpus format')
    return docs, manifest, index


def _match_snapshot(vehicle: dict, live_docs: list[dict], live_status: str,
                    corpus_docs: list[dict], manifest: dict) -> tuple[list[tuple[int, dict]] | None, str]:
    """A cached document is eligible only if confirmed in the successful live set.

    This deliberately requires exact normalized-text matches, so changes in the
    NHTSA source trigger fallback instead of potentially outdated recall claims.
    """
    if live_status != 'ok':
        return None, f'live_status_{live_status}'
    key = vehicle_key(normalize_vehicle(vehicle))
    entries = [row for row in manifest.get('queries', [])
               if vehicle_key(normalize_vehicle(row)) == key]
    if len(entries) != 1 or entries[0].get('status') != 'ok' or entries[0].get('possible_truncation'):
        return None, 'vehicle_not_completely_indexed'
    stored = []
    for i, doc in enumerate(corpus_docs):
        try:
            if key in {vehicle_key(v) for v in doc['vehicles']}:
                stored.append((i, doc))
        except (KeyError, TypeError, ValueError):
            return None, 'invalid_vehicle_metadata'
    live_by_id = {}
    for raw in live_docs:
        doc = normalize_campaign(raw)
        if doc is None:
            return None, 'unrecognized_live_campaign'
        previous = live_by_id.get(doc['id'])
        if previous is None or len(doc['text']) > len(previous['text']):
            live_by_id[doc['id']] = doc
    stored_by_id = {d['id']: d for _, d in stored}
    if len(stored_by_id) != len(stored) or set(stored_by_id) != set(live_by_id):
        return None, 'campaign_membership_changed'
    for ident, doc in live_by_id.items():
        if hashlib.sha256(doc['text'].encode()).hexdigest() != stored_by_id[ident].get('content_sha256'):
            return None, 'recall_text_changed'
    return stored, 'snapshot_verified'


def retrieve_cached(query: str, vehicle: dict, live_docs: list[dict], live_status: str,
                    corpus_directory: str | Path, classification: dict | None = None,
                    top_k: int = 3, threshold: float = 0.35, embedder=None
                    ) -> tuple[list[dict] | None, str]:
    """Return (results, reason); None means caller MUST use live retrieval.

    Results are scored using precomputed, vehicle-scoped FAISS embeddings, then
    the same lexical/metadata rescue policy as agent.rag. The model is only
    asked to embed the query (not recall docs) on each call.
    """
    if not query.strip() or top_k < 1:
        return [], 'empty_query'
    docs, manifest, index = _load_corpus(str(Path(corpus_directory).resolve()))
    allowed, reason = _match_snapshot(vehicle, live_docs, live_status, docs, manifest)
    if allowed is None:
        return None, reason
    if not allowed:
        return [], reason
    candidates = []
    for i, doc in allowed:
        text_hits = _matching_terms(query, doc)
        metadata_accepts = bool(_component_filter(query, [doc], classification))
        explicit_id = bool(doc.get('id') and str(doc['id']).upper() in query.upper())
        if metadata_accepts or len(text_hits) >= 2 or explicit_id:
            candidates.append((i, doc, text_hits, metadata_accepts, explicit_id))
    if not candidates:
        return [], reason
    model = embedder if embedder is not None else _embedding_model()
    query_vector = np.asarray(model.encode([query], normalize_embeddings=True,
                                            convert_to_numpy=True, show_progress_bar=False),
                              dtype=np.float32)
    if query_vector.shape != (1, index.d) or not np.isfinite(query_vector).all():
        raise ValueError('Bad query embedding shape or values')
    vectors = np.stack([np.asarray(index.reconstruct(i), dtype=np.float32) for i, *_ in candidates])
    scores = vectors @ query_vector[0]
    accepted = []
    for (_, doc, hits, metadata_accepts, explicit_id), score in zip(candidates, scores):
        similarity = float(score)
        if explicit_id or len(hits) >= 2 or (metadata_accepts and similarity >= threshold):
            accepted.append(({**doc, 'score': round(similarity, 4),
                              'retrieval_backend': 'persistent_faiss_hybrid'}, hits, explicit_id))
    accepted.sort(key=lambda x: (bool(x[2]), min(len(x[1]), 5), x[0]['score']), reverse=True)
    output, seen = [], set()
    for doc, _, _ in accepted:
        if doc['id'] in seen:
            continue
        output.append(doc)
        seen.add(doc['id'])
        if len(output) >= top_k:
            break
    return output, reason
