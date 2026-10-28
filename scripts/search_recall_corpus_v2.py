"""Query the persisted FAISS recall corpus, filtering by make/model/year FIRST.

This is a diagnostic CLI only. The live FastAPI path stays unchanged until
we validate the expanded corpus and retrieval policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_recall_corpus_v2 import DEFAULT_OUTPUT, vehicle_documents


def search(output: Path, make: str, model: str, year: int, query: str, top_k: int = 3) -> list[dict]:
    import faiss
    import numpy as np
    from agent.rag import _embedding_model
    if not query.strip() or top_k < 1:
        return []
    docs = [json.loads(line) for line in (output / 'documents.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    index = faiss.read_index(str(output / 'index.faiss'))
    if index.ntotal != len(docs):
        raise ValueError('FAISS index/document count mismatch; rebuild the corpus')
    scoped = [(i, d) for i, d in enumerate(docs) if d in vehicle_documents([d], make, model, year)]
    if not scoped:
        return []
    # Reuse stored document embeddings. Scope to the requested vehicle BEFORE ranking.
    allowed = [i for i, _ in scoped]
    doc_vectors = np.ascontiguousarray(np.vstack([index.reconstruct(i) for i in allowed]).astype(np.float32))
    subset = faiss.IndexFlatIP(index.d)
    subset.add(doc_vectors)
    q_vector = np.ascontiguousarray(np.asarray(_embedding_model().encode(
        [query], normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False), dtype=np.float32))
    scores, indices = subset.search(q_vector, min(top_k, len(scoped)))
    return [{**scoped[int(pos)][1], 'cosine_similarity': round(float(score), 4)}
            for score, pos in zip(scores[0], indices[0]) if pos >= 0]


def main() -> None:
    p = argparse.ArgumentParser(description='Offline vehicle-filtered recall search (diagnostic)')
    p.add_argument('--corpus', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--make', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--year', required=True, type=int)
    p.add_argument('--query', required=True)
    p.add_argument('--top-k', type=int, default=3)
    args = p.parse_args()
    results = search(args.corpus, args.make, args.model, args.year, args.query, args.top_k)
    print(f'Candidates for {args.year} {args.make} {args.model}: {len(results)}')
    for result in results:
        print(f'{result["id"]} cosine={result["cosine_similarity"]:.4f} component={result["component"]}')
    print('Similarity ranking only; not a relevance judgment, diagnosis, or VIN-status check.')


if __name__ == '__main__':
    main()
