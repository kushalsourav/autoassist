"""Frozen six-campaign RAG retrieval pilot, offline by default.

From repo root:
  python -m scripts.evaluate_retrieval_v1
  python -m scripts.evaluate_retrieval_v1 --classification none

Requires backend/requirements-rag.txt and, for default classification=v3, the
locally trained V3 DistilBERT artifact. Does not call NHTSA or OpenRouter.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from agent import rag
from backend.app.config import settings

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'eval' / 'recall_benchmark_v1.json'
DEFAULT_OUTPUT = ROOT / 'ml' / 'artifacts' / 'rag_eval_v1'
METHODS = ('semantic_faiss', 'lexical_tfidf', 'hybrid_production')


def load_benchmark(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    docs = data['documents']
    cases = data['cases']
    if not docs or not cases:
        raise ValueError('Benchmark must contain documents and cases')
    ids = [str(d['id']) for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate campaign IDs in fixture')
    case_ids = [str(c['id']) for c in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError('Duplicate case IDs in fixture')
    for doc in docs:
        if not all(str(doc.get(k, '')).strip() for k in ('id','component','text','url')):
            raise ValueError(f'Incomplete recall fixture: {doc!r}')
    for case in cases:
        if not case.get('query', '').strip():
            raise ValueError(f'Empty query: {case!r}')
        relevant = case['relevant_ids']
        if len(relevant) != len(set(relevant)) or not set(relevant) <= set(ids):
            raise ValueError(f'Invalid gold IDs in {case["id"]}: {relevant}')
    if not any(c['relevant_ids'] for c in cases) or not any(not c['relevant_ids'] for c in cases):
        raise ValueError('Benchmark must contain positive and no-match cases')
    return data


class CachedEmbedder:
    """Cache a single CPU encoding pass; keep production ranking code unchanged."""
    def __init__(self, model: Any, texts: list[str]):
        distinct = list(dict.fromkeys(texts))
        embeddings = np.asarray(model.encode(
            distinct, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(distinct):
            raise RuntimeError('Unexpected embedding output shape')
        self.vectors = dict(zip(distinct, embeddings))

    def encode(self, texts, **kwargs):
        missing = [text for text in texts if text not in self.vectors]
        if missing:
            raise ValueError(f'Uncached benchmark text: {missing[0][:80]!r}')
        return np.stack([self.vectors[text] for text in texts]).astype(np.float32)


def classification_for_query(query: str, mode: str) -> dict | None:
    if mode == 'none':
        return None
    from backend.app.classifier import predict_category
    result = predict_category(query)
    if result.get('method') != 'finetuned_multilabel_distilbert_v3':
        raise RuntimeError('Expected the trained V3 classifier. Set MODEL_VARIANT=multilabel_distilbert_v3 '
                           'in the root .env, or use --classification none. Got: '
                           f'{result.get("method")}')
    return result


def run_methods(query: str, docs: list[dict], embedder: Any,
                classification: dict | None, top_k: int, threshold: float) -> dict[str, list[dict]]:
    # No gold labels are passed into retrieval. All methods receive the SAME documents.
    return {
        'semantic_faiss': rag._faiss_retrieve(
            query, docs, top_k=top_k, threshold=threshold, embedder=embedder),
        'lexical_tfidf': rag._tfidf_retrieve(query, docs, top_k=top_k),
        'hybrid_production': rag.retrieve(
            query, docs, top_k=top_k, classification=classification,
            backend='faiss', min_similarity=threshold, embedder=embedder),
    }


def score_prediction(expected: list[str], predicted: list[str], k: int) -> dict:
    gold = set(expected)
    returned = predicted[:k]
    hits = len(gold.intersection(returned))
    return {
        'hits': hits,
        'gold_count': len(gold),
        'returned_count': len(returned),
        'recall_at_k': hits / len(gold) if gold else None,
        'precision_at_k': hits / k,  # Unfilled ranks are counted as nonrelevant.
        'precision_of_returned': hits / len(returned) if returned else None,
        'hit_at_k': int(hits > 0) if gold else None,
        'exact_set_match': int(set(returned) == gold),
        'no_match_correct': int(not returned) if not gold else None,
    }


def summarize(rows: list[dict], k: int) -> dict:
    positive = [r for r in rows if r['gold_count'] > 0]
    negative = [r for r in rows if r['gold_count'] == 0]
    if not positive or not negative:
        raise ValueError('Need both positive and no-match cases')
    mean = lambda seq: round(sum(seq) / len(seq), 4)
    hit_count = sum(r['hits'] for r in rows)
    return {
        'cases': len(rows),
        'positive_cases': len(positive),
        'no_match_cases': len(negative),
        f'mean_recall_at_{k}': mean([r['recall_at_k'] for r in positive]),
        f'mean_precision_at_{k}': mean([r['precision_at_k'] for r in rows]),
        f'hit_rate_at_{k}': mean([r['hit_at_k'] for r in positive]),
        f'exact_set_match_at_{k}': mean([r['exact_set_match'] for r in rows]),
        'precision_of_returned': round(hit_count / sum(r['returned_count'] for r in rows), 4)
                                 if sum(r['returned_count'] for r in rows) else None,
        'no_match_specificity': mean([r['no_match_correct'] for r in negative]),
        'false_positive_rate_on_no_match': mean([int(r['returned_count'] > 0) for r in negative]),
        'retrieved_relevant_campaigns': hit_count,
        'retrieved_total_campaigns': sum(r['returned_count'] for r in rows),
    }


def evaluate(data: dict, embedder: Any, mode: str, top_k: int, threshold: float) -> tuple[dict, list[dict]]:
    docs = data['documents']
    per_method: dict[str, list[dict]] = {name: [] for name in METHODS}
    all_rows: list[dict] = []
    for case in data['cases']:
        query = case['query']
        prediction = classification_for_query(query, mode)
        results = run_methods(query, docs, embedder, prediction, top_k, threshold)
        for name, result in results.items():
            predicted = [str(item['id']) for item in result]
            stats = score_prediction(case['relevant_ids'], predicted, top_k)
            row = {
                'case_id': case['id'], 'method': name, 'query': query,
                'gold_ids': case['relevant_ids'], 'predicted_ids': predicted,
                'classification': prediction.get('category') if prediction else 'not_used',
                'scores': [item.get('score') for item in result],
                **stats,
            }
            per_method[name].append(row)
            all_rows.append(row)
    return {name: summarize(rows, top_k) for name, rows in per_method.items()}, all_rows


def save_reports(output: Path, data: dict, metrics: dict, rows: list[dict],
                 top_k: int, threshold: float, mode: str):
    output.mkdir(parents=True, exist_ok=True)
    report = {
        'benchmark_version': data['version'],
        'title': data['title'],
        'provenance': data['provenance'],
        'classification_mode': mode,
        'top_k': top_k, 'semantic_similarity_threshold': threshold,
        'methods': metrics,
        'limitations': [
            'Thirty hand-authored, manually annotated queries based on SIX campaign summaries; not an independent adjudicated test set.',
            'Recall summaries are paraphrased evaluation fixtures, not live NHTSA API payloads.',
            'Metrics describe only this fixed six-campaign corpus; no claim about all recalls or VIN applicability.',
            'No recall may be relevant within this fixture even when other NHTSA recalls might be relevant in reality.',
            'Threshold was fixed before measurement; never tune it against these cases and report the same scores as held-out.',
            'The benchmark tests recall retrieval, not agent intent, generated-answer faithfulness, or end-to-end latency.',
        ],
    }
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    keys = list(rows[0])
    for filename, data_rows in [('cases.csv', rows),
                                ('failures.csv', [r for r in rows if not r['exact_set_match']])]:
        with (output / filename).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in data_rows:
                writer.writerow({k: json.dumps(v) if isinstance(v, list) else v
                                 for k, v in row.items()})


def main():
    p = argparse.ArgumentParser(description='Run a reproducible six-campaign recall retrieval benchmark')
    p.add_argument('--benchmark', type=Path, default=DEFAULT_DATA)
    p.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--min-similarity', type=float, default=settings.rag_min_similarity)
    p.add_argument('--classification', choices=['v3', 'none'], default='v3',
                   help='v3 = actual trained classifier; none = retrieval ablation')
    args = p.parse_args()
    if args.top_k < 1:
        p.error('--top-k must be at least 1')
    data = load_benchmark(args.benchmark)
    print(f'Benchmark: {len(data["cases"])} queries, {len(data["documents"])} verified campaigns, '
          f'top_k={args.top_k}, threshold={args.min_similarity}, classification={args.classification}')
    print('Loading MiniLM CPU embeddings once (no NHTSA/LLM calls)...')
    corpus = [str(doc['text'])[:3000] for doc in data['documents']]
    queries = [case['query'] for case in data['cases']]
    embedder = CachedEmbedder(rag._embedding_model(), corpus + queries)
    metrics, rows = evaluate(data, embedder, args.classification, args.top_k, args.min_similarity)
    save_reports(args.out, data, metrics, rows, args.top_k, args.min_similarity, args.classification)
    print(f'\nRetrieval strategy          Recall@{args.top_k}   Precision@{args.top_k}    Hit@{args.top_k}   No-match FP rate')
    for name in METHODS:
        entry = metrics[name]
        print(f'{name:<26} {entry[f"mean_recall_at_{args.top_k}"]:>8.3f}   '
              f'{entry[f"mean_precision_at_{args.top_k}"]:>10.3f}   '
              f'{entry[f"hit_rate_at_{args.top_k}"]:>6.3f}   '
              f'{entry["false_positive_rate_on_no_match"]:>8.3f}')
    print(f'\nReports: {args.out / "metrics.json"}, {args.out / "cases.csv"}, '
          f'{args.out / "failures.csv"}')
    print('These are pilot fixture metrics, NOT production recall accuracy or VIN eligibility.')


if __name__ == '__main__':
    main()
