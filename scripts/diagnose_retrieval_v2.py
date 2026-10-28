"""Offline audit of 18-query recall benchmark failures (no production changes).

For each vehicle-scoped query, inspect the exact cosine, TF-IDF and narrative
match evidence for every recall. Also evaluate a *candidate-stage* union of
semantic and lexical top-N without treating the union as final citations.

Run after evaluate_multivehicle_eval_v2. Does not access NHTSA or an LLM.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent.rag import _embedding_model, _matching_terms
from agent.corpus_retrieval import _load_corpus
from scripts.build_recall_corpus_v2 import normalize_vehicle, vehicle_key
from scripts.evaluate_multivehicle_eval_v2 import read_reviewed_labels, verify_snapshot
from scripts.prepare_multivehicle_eval_v2 import (
    DEFAULT_CORPUS, DEFAULT_OUT, DEFAULT_QUERIES, build_rows, load_inputs
)


def rank_all(query: str, docs: list[dict], vectors: np.ndarray, query_vector: np.ndarray) -> list[dict]:
    """Return all vehicle-scoped campaigns, independently of their gold labels."""
    if not docs:
        return []
    if vectors.shape[0] != len(docs) or vectors.shape[1] != len(query_vector):
        raise ValueError('Index vectors do not match query/document dimensions')
    semantic = vectors @ query_vector
    texts = [str(d['text']) for d in docs]
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit_transform(texts + [query])
    lexical = cosine_similarity(tfidf[-1], tfidf[:-1]).ravel()
    results = []
    for doc, sim, lex in zip(docs, semantic, lexical):
        terms = _matching_terms(query, doc)
        results.append({
            'campaign_id': doc['id'], 'component': doc.get('component', ''),
            'cosine': round(float(sim), 5), 'tfidf': round(float(lex), 5),
            'matched_terms': terms, 'narrative_match': len(terms) >= 2,
        })
    semantic_rank = sorted(results, key=lambda d: (-d['cosine'], d['campaign_id']))
    lexical_rank = sorted(results, key=lambda d: (-d['tfidf'], d['campaign_id']))
    for i, d in enumerate(semantic_rank, start=1):
        d['semantic_rank'] = i
    for i, d in enumerate(lexical_rank, start=1):
        d['lexical_rank'] = i
    return results


def candidate_union(ranked: list[dict], top_n: int) -> set[str]:
    """Candidate generation only; DO NOT pass these records to an LLM as citations."""
    if top_n < 1:
        raise ValueError('top_n must be at least 1')
    return {r['campaign_id'] for r in ranked
            if (r['semantic_rank'] <= top_n or (r['lexical_rank'] <= top_n and r['tfidf'] > 0))}


def audit(cases: list[dict], docs: list[dict], index, model,
          gold: dict[str, list[str]], predictions: dict[str, list[str]],
          top_n: int = 5, threshold: float = 0.35):
    if not -1.0 <= threshold <= 1.0:
        raise ValueError('threshold must be a cosine value between -1 and 1')
    queries = list(dict.fromkeys(c['query'] for c in cases))
    query_vectors = np.asarray(model.encode(
        queries, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ), dtype=np.float32)
    if query_vectors.shape != (len(queries), index.d):
        raise ValueError('Query embedding dimensions are incompatible with frozen index')
    vectors = np.stack([np.asarray(index.reconstruct(i), dtype=np.float32)
                        for i in range(len(docs))])
    queries_by_text = dict(zip(queries, query_vectors))
    rows, summary = [], []
    for case in cases:
        key = vehicle_key(normalize_vehicle(case))
        ids = [i for i,d in enumerate(docs) if key in {vehicle_key(v) for v in d['vehicles']}]
        scoped = [docs[i] for i in ids]
        ranked = rank_all(case['query'], scoped, vectors[ids], queries_by_text[case['query']])
        candidate_ids = candidate_union(ranked, top_n)
        expected = set(gold.get(case['id'], []))
        actual = set(predictions.get(case['id'], []))
        if not expected <= {d['id'] for d in scoped}:
            raise ValueError(f"Gold labels not present in vehicle scope: {case['id']}")
        for r in ranked:
            ident = r['campaign_id']
            accepted_without_classifier = r['narrative_match'] or r['cosine'] >= threshold
            if ident in expected and ident not in actual:
                reason = 'below_both_gates' if not accepted_without_classifier else 'rank_or_classifier_gate'
            elif ident not in expected and ident in actual:
                reason = 'extra_source'
            else:
                reason = ''
            rows.append({
                'case_id': case['id'], 'query': case['query'],
                'vehicle': f"{case['year']} {case['make']} {case['model']}",
                **r, 'gold_relevant': int(ident in expected),
                'hybrid_returned': int(ident in actual),
                'in_candidate_union': int(ident in candidate_ids),
                'passes_gates_without_classifier': int(accepted_without_classifier),
                'failure_reason': reason,
            })
        summary.append({
            'case_id': case['id'], 'query': case['query'],
            'gold_ids': sorted(expected), 'hybrid_ids': sorted(actual),
            'candidate_ids': sorted(candidate_ids),
            'missed_by_hybrid': sorted(expected - actual),
            'missed_by_candidate_stage': sorted(expected - candidate_ids),
            'extra_hybrid': sorted(actual - expected),
            'positive': bool(expected),
            'candidate_recall': len(expected & candidate_ids) / len(expected) if expected else None,
        })
    return rows, summary


def read_predictions(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}; run the V2 evaluation first')
    with path.open(newline='', encoding='utf-8') as f:
        data = list(csv.DictReader(f))
    relevant = [r for r in data if r['method'] == 'hybrid_live']
    if not relevant:
        raise ValueError('No hybrid_live predictions in cases.csv')
    output = {}
    for row in relevant:
        if row['case_id'] in output:
            raise ValueError('Duplicate case_id in cases.csv')
        output[row['case_id']] = json.loads(row['predicted_ids'])
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    p.add_argument('--queries', type=Path, default=DEFAULT_QUERIES)
    p.add_argument('--eval-dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--top-n', type=int, default=5)
    p.add_argument('--threshold', type=float, default=0.35)
    args = p.parse_args()
    if args.top_n < 1:
        p.error('--top-n must be >= 1')
    verify_snapshot(args.corpus, args.queries, args.eval_dir)
    _, docs, cases = load_inputs(args.corpus, args.queries)
    gold = read_reviewed_labels(args.eval_dir / 'judgments.csv', build_rows(docs, cases))
    stored, _, index = _load_corpus(str(args.corpus.resolve()))
    if [d['id'] for d in stored] != [d['id'] for d in docs]:
        raise ValueError('Frozen embedding index does not match reviewed corpus')
    predicted = read_predictions(args.eval_dir / 'results_v3' / 'cases.csv')
    if set(predicted) != {c['id'] for c in cases}:
        raise ValueError('V3 benchmark cases mismatch')
    rows, cases_report = audit(cases, docs, index, _embedding_model(), gold, predicted,
                              top_n=args.top_n, threshold=args.threshold)
    target = args.eval_dir / 'diagnostics_v2'
    target.mkdir(parents=True, exist_ok=True)
    with (target / 'campaign_evidence.csv').open('w', newline='', encoding='utf-8') as f:
        fields = list(rows[0])
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: json.dumps(v) if isinstance(v, list) else v for k,v in row.items()})
    with (target / 'cases.json').open('w', encoding='utf-8') as f:
        json.dump(cases_report, f, indent=2)
    positive = [c for c in cases_report if c['positive']]
    matched = sum(not c['missed_by_candidate_stage'] for c in positive)
    macro_recall = sum(c['candidate_recall'] for c in positive) / len(positive)
    total_expected = sum(len(c['gold_ids']) for c in positive)
    retrieved_expected = sum(len(set(c['gold_ids']) & set(c['candidate_ids'])) for c in positive)
    print(f'Candidate union top-{args.top_n} per method | positive cases={len(positive)}')
    print(f'Candidate-stage macro recall: {macro_recall:.3f} (NOT final retrieval Recall@3)')
    print(f'Candidate-stage relevant IDs: {retrieved_expected}/{total_expected}; full-hit cases={matched}/{len(positive)}')
    print('No-match cases:', sum(not c['positive'] for c in cases_report),
          '(candidate lists are NOT final citations; no abstention is applied)')
    print('Hybrid misses and evidence:')
    for case in cases_report:
        if case['missed_by_hybrid']:
            print(' ', case['case_id'], 'missed:', ', '.join(case['missed_by_hybrid']),
                  'candidate_pool_misses:', ', '.join(case['missed_by_candidate_stage']) or 'none')
    print('Evidence:', target / 'campaign_evidence.csv')
    print('Case summary:', target / 'cases.json')
    print('Pilot data has been inspected during development. Do not present these diagnostic numbers as held-out gains.')


if __name__ == '__main__':
    main()
