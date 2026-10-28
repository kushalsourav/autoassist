"""Offline-only two-stage recall retrieval helpers.

Candidate generation reuses the existing vehicle-scoped semantic+TFIDF union.
Cross-encoder scores are for ordering/analysis only: NO abstention is implied,
NO score is a probability of recall applicability, and no result should be used
as an automotive answer or citation without a separately validated gate.
"""
from __future__ import annotations

import math

import numpy as np

from scripts.diagnose_retrieval_v2 import candidate_union


def select_candidates(ranked: list[dict], top_n: int = 5) -> list[dict]:
    """Keep exactly the diagnosed top-N union; RRF orders the candidates.

    No relevance labels are read here. The RRF constant is a generic conventional
    value, not selected by optimizing the 18-query pilot.
    """
    ids = candidate_union(ranked, top_n)
    selected = []
    seen = set()
    for row in ranked:
        ident = row['campaign_id']
        if ident not in ids:
            continue
        if ident in seen:
            raise ValueError(f'Duplicate campaign in vehicle scope: {ident}')
        seen.add(ident)
        s_rank = int(row['semantic_rank'])
        l_rank = int(row['lexical_rank'])
        if s_rank < 1 or l_rank < 1:
            raise ValueError('Ranks must start at 1')
        rrf = 1.0 / (60 + s_rank)
        if float(row['tfidf']) > 0:
            rrf += 1.0 / (60 + l_rank)
        selected.append({**row, 'rrf_score': rrf})
    selected.sort(key=lambda r: (-r['rrf_score'], r['campaign_id']))
    if {r['campaign_id'] for r in selected} != ids:
        raise ValueError('Missing campaign in candidate union')
    return selected


def cross_score_candidates(query: str, candidates: list[dict], docs_by_id: dict[str, dict],
                           model, batch_size: int = 8) -> list[dict]:
    """Reorder a bounded vehicle-scoped pool with pretrained query/passage scores.

    These logits are uncalibrated ranking values. Do not turn this top-k into
    final citations without testing abstention on separate labeled data.
    """
    if not candidates:
        return []
    if not query.strip() or batch_size < 1:
        raise ValueError('Nonempty query and positive batch_size required')
    texts = []
    for r in candidates:
        ident = r['campaign_id']
        doc = docs_by_id[ident]
        passage = str(doc.get('text') or '').strip()
        if not passage:
            raise ValueError(f'Empty recall text: {ident}')
        texts.append(passage[:3500])
    values = np.asarray(model.predict(
        [(query, text) for text in texts],
        batch_size=batch_size, show_progress_bar=False,
    ), dtype=np.float64).reshape(-1)
    if len(values) != len(candidates) or not np.isfinite(values).all():
        raise ValueError('CrossEncoder returned invalid or non-finite scores')
    ordered = [{**row, 'cross_score': float(value)}
               for row, value in zip(candidates, values)]
    ordered.sort(key=lambda r: (-r['cross_score'], -r['rrf_score'], r['campaign_id']))
    return ordered


def ranking_metrics(cases: list[dict], gold: dict[str, list[str]],
                    rankings: dict[str, list[str]], top_k: int = 3) -> dict:
    """Measure UNGATED top-k rankings; no-match selections are false positives.

    This metric is intentionally not a deployable retriever metric because it
    has no abstention stage, and every eligible query gets top-k candidates.
    """
    if top_k < 1:
        raise ValueError('top_k must be positive')
    recalls, positive, negative, negative_selected = [], 0, 0, 0
    matched, retrieved, returned = 0, 0, 0
    for c in cases:
        ident = c['id']
        targets = set(gold.get(ident, []))
        predicted = rankings[ident][:top_k]
        if len(predicted) != len(set(predicted)):
            raise ValueError(f'Duplicate predicted campaign in {ident}')
        if targets:
            positive += 1
            hits = len(targets & set(predicted))
            recalls.append(hits / len(targets))
            matched += int(hits > 0)
        else:
            negative += 1
            negative_selected += int(bool(predicted))
        retrieved += len(targets & set(predicted))
        returned += len(predicted)
    return {
        'positive_cases': positive, 'no_match_cases': negative,
        'mean_recall_at_k': sum(recalls) / len(recalls) if recalls else None,
        'hit_rate_at_k': matched / positive if positive else None,
        'precision_of_returned': retrieved / returned if returned else None,
        'no_match_false_positive_rate_without_abstention': negative_selected / negative if negative else None,
        'retrieved_relevant': retrieved, 'returned_total': returned,
        'top_k': top_k,
        'warning': 'Ranking-only diagnostic; no abstention. NOT safe to cite these candidates.'
    }
