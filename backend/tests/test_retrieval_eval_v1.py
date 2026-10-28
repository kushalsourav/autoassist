"""Offline benchmark correctness tests; no model weights, network or FAISS required."""
from pathlib import Path
import copy
import pytest

from scripts.evaluate_retrieval_v1 import (
    DEFAULT_DATA, load_benchmark, score_prediction, summarize, save_reports,
    evaluate, METHODS,
)


def test_fixture_validates_all_gold_ids():
    data = load_benchmark(DEFAULT_DATA)
    assert len(data['documents']) == 6
    assert len(data['cases']) == 30
    assert sum(bool(c['relevant_ids']) for c in data['cases']) == 22
    assert sum(not c['relevant_ids'] for c in data['cases']) == 8
    assert all(c['relevant_ids'] or c['id'].startswith('negative_') for c in data['cases'])


def test_gold_does_not_accidentally_mark_all_engine_power_loss_queries_negative():
    data = load_benchmark(DEFAULT_DATA)
    case = next(c for c in data['cases'] if c['id'] == 'fuel_loss')
    assert set(case['relevant_ids']) == {'20V314000', '23V858000'}


def test_scoring_positive_and_no_match_cases():
    row = score_prediction(['A','B'], ['B', 'C', 'A'], 3)
    assert row['hits'] == 2
    assert row['recall_at_k'] == 1
    assert row['precision_at_k'] == pytest.approx(2 / 3)
    assert row['exact_set_match'] == 0
    negative = score_prediction([], [], 3)
    assert negative['no_match_correct'] == 1
    assert negative['recall_at_k'] is None
    assert negative['exact_set_match'] == 1
    false_positive = score_prediction([], ['A'], 3)
    assert false_positive['no_match_correct'] == 0


def test_summary_counts_no_match_false_positive_rate():
    pos = score_prediction(['A'], ['A'], 3)
    neg = score_prediction([], ['B'], 3)
    scores = summarize([pos, neg], 3)
    assert scores['mean_recall_at_3'] == 1
    assert scores['no_match_specificity'] == 0
    assert scores['false_positive_rate_on_no_match'] == 1
    assert scores['retrieved_total_campaigns'] == 2


def test_benchmark_rejects_unknown_gold(tmp_path):
    import json
    d = load_benchmark(DEFAULT_DATA)
    d['cases'][0]['relevant_ids'] = ['FAKE-RECALL']
    f = tmp_path / 'bad.json'
    f.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='Invalid gold'):
        load_benchmark(f)


def test_benchmark_never_uses_gold_ids_to_retrieve(monkeypatch):
    from scripts import evaluate_retrieval_v1 as evaluation
    data = load_benchmark(DEFAULT_DATA)
    sample = copy.deepcopy(data)
    sample['cases'] = [sample['cases'][0], sample['cases'][-1]]
    seen = []
    def fake_run(query, docs, embedder, classification, top_k, threshold):
        seen.append((query, len(docs), classification))
        # For a hand-picked positive, return the expected ID without ever reading gold.
        predicted = [docs[0]] if 'windshield wipers' in query else []
        return {method: predicted for method in METHODS}
    monkeypatch.setattr(evaluation, 'run_methods', fake_run)
    metrics, rows = evaluate(sample, embedder=object(), mode='none', top_k=3, threshold=0.35)
    assert len(seen) == 2 and all(item[2] is None for item in seen)
    assert len(rows) == 6
    assert metrics['semantic_faiss']['mean_recall_at_3'] == 1


def test_reports_are_written_read_only_from_synthetic_rows(tmp_path):
    data = load_benchmark(DEFAULT_DATA)
    rows = [{'case_id': 'x', 'method': 'semantic_faiss', 'query': 'Q', 'gold_ids':['A'],
             'predicted_ids':['A'], 'classification':'not_used', 'scores':[0.5],
             **score_prediction(['A'], ['A'], 3)}]
    save_reports(tmp_path, data, {'semantic_faiss': summarize([
        score_prediction(['A'], ['A'], 3), score_prediction([], [], 3)], 3)},
        rows, 3, 0.35, 'none')
    assert (tmp_path / 'metrics.json').exists()
    assert (tmp_path / 'cases.csv').exists()
    assert (tmp_path / 'failures.csv').exists()
