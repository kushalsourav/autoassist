"""Offline test: no live NHTSA, MiniLM downloads, or GPU required."""
import json
from pathlib import Path

import numpy as np
import pytest

from agent import corpus_retrieval
from scripts.build_recall_corpus_v2 import merge_campaigns

HONDA = {'make': 'Honda', 'model': 'Accord', 'year': 2018}
FAIL = {'make': 'Hyundai', 'model': 'Elantra', 'year': 2018}
BCM = {'id': '20V771000', 'component': 'ELECTRICAL SYSTEM',
       'text': 'Windshield wipers fail and rearview camera stops working because of a BCM software error.',
       'url': 'https://api.nhtsa.gov/recalls/recallsByVehicle'}
FUEL = {'id': '20V314000', 'component': 'FUEL SYSTEM, GASOLINE',
        'text': 'Fuel pump failure may cause engine stalls and loss of drive power.',
        'url': BCM['url']}
AIRBAG = {'id': '26V332000', 'component': 'AIR BAGS',
          'text': 'Passenger airbag occupant classification warning sensor defect.',
          'url': BCM['url']}


class FakeIndex:
    def __init__(self, docs):
        self.ntotal = len(docs)
        self.d = 2
        vectors = {
            '20V771000': [1.0, 0.0],
            '20V314000': [0.0, 1.0],
            '26V332000': [0.0, -1.0],
        }
        self.vectors = np.array([vectors[d['id']] for d in docs], dtype=np.float32)

    def reconstruct(self, i):
        return self.vectors[i]


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        assert len(texts) == 1
        return np.array([[1.0, 0.0]], dtype=np.float32)


@pytest.fixture
def corpus(monkeypatch):
    docs = merge_campaigns([(HONDA, [BCM, FUEL, AIRBAG])])
    manifest = {'version': 'v2', 'indexed_documents': len(docs),
                'embedding_dimension': 2, 'queries': [
                    {**HONDA, 'status': 'ok', 'possible_truncation': False},
                    {**FAIL, 'status': 'unavailable', 'possible_truncation': False}]}
    index = FakeIndex(docs)
    monkeypatch.setattr(corpus_retrieval, '_load_corpus', lambda path: (docs, manifest, index))
    return docs, manifest


def ml_cat(cat):
    return {'method': 'finetuned_multilabel_distilbert_v3',
            'labels': [{'category': cat, 'score': 0.9}]}


def test_exact_snapshot_allows_relevant_retrieval(corpus):
    result, reason = corpus_retrieval.retrieve_cached(
        'windshield wipers and rearview camera fail', HONDA,
        [BCM, FUEL, AIRBAG], 'ok', '/ignored',
        classification=ml_cat('Electrical'), threshold=0.35,
        embedder=FakeEmbedder())
    assert reason == 'snapshot_verified'
    assert [r['id'] for r in result] == ['20V771000']
    assert result[0]['retrieval_backend'] == 'persistent_faiss_hybrid'


def test_failed_vehicle_falls_back_even_if_index_exists(corpus):
    result, reason = corpus_retrieval.retrieve_cached('wipers fail', FAIL, [], 'unavailable', '/ignored')
    assert result is None
    assert reason == 'live_status_unavailable'


def test_missing_vehicle_snapshot_falls_back(corpus):
    vehicle = {'make': 'Toyota', 'model': 'Camry', 'year': 2018}
    result, reason = corpus_retrieval.retrieve_cached('wipers fail', vehicle, [], 'ok', '/ignored')
    assert result is None
    assert reason == 'vehicle_not_completely_indexed'


def test_live_campaigns_changed_falls_back(corpus):
    result, reason = corpus_retrieval.retrieve_cached('wipers fail', HONDA, [BCM, AIRBAG], 'ok', '/ignored')
    assert result is None
    assert reason == 'campaign_membership_changed'


def test_text_changes_fall_back(corpus):
    changed = {**BCM, 'text': BCM['text'] + ' new manufacturer remedy'}
    result, reason = corpus_retrieval.retrieve_cached('wipers fail', HONDA, [changed, FUEL, AIRBAG], 'ok', '/ignored')
    assert result is None
    assert reason == 'recall_text_changed'


def test_classification_not_required_for_literal_symptom_evidence(corpus):
    result, _ = corpus_retrieval.retrieve_cached('windshield wipers and rearview camera fail',
        HONDA, [BCM, FUEL, AIRBAG], 'ok', '/ignored',
        classification=ml_cat('Engine / Fuel'), embedder=FakeEmbedder())
    assert [r['id'] for r in result] == ['20V771000']


def test_no_fabricated_results_on_zero_live_and_zero_indexed(corpus):
    result, reason = corpus_retrieval.retrieve_cached('my car has a ghost issue',
        HONDA, [], 'ok', '/ignored', embedder=FakeEmbedder())
    assert result is None and reason == 'campaign_membership_changed'


def test_patcher_idempotent(tmp_path, monkeypatch):
    from scripts import patch_corpus_workflow
    config = tmp_path / 'backend' / 'app' / 'config.py'
    workflow = tmp_path / 'agent' / 'workflow.py'
    config.parent.mkdir(parents=True)
    workflow.parent.mkdir(parents=True)
    conf_src = Path(__file__).resolve().parents[2] / 'backend' / 'app' / 'config.py'
    flow_src = Path(__file__).resolve().parents[2] / 'agent' / 'workflow.py'
    config.write_text(conf_src.read_text())
    workflow.write_text(flow_src.read_text())
    monkeypatch.setattr(patch_corpus_workflow, 'ROOT', tmp_path)
    patch_corpus_workflow.patch()
    once = (config.read_text(), workflow.read_text())
    patch_corpus_workflow.patch()
    assert once == (config.read_text(), workflow.read_text())
    assert 'retrieve_cached(' in workflow.read_text()
    assert 'rag_corpus_mode: str' in config.read_text()
