"""Fast retrieval regression tests: no trained weights or live NHTSA needed."""
import numpy as np
from agent import rag


ENGINE = {'id': 'ENGINE_TEST', 'component': 'ENGINE', 'text': 'The engine may stall and lose power while accelerating'}
BRAKE = {'id': 'BRAKE_TEST', 'component': 'SERVICE BRAKES', 'text': 'Brake pedal may fail under braking'}
BCM = {'id': 'BCM_TEST', 'component': 'ELECTRICAL SYSTEM:BODY CONTROL MODULE:SOFTWARE',
       'text': 'Body control software disrupts windshield wipers, exterior lights and camera'}
CLASSIFICATION = {'method': 'finetuned_multilabel_distilbert_v3',
                  'labels': [{'category': 'Engine / Fuel', 'score': 0.97}]}


def test_engine_question_does_not_use_unrelated_bcm():
    assert rag.retrieve('Car loses power under acceleration', [BCM],
                        classification=CLASSIFICATION, backend='faiss') == []


def test_known_engine_component_is_retained():
    assert rag._component_filter('Engine loses power', [ENGINE, BCM, BRAKE],
                                 CLASSIFICATION) == [ENGINE]


def test_explicit_campaign_id_bypasses_gate():
    row = {**BCM, 'id': '20V771000'}
    assert rag._component_filter('Explain recall 20V771000', [row], CLASSIFICATION) == [row]


def test_uncertain_prediction_does_not_filter():
    assert rag._component_filter('Unclear issue', [BCM], {'category': 'Uncertain'}) == [BCM]


def test_empty_docs_never_fabricated():
    assert rag.retrieve('engine stalls', [], backend='faiss') == []


def test_legacy_tfidf_still_available():
    out = rag.retrieve('engine stalling', [ENGINE, BRAKE], backend='tfidf')
    assert out and out[0]['id'] == 'ENGINE_TEST'


def test_cosine_ranking_with_mock_faiss(monkeypatch):
    class IndexFlatIP:
        def __init__(self, dim):
            self.dim = dim
        def add(self, data):
            self.data = data
        def search(self, query, k):
            sims = (self.data @ query[0]).astype(np.float32)
            ids = np.argsort(-sims)[:k]
            return sims[ids][None, :], ids[None, :]
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, 'faiss', SimpleNamespace(IndexFlatIP=IndexFlatIP))
    class FakeEmbedder:
        def encode(self, texts, **kwargs):
            assert kwargs['normalize_embeddings'] is True
            assert len(texts) == 3
            return np.array([[1, 0], [0, 1], [0.8, 0.6]], dtype=np.float32)
    out = rag.retrieve('engine acceleration', [ENGINE, BRAKE], backend='faiss',
                       min_similarity=0.7, embedder=FakeEmbedder())
    assert len(out) == 1
    assert out[0]['id'] == 'ENGINE_TEST'
    assert out[0]['score'] == 0.8
    assert out[0]['retrieval_backend'] == 'faiss_minilm_cosine'
