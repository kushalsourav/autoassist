"""Regression tests for symptom-match recall recovery after hard metadata gating."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from agent import rag


BCM_TEXT = (
    'Honda is recalling certain 2018-2020 Accord vehicles. A software error '
    'may disrupt the Body Control Module and cause the windshield wipers and '
    'defroster, rearview camera, exterior lights and power windows to malfunction.'
)
BCM = {'id': '20V771000', 'component': 'ELECTRICAL SYSTEM:BODY CONTROL MODULE:SOFTWARE',
       'text': BCM_TEXT}
WIPERS = {'id': '20V771000', 'component': 'VISIBILITY/WIPER', 'text': BCM_TEXT}
ENGINE = {'id': 'ENGINE_TEST', 'component': 'ENGINE',
          'text': 'Engine may stall or lose power while accelerating.'}


@pytest.fixture(autouse=True)
def fake_faiss(monkeypatch):
    class IndexFlatIP:
        def __init__(self, dim):
            self.dim = dim
        def add(self, data):
            self.data = data
        def search(self, query, k):
            values = (self.data @ query[0]).astype(np.float32)
            ids = np.argsort(-values)[:k]
            return values[ids][None, :], ids[None, :]
    monkeypatch.setitem(sys.modules, 'faiss', SimpleNamespace(IndexFlatIP=IndexFlatIP))


class FakeEmbedder:
    def __init__(self, doc_vectors, query_vector):
        self.vectors = np.asarray(doc_vectors + [query_vector], dtype=np.float32)
    def encode(self, texts, **kwargs):
        assert len(texts) == len(self.vectors)
        return self.vectors


def classifier(name):
    return {'method': 'finetuned_multilabel_distilbert_v3',
            'labels': [{'category': name, 'score': 0.82}]}


def test_wiper_camera_retrieved_even_if_embedding_score_below_candidate_threshold():
    query = 'My windshield wipers stop working and my rearview camera is black'
    # Cosine 0.20 < 0.35, but two strong terms in NHTSA summary match the symptoms.
    embed = FakeEmbedder([[1.0, 0.0]], [0.2, 0.9798])
    result = rag.retrieve(query, [BCM], backend='faiss', classification=classifier('Electrical'),
                          min_similarity=0.35, embedder=embed)
    assert [r['id'] for r in result] == ['20V771000']


def test_unmapped_nhtsa_component_can_be_recovered_from_recall_narrative():
    query = 'windshield wipers and rearview camera stopped'
    embed = FakeEmbedder([[1.0, 0.0]], [0.1, 0.995])
    assert rag._component_filter(query, [WIPERS], classifier('Electrical')) == [WIPERS]  # unknown retained
    assert [r['id'] for r in rag.retrieve(query, [WIPERS], backend='faiss',
            classification=classifier('Electrical'), min_similarity=0.35,
            embedder=embed)] == ['20V771000']


def test_classifier_mistake_does_not_hide_explicitly_matching_narrative():
    query = 'rearview camera and windshield wipers failed'
    embed = FakeEmbedder([[1.0, 0.0]], [0.1, 0.995])
    assert rag._component_filter(query, [BCM], classifier('Engine / Fuel')) == []
    assert [r['id'] for r in rag.retrieve(query, [BCM], backend='faiss',
            classification=classifier('Engine / Fuel'), min_similarity=0.35,
            embedder=embed)] == ['20V771000']


def test_bcm_does_not_appear_for_engine_power_loss_even_when_cosine_high():
    query = 'My car loses power while accelerating and the check engine light is on.'
    # Semantic score alone is insufficient when a distinct official component
    # contradicts the ML category and no specific symptom words match.
    embed = FakeEmbedder([[1.0, 0.0]], [0.9, 0.4359])
    assert rag.retrieve(query, [BCM], backend='faiss',
            classification=classifier('Engine / Fuel'), min_similarity=0.35,
            embedder=embed) == []


def test_campaign_number_request_can_retrieve_even_with_low_semantic_score():
    embed = FakeEmbedder([[1.0, 0.0]], [0.1, 0.995])
    result = rag.retrieve('Explain 20V771000', [BCM], backend='faiss',
                          classification=classifier('Engine / Fuel'),
                          min_similarity=0.35, embedder=embed)
    assert result[0]['id'] == '20V771000'


def test_no_raw_recall_never_generates_a_source():
    assert rag.retrieve('My windshield wipers fail', [], backend='faiss') == []


def test_deduplicates_same_campaign_from_multiple_nhtsa_component_records():
    query = 'windshield wipers and rearview camera failed'
    embed = FakeEmbedder([[1, 0], [0.9, 0.4359]], [1, 0])
    result = rag.retrieve(query, [BCM, WIPERS], backend='faiss',
            classification=classifier('Electrical'), min_similarity=0.35,
            embedder=embed)
    assert len(result) == 1
    assert result[0]['id'] == '20V771000'
