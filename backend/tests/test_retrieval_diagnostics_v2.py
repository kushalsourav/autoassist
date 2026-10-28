"""Unit tests using synthetic frozen vectors; never download models or hit NHTSA."""
import numpy as np
import pytest
from scripts.diagnose_retrieval_v2 import rank_all, candidate_union, audit


class FakeIndex:
    d = 2
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=np.float32)
    def reconstruct(self, i):
        return self.vectors[i]


class FakeModel:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=np.float32)
    def encode(self, questions, **kwargs):
        assert len(questions) == len(self.vectors)
        return self.vectors


VEHICLE = {'make': 'Honda', 'model': 'Accord', 'year': 2018}


def test_candidate_union_rescues_low_cosine_tfidf_candidate():
    docs = [
        {'id': 'RELEVANT', 'text': 'Fuel pump might make the engine stall and lose power', 'component': 'FUEL'},
        {'id': 'UNRELATED', 'text': 'Camera display turns black during reversing', 'component': 'CAMERA'},
    ]
    ranked = rank_all('Engine stall and loss of power', docs,
                      np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
                      np.array([1.0, 0.0], dtype=np.float32))
    assert ranked[0]['campaign_id'] == 'RELEVANT'
    assert ranked[0]['tfidf'] > ranked[1]['tfidf']
    assert {'RELEVANT', 'UNRELATED'} == candidate_union(ranked, 1)
    assert ranked[0]['cosine'] == 0.0


def test_audit_flags_failure_caused_by_lexical_cosine_gates():
    query = 'Engine stalls suddenly'
    doc = {'id': 'FUEL', 'text': 'A broken fuel pump might cause unexpected failure',
           'component': 'FUEL', 'vehicles': [VEHICLE]}
    rows, report = audit([{'id': 'loss', **VEHICLE, 'query': query}], [doc],
                         FakeIndex([[0.0, 1.0]]), FakeModel([[1.0, 0.0]]),
                         {'loss': ['FUEL']}, {'loss': []}, top_n=1, threshold=0.35)
    assert rows[0]['failure_reason'] == 'below_both_gates'
    assert report[0]['missed_by_hybrid'] == ['FUEL']
    assert report[0]['missed_by_candidate_stage'] == []


def test_no_match_candidates_are_not_final_citations():
    query = 'My navigation screen is dim'
    doc = {'id': 'FUEL', 'text': 'Fuel pump failure causes stall',
           'component': 'FUEL', 'vehicles': [VEHICLE]}
    rows, report = audit([{'id': 'none', **VEHICLE, 'query': query}], [doc],
                         FakeIndex([[1.0, 0.0]]), FakeModel([[1.0, 0.0]]),
                         {}, {'none': []}, top_n=1, threshold=0.35)
    assert rows[0]['in_candidate_union'] == 1
    assert report[0]['candidate_ids'] == ['FUEL']
    assert not report[0]['positive']


def test_top_n_validation():
    with pytest.raises(ValueError):
        candidate_union([], 0)
