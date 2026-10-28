"""CPU-only deterministic reranker tests; no network, no model downloads."""
import json
import numpy as np
import pytest

from agent.rerank_experiment import select_candidates, cross_score_candidates, ranking_metrics
from scripts.evaluate_reranker_v3 import run

VEHICLE={'make':'Honda','model':'Accord','year':2018}

class FakeIndex:
    d=2
    def __init__(self,vectors): self.vectors=np.asarray(vectors,dtype=np.float32)
    def reconstruct(self,i): return self.vectors[i]

class FakeEmbedder:
    def encode(self,texts,**kwargs):
        return np.array([[1.,0.] for _ in texts],dtype=np.float32)

class FakeCrossEncoder:
    def __init__(self, scoring): self.scoring=scoring; self.pairs=[]
    def predict(self,pairs,**kwargs):
        self.pairs.extend(pairs)
        return np.array([self.scoring[text] for _,text in pairs],dtype=np.float32)


def ranks():
    return [
        {'campaign_id':'A','semantic_rank':1,'lexical_rank':2,'tfidf':0.2,'cosine':.4,'component':'X'},
        {'campaign_id':'B','semantic_rank':2,'lexical_rank':1,'tfidf':0.5,'cosine':.3,'component':'Y'},
        {'campaign_id':'C','semantic_rank':3,'lexical_rank':3,'tfidf':0.,'cosine':.1,'component':'Z'},
    ]


def test_candidate_union_preserves_diagnostic_definition():
    got=select_candidates(ranks(),top_n=1)
    assert {r['campaign_id'] for r in got}=={'A','B'}
    assert got[0]['campaign_id']=='A'  # symmetric RRF ranks tie; campaign ID breaks ties


def test_cross_encoder_can_rerank_without_using_gold():
    records=select_candidates(ranks(),top_n=2)
    fake=FakeCrossEncoder({'A text':3.0,'B text':-2.0})
    scored=cross_score_candidates('Symptoms',records,
                                  {'A':{'text':'A text'},'B':{'text':'B text'}},fake)
    assert [r['campaign_id'] for r in scored]==['A','B']
    assert len(fake.pairs)==2
    assert all(q=='Symptoms' for q,_ in fake.pairs)


def test_negative_queries_are_not_silently_treated_as_abstentions():
    cases=[{'id':'positive'},{'id':'negative'}]
    gold={'positive':['A']}
    predictions={'positive':['A'],'negative':['X']}
    m=ranking_metrics(cases,gold,predictions,top_k=3)
    assert m['mean_recall_at_k']==1
    assert m['no_match_false_positive_rate_without_abstention']==1
    assert 'NOT safe' in m['warning']


def test_empty_negative_is_true_abstention_but_not_guaranteed_by_ranking():
    m=ranking_metrics([{'id':'n'}],{}, {'n':[]})
    assert m['no_match_false_positive_rate_without_abstention']==0
    assert m['mean_recall_at_k'] is None


def test_rerank_rejects_invalid_model_scores():
    class Bad(FakeCrossEncoder):
        def predict(self,pairs,**kwargs): return [float('nan') for _ in pairs]
    candidates=select_candidates(ranks(),top_n=1)
    with pytest.raises(ValueError,match='invalid'):
        cross_score_candidates('hi',candidates,
                {'A':{'text':'A'},'B':{'text':'B'}},Bad({}))


def test_run_restricts_to_vehicle_memberships():
    docs=[
        {'id':'A','text':'Engine stalls due to failing fuel pump', 'component':'FUEL','vehicles':[VEHICLE]},
        {'id':'B','text':'Windshield camera display becomes black','component':'CAMERA','vehicles':[VEHICLE]},
        {'id':'FOREIGN','text':'Engine suddenly stalls', 'component':'FUEL',
         'vehicles':[{'make':'Toyota','model':'Camry','year':2018}]},
    ]
    index=FakeIndex([[1.,0.],[0.,1.],[1.,0.]])
    cases=[{'id':'power',**VEHICLE,'query':'Engine stalls'}]
    m,ranked,per_case=run(cases,docs,index,FakeEmbedder(),{'power':['A']},
                          crossencoder=FakeCrossEncoder({docs[0]['text']:3.0,docs[1]['text']:-2.0}),
                          top_n=2)
    assert all(r['campaign_id']!='FOREIGN' for r in ranked)
    assert json.loads(per_case[0]['cross_encoder_ids'])[0]=='A'
    assert m['cross_encoder_without_abstention']['retrieved_relevant']==1


def test_reject_invalid_rank_bounds():
    with pytest.raises(ValueError):
        select_candidates(ranks(),top_n=0)
    with pytest.raises(ValueError):
        ranking_metrics([{'id':'x'}],{}, {'x':[]},top_k=0)
