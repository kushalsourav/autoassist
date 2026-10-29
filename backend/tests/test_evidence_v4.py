"""Offline conservative evidence tests. No model downloads, NHTSA or LLM calls."""
import csv
from pathlib import Path

import pytest

from evaluation.evidence_verifier_v4 import verify_evidence, filter_ranked_evidence
from scripts.evaluate_evidence_v4 import evaluate, load_review_and_rankings


def _doc(text):
    return {'id':'A','text':text,'component':'AUTOMOTIVE'}


def test_explicit_camera_and_wiper_failure():
    query='My windshield wipers stop and the rearview camera goes black'
    d=_doc('Disruptions in BCM may result in malfunctions of windshield wipers and rearview camera.')
    result=verify_evidence(query,d)
    assert result.tier=='direct'
    assert 'wiper_failure' in result.matched_symptoms
    assert 'camera_failure' in result.matched_symptoms
    assert all(s in d['text'] for s in result.supported_by)


def test_camera_display_recalls_direct():
    result=verify_evidence('Backup camera goes black',
                _doc('The back-up camera center display may not function properly.'))
    assert result.tier=='direct'


def test_fuel_pump_is_mechanism_not_diagnosis():
    result=verify_evidence('Engine suddenly loses power when accelerating',
                _doc('The low-pressure fuel pump inside the fuel tank may fail.'))
    assert result.tier=='mechanism'
    assert result.reason=='component_mechanism_only_symptom_not_explicit'


def test_tailgate_does_not_equal_passenger_door():
    assert verify_evidence('My passenger door opens unexpectedly while driving',
            _doc('The tailgate latches may unlatch unexpectedly while driving.')).tier=='abstain'


def test_brightness_is_not_flicker():
    assert verify_evidence('Dashboard and exterior lights flicker while driving',
            _doc('Daytime running lamps may remain fully bright instead of dimming.')).tier=='abstain'


def test_seatbelt_latch():
    assert verify_evidence('Front passenger seat belt buckle will not latch',
            _doc('A defect can prevent the seat belt buckle from latching.')).tier=='direct'


def test_no_narrative_no_fabrication():
    assert verify_evidence('My engine stalls',{'id':'A'}).tier=='abstain'
    assert verify_evidence('',_doc('Engine stalls')).tier=='abstain'


def test_filter_respects_rank_order_and_can_abstain():
    rows=[{'campaign_id':'A'}, {'campaign_id':'B'}]
    docs={'A':_doc('The tailgate may unlatch.'),
          'B':{'id':'B','text':'Passenger door latch may fail and the door may open unexpectedly.'}}
    got,audit=filter_ranked_evidence('Passenger door opens while driving',rows,docs)
    assert [d['id'] for d in got]==['B']
    assert [r['tier'] for r in audit]==['abstain','direct']
    assert all(d['id'] in docs for d in got)
    got,_=filter_ranked_evidence('Air conditioning squeaks',rows,docs)
    assert got==[]


def test_balanced_policy_discloses_mechanism():
    rows=[{'campaign_id':'A'}]
    docs={'A': _doc('The low-pressure fuel pump may fail.')}
    assert filter_ranked_evidence('Engine loses power',rows,docs)[0]==[]
    got,_=filter_ranked_evidence('Engine loses power',rows,docs,allow_mechanism=True)
    assert got[0]['evidence_tier']=='mechanism'


def test_cross_case_scope_and_no_label_leakage():
    cases=[{'id':'p','query':'Backup camera black screen', 'vehicle':'2018 Honda Accord'},
           {'id':'n','query':'Passenger door opens', 'vehicle':'2018 Honda Accord'}]
    docs={'p':{'A':_doc('Back-up camera display may not function properly.')},
          'n':{'B':_doc('Tailgate latches may unlatch unexpectedly while driving.')}}
    ranked={'p':[{'campaign_id':'A','rrf_rank':'1','cross_encoder_rank':'1'}],
            'n':[{'campaign_id':'B','rrf_rank':'1','cross_encoder_rank':'1'}]}
    out,_,_=evaluate(cases,{'p':['A']},docs,ranked)
    assert out['rrf_strict_direct']['ids']=={'p':['A'],'n':[]}
    assert out['cross_encoder_strict_direct']['metrics']['no_match_false_positive_rate_without_abstention']==0
    # gold must NOT alter predictions
    out2,_,_=evaluate(cases,{'p':['B'], 'n':['A']},docs,ranked)
    assert out2['rrf_strict_direct']['ids']==out['rrf_strict_direct']['ids']


def test_duplicates_and_out_of_scope_rejected():
    rows=[{'campaign_id':'A'}, {'campaign_id':'A'}]
    with pytest.raises(ValueError,match='duplicate'):
        filter_ranked_evidence('camera fails',rows,{'A':_doc('Camera fails')})
    with pytest.raises(ValueError,match='not in scoped corpus'):
        filter_ranked_evidence('camera fails',[{'campaign_id':'OTHER'}],{'A':_doc('Camera fails')})


def test_incomplete_judgments_fail_closed(tmp_path):
    j=tmp_path/'judgments.csv'
    r=tmp_path/'ranked.csv'
    with j.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','campaign_id','query','make','model','year','relevant','recall_text','component'])
        w.writeheader(); w.writerow(dict(case_id='c',campaign_id='A',query='my car stalls',make='Honda',model='Accord',year=2018,relevant='',recall_text='engine stalls',component='ENGINE'))
    r.write_text('case_id,campaign_id\nc,A\n')
    with pytest.raises(ValueError,match='Incomplete'):
        load_review_and_rankings(j,r)
