"""Offline validation of blind recall relevance preparation and scoring safeguards."""
import csv
from pathlib import Path
import pytest
from scripts.prepare_multivehicle_eval_v2 import COLUMNS, build_rows, write_preparation
from scripts.evaluate_multivehicle_eval_v2 import read_reviewed_labels


def docs():
    return [
        {'id':'20V771000','component':'ELECTRICAL SYSTEM','url':'https://nhtsa.gov',
         'text':'The BCM can interrupt wipers and the backup camera.',
         'vehicles':[{'make':'Honda','model':'Accord','year':2018}]},
        {'id':'18V629000','component':'BACK OVER PREVENTION','url':'https://nhtsa.gov',
         'text':'The backup camera sometimes fails.',
         'vehicles':[{'make':'Honda','model':'Accord','year':2018}]},
        {'id':'20V314000','component':'FUEL SYSTEM','url':'https://nhtsa.gov',
         'text':'Fuel pump may stall engine.',
         'vehicles':[{'make':'Toyota','model':'Camry','year':2018}]}
    ]


def cases():
    return [{'id':'wipers','make':'Honda','model':'Accord','year':2018,
             'query':'My wipers are not working.'},
            {'id':'stalls','make':'Toyota','model':'Camry','year':2018,
             'query':'The engine dies.'}]


def write_sheet(path, rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_review_sheet_includes_every_vehicle_campaign_and_not_other_vehicles():
    rows=build_rows(docs(),cases())
    assert {(r['case_id'],r['campaign_id']) for r in rows} == {
        ('wipers','18V629000'),('wipers','20V771000'),('stalls','20V314000')}
    assert all(r['relevant']=='' for r in rows)


def test_incomplete_labels_are_refused(tmp_path):
    rows=build_rows(docs(),cases())
    sheet=tmp_path/'judgments.csv'
    write_sheet(sheet,rows)
    with pytest.raises(ValueError,match='Unreviewed'):
        read_reviewed_labels(sheet,rows)


def test_reviewed_labels_only_and_no_match(tmp_path):
    rows=build_rows(docs(),cases())
    for r in rows:
        r['relevant']='1' if r['campaign_id']=='20V771000' else '0'
    sheet=tmp_path/'judgments.csv'
    write_sheet(sheet,rows)
    gold=read_reviewed_labels(sheet,build_rows(docs(),cases()))
    assert gold=={'wipers':['20V771000']}


def test_modified_campaign_id_rejected(tmp_path):
    rows=build_rows(docs(),cases())
    for r in rows: r['relevant']='0'
    rows[0]['campaign_id']='23V858000'
    sheet=tmp_path/'judgments.csv'
    write_sheet(sheet,rows)
    with pytest.raises(ValueError,match='Unexpected'):
        read_reviewed_labels(sheet,build_rows(docs(),cases()))


def test_modified_recall_evidence_rejected(tmp_path):
    rows=build_rows(docs(),cases())
    for r in rows: r['relevant']='0'
    rows[0]['recall_text']='not what the recall says'
    sheet=tmp_path/'judgments.csv'
    write_sheet(sheet,rows)
    with pytest.raises(ValueError,match='Original review sheet'):
        read_reviewed_labels(sheet,build_rows(docs(),cases()))


def test_missing_row_rejected(tmp_path):
    rows=build_rows(docs(),cases())[:-1]
    for r in rows: r['relevant']='0'
    sheet=tmp_path/'judgments.csv'
    write_sheet(sheet,rows)
    with pytest.raises(ValueError,match='count differs'):
        read_reviewed_labels(sheet,build_rows(docs(),cases()))


def test_frozen_query_embedding_cache(monkeypatch):
    import numpy as np
    from scripts.evaluate_multivehicle_eval_v2 import VectorReuseEmbedder

    class Index:
        d=2
        def reconstruct(self,i):
            return [[1.,0.],[0.,1.]][i]

    class Model:
        def encode(self,texts,**kwargs):
            assert texts==['a symptom','another symptom']
            return np.array([[0.,1.],[1.,0.]],dtype=np.float32)

    input_docs = [
        {'text':'Recall A'}, {'text':'Recall B'},
    ]
    input_cases = [{'query':'a symptom'},{'query':'another symptom'}]
    emb=VectorReuseEmbedder(input_docs,Index(),input_cases,Model())
    assert emb.encode(['Recall A','another symptom']).tolist()==[[1.,0.],[1.,0.]]
    with pytest.raises(ValueError,match='not included'):
        emb.encode(['unseen source'])
