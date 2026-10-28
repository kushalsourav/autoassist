"""Offline retrieval benchmark using externally reviewed case/campaign relevance.

No NHTSA requests, OpenRouter calls, or Gold label use inside retrievers.
DO NOT run this as a benchmark until all review rows have been adjudicated.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from agent import rag
from agent.corpus_retrieval import _load_corpus, retrieve_cached
from backend.app.config import settings
from scripts.build_recall_corpus_v2 import vehicle_key, normalize_vehicle
from scripts.evaluate_retrieval_v1 import score_prediction, summarize, classification_for_query
from scripts.prepare_multivehicle_eval_v2 import (
    DEFAULT_CORPUS, DEFAULT_OUT, DEFAULT_QUERIES, COLUMNS,
    build_rows, file_sha256, load_inputs,
)

METHODS = ('semantic_faiss', 'lexical_tfidf', 'hybrid_live', 'hybrid_persistent')


def read_reviewed_labels(path: Path, expected_rows: list[dict]) -> dict[str,list[str]]:
    if not path.exists():
        raise FileNotFoundError(f'Missing review sheet: {path}. Run prepare_multivehicle_eval_v2 first.')
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError('Review CSV columns changed. Preserve the original sheet and its column order.')
        rows = list(reader)
    expected = {(r['case_id'], r['campaign_id']):r for r in expected_rows}
    if len(rows) != len(expected):
        raise ValueError(f'Review row count differs ({len(rows)} vs {len(expected)}). Do not delete rows.')
    seen, labels = set(), {}
    for row in rows:
        key = (row['case_id'],row['campaign_id'])
        if key not in expected or key in seen:
            raise ValueError(f'Unexpected or duplicated campaign/case judgment: {key!r}')
        seen.add(key)
        exp = expected[key]
        for field in ('case_id','make','model','year','query','campaign_id','component','source_url','recall_text'):
            if str(row[field]) != str(exp[field]):
                raise ValueError(f'Original review sheet field modified: {field} at {key}')
        value = (row.get('relevant') or '').strip()
        if value not in ('1','0'):
            raise ValueError(f'Unreviewed or invalid label at {key}: {value!r}; use only 1 or 0')
        if value == '1':
            labels.setdefault(row['case_id'],[]).append(row['campaign_id'])
    if seen != set(expected):
        raise ValueError('Review sheet is incomplete')
    return labels


def verify_snapshot(corpus:Path, queries:Path, out:Path):
    review_file = out/'review_manifest.json'
    if not review_file.exists():
        raise FileNotFoundError(f'Missing {review_file}; run preparation first')
    review = json.loads(review_file.read_text(encoding='utf-8'))
    paths = {
        'corpus_documents_sha256':corpus/'documents.jsonl',
        'corpus_manifest_sha256':corpus/'manifest.json',
        'corpus_index_sha256':corpus/'index.faiss',
        'cases_sha256':queries,
    }
    for name,path in paths.items():
        if file_sha256(path) != review.get('hashes',{}).get(name):
            raise ValueError(f'{name}: corpus or queries changed since the review sheet was prepared')
    return review


class VectorReuseEmbedder:
    """Use exact saved document vectors, encode benchmark queries once."""
    def __init__(self, docs:list[dict], index, cases:list[dict], model):
        self.vectors = {}
        for i,doc in enumerate(docs):
            self.vectors[doc['text'][:3000]] = np.asarray(index.reconstruct(i),dtype=np.float32)
        questions = list(dict.fromkeys(c['query'] for c in cases))
        query_vectors = np.asarray(model.encode(questions, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
        if query_vectors.shape != (len(questions),index.d):
            raise ValueError('Unexpected query embedding dimensions')
        for question,vec in zip(questions,query_vectors):
            self.vectors[question] = vec

    def encode(self, texts, **kwargs):
        try:
            return np.stack([self.vectors[t] for t in texts]).astype(np.float32)
        except KeyError as exc:
            raise ValueError(f'Text not included in frozen evaluation embedding cache: {exc}') from exc


def compare_methods(query, vehicle, docs, full_docs, status, corpus, classification,
                    top_k, threshold, embedder):
    semantic = rag._faiss_retrieve(query, docs, top_k=top_k,
                                   threshold=threshold, embedder=embedder)
    lexical = rag._tfidf_retrieve(query, docs, top_k=top_k)
    hybrid = rag.retrieve(query, docs, top_k=top_k, classification=classification,
                          backend='faiss', min_similarity=threshold, embedder=embedder)
    persistent, reason = retrieve_cached(query,vehicle,docs,status,corpus,
                            classification=classification, top_k=top_k,
                            threshold=threshold, embedder=embedder)
    if persistent is None:
        raise ValueError(f'Persistent offline replay unavailable: {reason}')
    return {
        'semantic_faiss':semantic, 'lexical_tfidf':lexical,
        'hybrid_live':hybrid, 'hybrid_persistent':persistent,
    }


def evaluate(cases, docs, gold, embedder, corpus_path, top_k, threshold, mode):
    by_method = {key:[] for key in METHODS}
    all_rows = []
    for case in cases:
        v = normalize_vehicle(case)
        key = vehicle_key(v)
        eligible = [doc for doc in docs if key in {vehicle_key(x) for x in doc['vehicles']}]
        result = compare_methods(case['query'],v,eligible,docs,'ok',corpus_path,
                      classification_for_query(case['query'],mode),top_k,threshold,embedder)
        for name in METHODS:
            predicted = [d['id'] for d in result[name]]
            stats = score_prediction(gold.get(case['id'],[]),predicted,top_k)
            row = {'case_id':case['id'], 'make':v['make'],'model':v['model'],'year':v['year'],
                   'query':case['query'],'method':name,'gold_ids':gold.get(case['id'],[]),
                   'predicted_ids':predicted,
                   'scores':[item.get('score') for item in result[name]],**stats}
            by_method[name].append(row)
            all_rows.append(row)
    metrics = {key:summarize(rows,top_k) for key,rows in by_method.items()}
    discrepancies = [c['id'] for c in cases if next(r['predicted_ids'] for r in by_method['hybrid_live']
                                                    if r['case_id']==c['id']) != next(r['predicted_ids']
                                                    for r in by_method['hybrid_persistent'] if r['case_id']==c['id'])]
    return metrics, all_rows, discrepancies


def save_results(out:Path, metrics:dict, rows:list[dict], disagreements:list[str],
                 manifest:dict, top_k:int, threshold:float, mode:str):
    results = out / ('results_v3' if mode == 'v3' else 'results_no_classifier')
    results.mkdir(parents=True,exist_ok=True)
    report = {
        'version':'multivehicle-eval-v2', 'classification_mode':mode,
        'top_k':top_k, 'similarity_threshold':threshold,
        'reviewed_cases':manifest['case_count'],'reviewed_campaign_judgments':manifest['candidate_judgments'],
        'methods':metrics,'hybrid_live_vs_persistent_disagree_case_ids':disagreements,
        'limitations':[
            'All campaign relevance labels were supplied manually, not generated by retrieval.',
            'New queries are hand-authored; review may not be independent or expert-adjudicated.',
            'The fixed NHTSA corpus snapshot can omit relevant campaigns from other vehicle trims/years or unavailable API lookups.',
            'Offline replay validates ranking given frozen data, NOT freshness of live NHTSA responses.',
            'Listed recalls do not confirm VIN eligibility, unrepaired status, or diagnosis.',
            'No end-to-end LLM grounding or latency is measured.',
        ]}
    (results/'metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for name,data in [('cases.csv',rows),('failures.csv',[r for r in rows if not r['exact_set_match']])]:
        with (results/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            for row in data:
                writer.writerow({key:json.dumps(value) if isinstance(value,list) else value
                                 for key,value in row.items()})
    return results


def main():
    p=argparse.ArgumentParser(description='Score independently reviewed multi-vehicle recall relevance')
    p.add_argument('--corpus',type=Path,default=DEFAULT_CORPUS)
    p.add_argument('--queries',type=Path,default=DEFAULT_QUERIES)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--classification',choices=['none','v3'],default='none')
    p.add_argument('--top-k',type=int,default=3)
    p.add_argument('--min-similarity',type=float,default=0.35)
    args=p.parse_args()
    if args.top_k < 1 or not -1.0 <= args.min_similarity <= 1.0:
        p.error('top-k must be positive and similarity must be between -1 and 1')
    meta = verify_snapshot(args.corpus,args.queries,args.out)
    _,docs,cases=load_inputs(args.corpus,args.queries)
    gold=read_reviewed_labels(args.out/'judgments.csv',build_rows(docs,cases))
    pos=sum(bool(gold.get(c['id'])) for c in cases)
    neg=len(cases)-pos
    if pos == 0 or neg == 0:
        raise ValueError('Benchmark needs both relevant and no-match cases. Review labels and cases before proceeding.')
    print(f'Reviewed {len(cases)} queries: {pos} positive, {neg} no-match; '
          f'{len(build_rows(docs,cases))} judgments. No NHTSA or LLM calls.')
    stored_docs,_,index = _load_corpus(str(args.corpus.resolve()))
    if [d['id'] for d in docs] != [d['id'] for d in stored_docs]:
        raise ValueError('Corpus index document order differs from snapshot')
    embedder=VectorReuseEmbedder(docs,index,cases,rag._embedding_model())
    metrics,rows,disagreements=evaluate(cases,docs,gold,embedder,args.corpus,args.top_k,
                                      args.min_similarity,args.classification)
    print(f'\n{"Method":22} {f"Recall@{args.top_k}":>9} {"Precision of returned":>22} {"No-match FP":>13}')
    for name,report in metrics.items():
        precision = report['precision_of_returned']
        precision_display = f'{precision:.3f}' if precision is not None else 'n/a'
        print(f'{name:22} {report[f"mean_recall_at_{args.top_k}"]:>9.3f} '
              f'{precision_display:>22} '
              f'{report["false_positive_rate_on_no_match"]:>13.3f}')
    print('Hybrid live/persistent disagreements:',len(disagreements))
    output=save_results(args.out,metrics,rows,disagreements,meta,args.top_k,
                        args.min_similarity,args.classification)
    print('Reports:',output)
    print('These are manually adjudicated SNAPSHOT metrics, not real-world or VIN eligibility accuracy.')


if __name__=='__main__':
    main()
