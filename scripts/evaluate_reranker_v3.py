"""Offline two-stage ranking diagnostic, NOT production retrieval.

Usage:
  python -m scripts.evaluate_reranker_v3
  python -m scripts.evaluate_reranker_v3 --skip-cross-encoder

Uses saved V2 corpus embeddings + 18 assistant-reviewed queries; no NHTSA,
OpenRouter or classifier calls. The first run downloads an optional pretrained
MS MARCO cross-encoder. Ranking-only outputs DO NOT abstain or justify citations.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from agent.corpus_retrieval import _load_corpus
from agent.rag import _embedding_model
from agent.rerank_experiment import select_candidates, cross_score_candidates, ranking_metrics
from scripts.build_recall_corpus_v2 import normalize_vehicle, vehicle_key
from scripts.diagnose_retrieval_v2 import rank_all
from scripts.evaluate_multivehicle_eval_v2 import read_reviewed_labels, verify_snapshot
from scripts.prepare_multivehicle_eval_v2 import (
    DEFAULT_CORPUS, DEFAULT_QUERIES, DEFAULT_OUT, build_rows, load_inputs
)

DEFAULT_MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'


def run(cases, docs, index, embedding_model, gold, crossencoder=None, top_n=5,
        top_k=3, batch_size=8):
    """Reproducible offline replay; testable with fake models and FAISS index."""
    if top_n < 1 or top_k < 1:
        raise ValueError('top_n and top_k must be positive')
    qstrings = list(dict.fromkeys(str(c['query']) for c in cases))
    vectors = np.asarray(embedding_model.encode(
        qstrings, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False), dtype=np.float32)
    if vectors.shape != (len(qstrings), index.d) or not np.isfinite(vectors).all():
        raise ValueError('Bad query embeddings or embedding dimension mismatch')
    vector_by_query = dict(zip(qstrings, vectors))
    doc_vectors = np.stack([np.asarray(index.reconstruct(i), dtype=np.float32)
                            for i in range(len(docs))])
    if doc_vectors.shape != (len(docs), index.d) or not np.isfinite(doc_vectors).all():
        raise ValueError('Saved FAISS embeddings do not match corpus')
    rrfs, reranked, output, case_rows, timings = {}, {}, [], [], []
    for case in cases:
        vehicle = vehicle_key(normalize_vehicle(case))
        idx = [i for i, d in enumerate(docs) if vehicle in {vehicle_key(v) for v in d['vehicles']}]
        scoped = [docs[i] for i in idx]
        ranked = rank_all(case['query'], scoped, doc_vectors[idx], vector_by_query[case['query']])
        candidates = select_candidates(ranked, top_n)
        scoped_by_id = {d['id']: d for d in scoped}
        rrfs[case['id']] = [r['campaign_id'] for r in candidates]
        if crossencoder is not None:
            start = perf_counter()
            ordered = cross_score_candidates(case['query'], candidates, scoped_by_id,
                                             crossencoder, batch_size=batch_size)
            timings.append((perf_counter()-start)*1000)
            reranked[case['id']] = [r['campaign_id'] for r in ordered]
            cross_scores = {r['campaign_id']: r['cross_score'] for r in ordered}
        else:
            cross_scores = {}
        gold_ids = set(gold.get(case['id'], []))
        available = {d['id'] for d in scoped}
        if not gold_ids <= available:
            raise ValueError(f'Gold contains campaign outside vehicle scope: {case["id"]}')
        case_rows.append({
            'case_id':case['id'],'vehicle':f"{case['year']} {case['make']} {case['model']}",
            'query':case['query'], 'gold_ids':json.dumps(sorted(gold_ids)),
            'rrf_ids':json.dumps(rrfs[case['id']][:top_k]),
            'cross_encoder_ids':json.dumps(reranked.get(case['id'], [])[:top_k]),
            'missing_from_candidates':json.dumps(sorted(gold_ids-set(rrfs[case['id']]))),
            'note':'UNGATED ranking candidates, not approved citations'
        })
        for r in candidates:
            rid = r['campaign_id']
            output.append({
                'case_id':case['id'], 'query':case['query'],
                'vehicle':f"{case['year']} {case['make']} {case['model']}",
                'campaign_id':rid, 'component':r['component'],
                'gold_relevant_for_audit_only':int(rid in gold_ids),
                'rrf_rank':rrfs[case['id']].index(rid)+1,
                'rrf_score':round(r['rrf_score'],7),
                'semantic_rank':r['semantic_rank'],
                'lexical_rank':r['lexical_rank'],
                'cosine':r['cosine'], 'tfidf':r['tfidf'],
                'cross_encoder_rank':(reranked[case['id']].index(rid)+1 if crossencoder is not None else ''),
                'cross_encoder_score':round(cross_scores[rid],5) if crossencoder is not None else '',
            })
    metrics = {'candidate_recall_at_pool':ranking_metrics(cases,gold,rrfs,top_k=2*top_n),
               'rrf_without_abstention':ranking_metrics(cases,gold,rrfs,top_k=top_k)}
    if crossencoder is not None:
        metrics['cross_encoder_without_abstention'] = ranking_metrics(cases,gold,reranked,top_k=top_k)
        metrics['cross_encoder_reranking_ms_per_case'] = {
            'median':round(float(np.median(timings)),3),
            'mean':round(float(np.mean(timings)),3),
        }
    return metrics, output, case_rows


def save(out:Path, metrics:dict, ranking_rows:list[dict], case_rows:list[dict], args):
    out.mkdir(parents=True,exist_ok=True)
    report = {
        'version':'reranker-v3-offline-diagnostic',
        'corpus':'v2 snapshot, precomputed MiniLM vectors',
        'model':None if args.skip_cross_encoder else args.model,
        'candidate_pool_per_method':args.top_n, 'top_k':args.top_k,
        'metrics':metrics,
        'limitations':[
            'Zero-shot pretrained cross-encoder reranks candidates only; no abstention implemented.',
            'No-match FP rate is intentionally high for ungated ranking and is not production performance.',
            'Gold labels are used only for retrospective scoring, never for ranking/candidate inclusion.',
            'This 18-case benchmark and its failures have been inspected repeatedly; NOT a held-out test.',
            'The 18 cases include a query with four gold campaigns; Recall@3 has a mathematical ceiling.',
            'Assistant-reviewed relevance judgments are not independent expert adjudication.',
            'The snapshot is incomplete and is not VIN-specific, diagnostic, or repair-status evidence.',
            'Do not integrate into FastAPI before validation of abstention on new cases.',
        ]}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for filename, rows in [('ranked_candidates.csv', ranking_rows), ('cases.csv',case_rows)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,default=DEFAULT_CORPUS)
    p.add_argument('--queries',type=Path,default=DEFAULT_QUERIES)
    p.add_argument('--eval-dir',type=Path,default=DEFAULT_OUT)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--top-n',type=int,default=5)
    p.add_argument('--top-k',type=int,default=3)
    p.add_argument('--model',default=DEFAULT_MODEL)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--skip-cross-encoder',action='store_true')
    args=p.parse_args()
    if args.top_n < 1 or args.top_k < 1 or args.batch_size < 1:
        p.error('top-n, top-k and batch-size must be positive')
    verify_snapshot(args.corpus,args.queries,args.eval_dir)
    _, docs, cases=load_inputs(args.corpus,args.queries)
    gold=read_reviewed_labels(args.eval_dir/'judgments.csv',build_rows(docs,cases))
    frozen, _, index=_load_corpus(str(args.corpus.resolve()))
    if [d['id'] for d in docs] != [d['id'] for d in frozen]:
        raise ValueError('Saved FAISS document order does not match reviewed corpus')
    model=None
    if not args.skip_cross_encoder:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError('Install backend/requirements-rag.txt for Sentence Transformers') from exc
        from torch import nn
        print(f'Loading CrossEncoder on CPU: {args.model}')
        model=CrossEncoder(args.model, device='cpu', activation_fn=nn.Identity())
    print('Replaying vehicle-scoped candidate union; NO NHTSA or OpenRouter calls.')
    metrics, rows, cases_report = run(cases, docs, index, _embedding_model(), gold, model,
                                      args.top_n,args.top_k,args.batch_size)
    print(f"{'Ranking method':28} {'Recall@3':>10} {'Returned precision':>20} {'No-match FP':>13}")
    for label,key in [('RRF (no abstention)','rrf_without_abstention'),
                      ('CrossEncoder (no abstention)','cross_encoder_without_abstention')]:
        if key not in metrics:
            continue
        m=metrics[key]
        print(f"{label:28} {m['mean_recall_at_k']:>10.3f} "
              f"{m['precision_of_returned']:>20.3f} "
              f"{m['no_match_false_positive_rate_without_abstention']:>13.3f}")
    m=metrics['candidate_recall_at_pool']
    total_gold = sum(len(set(gold.get(c['id'], []))) for c in cases)
    print(f'Candidate pool relevant campaign selections: {m["retrieved_relevant"]}/{total_gold}')
    location=save(args.out or args.eval_dir/'reranker_v3',metrics,rows,cases_report,args)
    print('Reports:',location)
    print('WARNING: UNGATED top-k = ranking only. Do not feed these candidates to the LLM.')
    print('Next: develop and calibrate abstention on NEW reviewed query/campaign pairs.')

if __name__=='__main__':
    main()
