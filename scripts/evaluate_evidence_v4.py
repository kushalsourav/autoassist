"""Offline V4 evidence-verifier/abstention diagnostic (not a production switch).

Consumes the already-generated V3 RRF and cross-encoder candidate rankings and
reviewed V2 recall narratives. Makes no NHTSA, embedding-model, or LLM calls.
Reports DIRECT-only and DIRECT-or-MECHANISM evidence policies separately.

IMPORTANT: These 18 cases have been inspected during development. Results are
retrospective and may not be reported as held-out gains or deployed directly.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from evaluation.evidence_verifier_v4 import filter_ranked_evidence
from evaluation.rerank_experiment import ranking_metrics
from scripts.prepare_multivehicle_eval_v2 import DEFAULT_OUT


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def load_review_and_rankings(judgments: Path, ranking: Path):
    """Validate one complete reviewed label per campaign/case before scoring."""
    rows = _read_csv(judgments)
    if not rows:
        raise ValueError('Empty judgments')
    gold: dict[str, list[str]] = defaultdict(list)
    cases: dict[str, dict] = {}
    docs_by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        cid, ident = row['case_id'], row['campaign_id']
        if not cid or not ident or row['relevant'] not in {'0', '1'}:
            raise ValueError(f'Incomplete or invalid judgment {cid}/{ident}')
        if ident in docs_by_case[cid]:
            raise ValueError(f'Duplicate judgment {cid}/{ident}')
        if cid in cases and cases[cid]['query'] != row['query']:
            raise ValueError(f'Conflicting query in {cid}')
        if cid not in cases:
            cases[cid] = {'id': cid, 'query': row['query'],
                          'vehicle': f"{row['year']} {row['make']} {row['model']}"}
        docs_by_case[cid][ident] = {'id':ident, 'text':row['recall_text'],
                                    'component':row['component'], 'url':row.get('source_url','')}
        if row['relevant'] == '1':
            gold[cid].append(ident)

    ranked = defaultdict(list)
    for row in _read_csv(ranking):
        cid, ident = row['case_id'], row['campaign_id']
        if cid not in cases or ident not in docs_by_case[cid]:
            raise ValueError(f'Ranking contains non-scoped candidate: {cid}/{ident}')
        if row['query'] != cases[cid]['query'] or row['vehicle'] != cases[cid]['vehicle']:
            raise ValueError(f'Ranking/query/vehicle disagreement: {cid}/{ident}')
        if any(r['campaign_id'] == ident for r in ranked[cid]):
            raise ValueError(f'Duplicate ranked ID: {cid}/{ident}')
        ranked[cid].append(row)
    if set(ranked) != set(cases):
        raise ValueError('Ranked cases do not match reviewed cases')
    # Deliberately NEVER use the audit gold column attached to rankings.
    return list(cases.values()), dict(gold), dict(docs_by_case), dict(ranked)


def evaluate(cases, gold, docs_by_case, ranked_by_case, *, top_k: int = 3):
    """Compare two rankings x two abstention policies without changing the API."""
    if top_k < 1:
        raise ValueError('top_k must be positive')
    predictions = {}
    case_rows, audit_rows = [], []
    for ranking_name, rank_field in [('rrf','rrf_rank'),('cross_encoder','cross_encoder_rank')]:
        for policy_name, allow_mechanism in [('strict_direct',False),('balanced_mechanism',True)]:
            key = f'{ranking_name}_{policy_name}'
            preds = {}
            for c in cases:
                cid = c['id']
                entries = ranked_by_case[cid]
                if ranking_name == 'cross_encoder' and any(not r.get(rank_field) for r in entries):
                    raise ValueError('Cross-encoder scores missing; run reranker V3 first')
                ordered = sorted(entries, key=lambda r: int(r[rank_field]))
                selected, decisions = filter_ranked_evidence(
                    c['query'], ordered, docs_by_case[cid], top_k=top_k,
                    allow_mechanism=allow_mechanism)
                predicted = [d['id'] for d in selected]
                preds[cid] = predicted
                target = set(gold.get(cid, []))
                case_rows.append({'method':key, 'case_id':cid, 'vehicle':c['vehicle'],
                                  'query':c['query'],'gold_ids':json.dumps(sorted(target)),
                                  'predicted_ids':json.dumps(predicted),
                                  'missed_ids':json.dumps(sorted(target-set(predicted))),
                                  'extra_ids':json.dumps(sorted(set(predicted)-target)),
                                  'abstained':int(not predicted)})
                for decision in decisions:
                    audit_rows.append({'method':key,'case_id':cid,
                                       'gold_relevant_for_audit_only':int(decision['campaign_id'] in target),
                                       **{k:json.dumps(v) if isinstance(v,list) else v
                                          for k,v in decision.items()}})
            predictions[key] = {'metrics':ranking_metrics(cases,gold,preds,top_k=top_k),
                                'ids':preds}
    return predictions, case_rows, audit_rows


def save(out, predictions, cases, audits, *, top_k):
    out.mkdir(parents=True, exist_ok=True)
    report = {
        'version':'evidence-v4-offline-retrospective', 'top_k':top_k,
        'methods':{key:val['metrics'] for key,val in predictions.items()},
        'limitations':[
            '18 query/vehicle fixtures have been repeatedly inspected: retrospective diagnostic only, NOT a held-out test.',
            '147 relevance judgments were assistant-reviewed, not independently adjudicated by experts.',
            'V4 rules are handcrafted and can miss relevant records or accept unrelated records.',
            'Mechanism tier means plausible component connection, not documented exact symptom or diagnosis.',
            'No-match results are specific to this incomplete 67-campaign snapshot; NOT evidence no recalls exist.',
            'These results do not validate VIN eligibility, unrepaired status, production performance, or LLM answers.',
            'Do NOT enable V4 citations in production without new independently labeled evaluation.',
        ]}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for filename,rows in [('cases.csv',cases),('evidence_audit.csv',audits)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['empty'])
            writer.writeheader()
            writer.writerows(rows)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--judgments',type=Path,default=DEFAULT_OUT/'judgments.csv')
    p.add_argument('--rankings',type=Path,default=DEFAULT_OUT/'reranker_v3'/'ranked_candidates.csv')
    p.add_argument('--out',type=Path,default=DEFAULT_OUT/'evidence_v4')
    p.add_argument('--top-k',type=int,default=3)
    args=p.parse_args()
    cases,gold,docs,ranked=load_review_and_rankings(args.judgments,args.rankings)
    result,rows,audit=evaluate(cases,gold,docs,ranked,top_k=args.top_k)
    print(f'Offline evidence diagnostic: {len(cases)} cases, {len(audit)//4} ranked query/document pairs, top_k={args.top_k}')
    print(f"{'Method':35} {'Recall':>7} {'Precision returned':>18} {'No-match FP':>13}")
    for name,details in result.items():
        m=details['metrics']
        print(f"{name:35} {m['mean_recall_at_k']:.3f} {m['precision_of_returned']:.3f} "
              f"{m['no_match_false_positive_rate_without_abstention']:.3f}")
    location=save(args.out,result,rows,audit,top_k=args.top_k)
    print('Reports:',location)
    print('DO NOT DEPLOY these rules based on the inspected 18-case fixture.')
    print('Next: independent, blinded evaluation of new vehicle-scoped query/campaign pairs.')

if __name__=='__main__':
    main()
