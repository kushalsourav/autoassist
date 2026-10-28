"""Produce a *blind* review sheet containing EVERY listed campaign for each case.

Does not use a retrieval model, suggest gold labels, or write performance metrics.
No VIN eligibility, repair status, or diagnosis is implied by campaign membership.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.build_recall_corpus_v2 import normalize_vehicle, vehicle_key

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / 'ml/artifacts/recall_corpus_v2'
DEFAULT_QUERIES = ROOT / 'eval/multivehicle_queries_v2.json'
DEFAULT_OUT = ROOT / 'ml/artifacts/multivehicle_eval_v2'
COLUMNS = ('case_id','make','model','year','query','campaign_id','component',
           'source_url','recall_text','relevant','reviewer_note')


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(corpus: Path, queries: Path):
    manifest_path = corpus / 'manifest.json'
    docs_path = corpus / 'documents.jsonl'
    index_path = corpus / 'index.faiss'
    for f in (manifest_path, docs_path, index_path, queries):
        if not f.is_file():
            raise FileNotFoundError(f'Required file missing: {f}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    docs = [json.loads(line) for line in docs_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    questions = json.loads(queries.read_text(encoding='utf-8'))['cases']
    if manifest.get('version') != 'v2' or manifest.get('indexed_documents') != len(docs):
        raise ValueError('Expected indexed V2 corpus with matching document count')
    if not docs or not questions or len({d['id'] for d in docs}) != len(docs):
        raise ValueError('Empty or duplicate campaign document IDs / cases')
    if len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Duplicate case IDs')
    statuses = {vehicle_key(normalize_vehicle(s)): s for s in manifest['queries']}
    for q in questions:
        if not str(q.get('query','')).strip() or not str(q.get('id','')).strip():
            raise ValueError('Case ID and query must be nonempty')
        key = vehicle_key(normalize_vehicle(q))
        entry = statuses.get(key)
        if not entry or entry['status'] != 'ok' or entry.get('possible_truncation'):
            raise ValueError(f'Query {q["id"]}: vehicle lookup not verified complete; do not label as no match')
        if not any(key in {vehicle_key(v) for v in d['vehicles']} for d in docs):
            raise ValueError(f'Query {q["id"]}: successful vehicle has no indexed campaigns')
    return manifest, docs, questions


def build_rows(docs: list[dict], cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        key = vehicle_key(normalize_vehicle(case))
        candidates = [d for d in docs if key in {vehicle_key(v) for v in d['vehicles']}]
        for doc in sorted(candidates, key=lambda d: d['id']):
            rows.append({
                'case_id': case['id'], 'make': case['make'], 'model': case['model'],
                'year': case['year'], 'query': case['query'], 'campaign_id': doc['id'],
                'component': doc['component'], 'source_url': doc.get('url',''),
                'recall_text': doc['text'], 'relevant': '', 'reviewer_note': ''
            })
    return rows


def write_preparation(out: Path, rows: list[dict], manifest: dict, corpus: Path, queries: Path):
    out.mkdir(parents=True, exist_ok=True)
    sheet = out / 'judgments.csv'
    if sheet.exists():
        raise FileExistsError(f'{sheet} already exists. Refusing to overwrite manual labels.')
    with sheet.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    hashes = {
        'corpus_documents_sha256': file_sha256(corpus/'documents.jsonl'),
        'corpus_manifest_sha256': file_sha256(corpus/'manifest.json'),
        'corpus_index_sha256': file_sha256(corpus/'index.faiss'),
        'cases_sha256': file_sha256(queries),
    }
    meta = {'version':'multivehicle-eval-v2','status':'awaiting_manual_review',
            'case_count':len(set(row['case_id'] for row in rows)),
            'candidate_judgments':len(rows), 'corpus_unique_campaigns':manifest['unique_campaigns'],
            'corpus_failed_vehicle_queries': manifest['failed_queries'],
            'hashes':hashes,
            'limitations':['Hand-review each row against the campaign narrative; 1=relevant, 0=not relevant.',
              'Empty cells are unreviewed, not negative judgments.',
              'Manufacturer/model/year membership is NOT VIN-specific applicability or repair status.',
              'Source is an incomplete 67-campaign snapshot; do not claim a wider recall search.',
              'Queries were drafted for this evaluation; labels must be reviewed independently of retriever outputs.']}
    (out/'review_manifest.json').write_text(json.dumps(meta, indent=2)+'\n',encoding='utf-8')
    return sheet


def main():
    p = argparse.ArgumentParser(description='Prepare blinded multi-vehicle recall relevance judgments')
    p.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    p.add_argument('--queries', type=Path, default=DEFAULT_QUERIES)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    manifest, docs, questions = load_inputs(args.corpus, args.queries)
    rows = build_rows(docs, questions)
    sheet = write_preparation(args.out, rows, manifest, args.corpus, args.queries)
    print(f'Prepared {len(questions)} queries, {len(rows)} case/campaign review rows across '
          f'{len({(q["make"],q["model"],q["year"]) for q in questions})} vehicles.')
    print('Review every "relevant" cell in:', sheet)
    print('Enter 1 (relevant) or 0 (not relevant). Do not infer no recalls from unavailable vehicle requests.')
    print('Scoring intentionally refuses incomplete judgments.')


if __name__ == '__main__':
    main()
