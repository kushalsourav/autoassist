"""Create a versioned recall corpus + persistent FAISS index, WITHOUT changing the API.

Run from repository root:
    python -m scripts.build_recall_corpus_v2
    python -m scripts.search_recall_corpus_v2 --make Honda --model Accord --year 2018 --query 'wipers and rear camera fail'

The NHTSA fetch adapter currently caps each make/model/year response at 25 records.
All vehicle membership is *model-year listing*, never VIN applicability.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VEHICLES = ROOT / 'eval' / 'vehicles_recall_v2.json'
DEFAULT_OUTPUT = ROOT / 'ml' / 'artifacts' / 'recall_corpus_v2'
_CAMPAIGN = re.compile(r'^\d{2}[A-Z]\d{6}$')


def normalize_vehicle(raw: dict) -> dict:
    make = ' '.join(str(raw.get('make') or '').split())
    model = ' '.join(str(raw.get('model') or '').split())
    year = raw.get('year')
    if not make or not model or not isinstance(year, int) or isinstance(year, bool) or not (1990 <= year <= 2100):
        raise ValueError(f'Invalid vehicle {raw!r}')
    return {'make': make, 'model': model, 'year': year}


def vehicle_key(v: dict) -> tuple[str, str, int]:
    return (v['make'].casefold(), v['model'].casefold(), v['year'])


def normalize_campaign(raw: dict) -> dict | None:
    ident = str(raw.get('id') or '').strip().upper()
    if not _CAMPAIGN.fullmatch(ident):
        return None
    text = ' '.join(str(raw.get('text') or '').split())[:4000]
    if not text:
        return None
    return {
        'id': ident,
        'component': str(raw.get('component') or '')[:300],
        'text': text,
        'url': str(raw.get('url') or ''),
        'source_type': 'nhtsa_recalls_by_vehicle',
    }


def merge_campaigns(batch: list[tuple[dict, list[dict]]]) -> list[dict]:
    """One row per campaign, with explicitly observed vehicle-query memberships."""
    merged: dict[str, dict] = {}
    for raw_vehicle, records in batch:
        vehicle = normalize_vehicle(raw_vehicle)
        for raw in records:
            doc = normalize_campaign(raw)
            if doc is None:
                continue
            ident = doc['id']
            if ident not in merged:
                merged[ident] = {**doc, 'vehicles': []}
            current = merged[ident]
            if len(doc['text']) > len(current['text']):
                current.update({key: doc[key] for key in ('component', 'text', 'url')})
            if vehicle_key(vehicle) not in {vehicle_key(v) for v in current['vehicles']}:
                current['vehicles'].append(vehicle)
    result = sorted(merged.values(), key=lambda doc: doc['id'])
    for doc in result:
        doc['vehicles'].sort(key=vehicle_key)
        doc['content_sha256'] = hashlib.sha256(doc['text'].encode()).hexdigest()
    return result


def vehicle_documents(documents: list[dict], make: str, model: str, year: int) -> list[dict]:
    """Apply vehicle metadata before retrieval; never rely on FAISS similarity for eligibility."""
    requested = vehicle_key(normalize_vehicle({'make': make, 'model': model, 'year': year}))
    return [d for d in documents if requested in {vehicle_key(v) for v in d['vehicles']}]


async def collect(vehicles: list[dict], delay_seconds: float) -> tuple[list[dict], list[dict]]:
    from agent.nhtsa import fetch_recalls
    batch = []
    statuses = []
    for i, vehicle in enumerate(vehicles, 1):
        records, status = await fetch_recalls(**vehicle)
        statuses.append({**vehicle, 'status': status, 'records_received': len(records),
                         'possible_truncation': len(records) >= 25})
        print(f'[{i}/{len(vehicles)}] {vehicle["year"]} {vehicle["make"]} {vehicle["model"]}: '
              f'{status}, {len(records)} records')
        if status == 'ok':
            batch.append((vehicle, records))
        if i < len(vehicles) and delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
    return merge_campaigns(batch), statuses


def write_corpus(output: Path, documents: list[dict], statuses: list[dict],
                 vehicle_file: Path, write_index: bool = True) -> dict:
    if not documents:
        raise RuntimeError('No recall documents collected; refusing to save an empty index')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'documents.jsonl').write_text(
        ''.join(json.dumps(doc, ensure_ascii=False) + '\n' for doc in documents),
        encoding='utf-8',
    )
    manifest = {
        'version': 'v2',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'vehicle_inventory': str(vehicle_file),
        'vehicle_queries': len(statuses),
        'successful_queries': sum(v['status'] == 'ok' for v in statuses),
        'failed_queries': sum(v['status'] != 'ok' for v in statuses),
        'possibly_truncated_queries': sum(v['possible_truncation'] for v in statuses),
        'unique_campaigns': len(documents),
        'total_listed_vehicle_memberships': sum(len(d['vehicles']) for d in documents),
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2' if write_index else None,
        'limitations': [
            'Make/model/year listings do not verify VIN eligibility or repair status.',
            'The current NHTSA fetch adapter retains at most 25 records per vehicle query.',
            'Some NHTSA requests may fail; check failed_queries before using the corpus.',
            'No independently annotated relevance judgments exist for this expanded corpus.',
            'Recall records are not a diagnosis or a repair manual.',
        ],
        'queries': statuses,
    }
    if write_index:
        import faiss
        import numpy as np
        from agent.rag import _embedding_model
        model = _embedding_model()
        text = [d['text'][:3000] for d in documents]
        vectors = np.ascontiguousarray(np.asarray(model.encode(
            text, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=True), dtype=np.float32))
        if vectors.ndim != 2 or vectors.shape[0] != len(documents):
            raise RuntimeError('Unexpected embedding matrix dimensions')
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(output / 'index.faiss'))
        manifest['embedding_dimension'] = int(vectors.shape[1])
        manifest['indexed_documents'] = int(index.ntotal)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Build an exploratory multi-vehicle NHTSA recall corpus')
    parser.add_argument('--vehicles', type=Path, default=DEFAULT_VEHICLES)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--delay', type=float, default=0.35, help='Pause between API requests')
    parser.add_argument('--no-index', action='store_true', help='Write corpus without loading MiniLM/FAISS')
    args = parser.parse_args()
    if args.delay < 0:
        parser.error('--delay cannot be negative')
    inventory = json.loads(args.vehicles.read_text(encoding='utf-8'))
    vehicles = [normalize_vehicle(v) for v in inventory['vehicles']]
    if not vehicles or len({vehicle_key(v) for v in vehicles}) != len(vehicles):
        parser.error('Vehicle inventory is empty or contains duplicate make/model/year keys')
    documents, statuses = asyncio.run(collect(vehicles, args.delay))
    manifest = write_corpus(args.out, documents, statuses, args.vehicles, not args.no_index)
    print(f'\nUnique campaigns={manifest["unique_campaigns"]}; '
          f'vehicle memberships={manifest["total_listed_vehicle_memberships"]}; '
          f'failed vehicle queries={manifest["failed_queries"]}')
    print(f'Saved corpus to {args.out}')
    if manifest['failed_queries'] or manifest['possibly_truncated_queries']:
        print('WARNING: Corpus is incomplete; inspect manifest.json before use.')
    print('This corpus contains model-year-listed recalls, NOT VIN eligibility.')


if __name__ == '__main__':
    main()
