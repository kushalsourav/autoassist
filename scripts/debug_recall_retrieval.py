"""Debug a model-year recall false negative, read-only. Run from project root:

python -m scripts.debug_recall_retrieval --make Honda --model Accord --year 2018 \
  --category Electrical \
  --message 'My windshield wipers stop and the rearview camera shows a black screen.'

This reports raw recall count, component gate decisions, embedding similarities,
and the campaigns selected by the live RAG code. No LLM calls are made.
"""
import argparse
import asyncio

from agent.nhtsa import fetch_recalls
from agent.rag import _component_filter, _faiss_retrieve, _matching_terms, retrieve
from backend.app.config import settings


async def main():
    p = argparse.ArgumentParser(description='Diagnose AutoAssist recall retrieval')
    p.add_argument('--make', default='Honda')
    p.add_argument('--model', default='Accord')
    p.add_argument('--year', type=int, default=2018)
    p.add_argument('--category', default='Electrical')
    p.add_argument('--message', default=('My windshield wipers intermittently stop working and '
                                         'the rearview camera sometimes shows a black screen.'))
    args = p.parse_args()
    classification = {
        'method': 'finetuned_multilabel_distilbert_v3',
        'labels': [{'category': args.category, 'score': 0.9}],
    }
    records, status = await fetch_recalls(make=args.make, model=args.model, year=args.year)
    print(f'NHTSA status={status}; raw records={len(records)}; distinct campaigns='
          f'{len(set(str(d.get("id")) for d in records))}')
    if not records:
        print('No raw records to rank. Check NHTSA response and exact vehicle inputs.')
        return
    print(f'Configured backend={settings.rag_backend}; FAISS min similarity={settings.rag_min_similarity}')
    raw_scores = _faiss_retrieve(args.message, records, top_k=len(records), threshold=-2)
    for item in raw_scores:
        metadata_match = bool(_component_filter(args.message, [item], classification))
        terms = _matching_terms(args.message, item)
        explicit_id = bool(item.get('id') and str(item['id']).upper() in args.message.upper())
        selected = explicit_id or (item['score'] >= settings.rag_min_similarity and metadata_match) or len(terms) >= 2
        print(f'id={item.get("id")}, cosine={item["score"]:.4f}, '
              f'metadata_match={metadata_match}, lexical={terms}, candidate={selected}, '
              f'component={item.get("component")}')
    chosen = retrieve(args.message, records, classification=classification,
                      backend='faiss', min_similarity=settings.rag_min_similarity)
    print('SELECTED:', [d.get('id') for d in chosen])
    if not chosen:
        print('No recall passed retrieval. This is NOT proof no applicable recalls exist.')


if __name__ == '__main__':
    asyncio.run(main())
