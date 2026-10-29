"""Full retrieval trace for 2008 Honda Accord + power loss symptoms."""
import asyncio
from backend.app.config import settings
from agent.nhtsa import fetch_recalls
from agent.rag import retrieve

# 1. Fetch live records
records, status = asyncio.run(fetch_recalls("Honda", "Accord", 2008))
print(f"Live NHTSA: status={status}, {len(records)} records")
for r in records:
    print(f"  {r['id']:12s}  {r['component'][:70]}")

symptoms = "Engine loses power when accelerating, and the check-engine light comes on."
classification = {"category": "Engine / Fuel", "confidence": 0.966,
                  "labels": [{"category": "Engine / Fuel", "score": 0.966}],
                  "method": "finetuned_multilabel_distilbert_v3"}

# 2. Try the cached corpus path
print("\n--- Corpus path ---")
try:
    from agent.corpus_retrieval import retrieve_cached
    cached, reason = retrieve_cached(
        symptoms,
        {"make": "Honda", "model": "Accord", "year": 2008},
        records, status,
        settings.rag_corpus_dir,
        classification=classification,
        threshold=settings.rag_min_similarity
    )
    print(f"corpus reason={reason!r}, results={cached}")
except Exception as e:
    print(f"corpus error: {e}")

# 3. Try the live FAISS path directly
print("\n--- Live FAISS path ---")
from agent.rag import _faiss_hybrid_retrieve, _component_filter, _matching_terms
for doc in records:
    filt = _component_filter(symptoms, [doc], classification)
    terms = _matching_terms(symptoms, doc)
    print(f"  {doc['id']:12s}  filter={bool(filt)}  terms({len(terms)})={terms[:5]}")

results = retrieve(symptoms, records, classification=classification)
print(f"\nretrieve() -> {len(results)} candidates")
for r in results:
    print(f"  score={r.get('score'):.3f}  {r['id']:12s}  {r['component'][:60]}")

# 4. Try with lower threshold
print("\n--- Live FAISS with threshold=0.1 ---")
from agent.rag import _faiss_hybrid_retrieve
results2 = _faiss_hybrid_retrieve(symptoms, records, top_k=5, threshold=0.1, classification=classification)
print(f"  {len(results2)} candidates")
for r in results2:
    print(f"  score={r.get('score'):.3f}  {r['id']:12s}  {r['component'][:60]}")

