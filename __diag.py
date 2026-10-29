from backend.app.config import settings
import pathlib, json

print("rag_backend:", settings.rag_backend)
print("rag_corpus_mode:", settings.rag_corpus_mode)
print("rag_min_similarity:", settings.rag_min_similarity)
print("rag_corpus_dir:", settings.rag_corpus_dir)

corpus = pathlib.Path(settings.rag_corpus_dir)
print("corpus exists:", corpus.exists())
if corpus.exists():
    for f in corpus.iterdir():
        print(f"  {f.name}  {f.stat().st_size} bytes")
    manifest_path = corpus / "manifest.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        print("\nManifest summary:")
        print("  version:", m.get("version"))
        print("  indexed_documents:", m.get("indexed_documents"))
        print("  embedding_dimension:", m.get("embedding_dimension"))
        queries = m.get("queries", [])
        print(f"  queries ({len(queries)} vehicles):")
        for q in queries[:10]:
            print(f"    {q}")
else:
    print("  CORPUS DIRECTORY MISSING")

# Also check what the live NHTSA recall docs look like
print("\nChecking live recall fetch for 2008 Honda Accord...")
import asyncio
from agent.nhtsa import fetch_recalls
records, status = asyncio.run(fetch_recalls("Honda", "Accord", 2008))
print(f"  status={status}, count={len(records)}")
if records:
    for r in records[:3]:
        keys = list(r.keys())
        print(f"  record keys: {keys}")
        print(f"  id={r.get('id')}, component={r.get('component','')[:60]}")
        print(f"  text[:120]={str(r.get('text',''))[:120]}")

