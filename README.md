# AutoAssist — Evidence-Grounded Automotive AI Engineer

> **Status: milestone 1 scaffold (not a completed ML portfolio result).**
> A decision-support prototype for classifying *reported* vehicle complaints and retrieving official US-market NHTSA recall evidence. It does **not** identify confirmed mechanical faults or determine recall eligibility for a specific VIN.

## One monorepo

```text
autoassist/
├── frontend/                  # Next.js App Router + TypeScript; deploy to Netlify
├── backend/                   # FastAPI REST endpoints and chat persistence; CPU deploy
├── ml/                        # NHTSA ingestion, baselines, fine-tuning, metrics, artifacts
├── agent/                     # LangGraph, NHTSA tool, TF-IDF RAG, OpenRouter adapter
├── supabase/schema.sql        # Chat tables, owner-scoped RLS policies
├── .env.example
└── README.md
```

## Current features and what is still planned

**Milestone 1 provided:**
- Next.js App Router frontend for vehicle make/model/year, complaints, responses, retrieved recall citations, and saved chats.
- FastAPI endpoints: `/api/health`, `/api/metrics`, `/api/analyze`, `/api/chats`, `/api/chats/{id}/messages`.
- LangGraph pipeline: classify -> NHTSA recalls -> lexical TF-IDF retrieval -> optional LLM or transparent evidence-only fallback.
- Local SQLite chat storage or Supabase Postgres using anonymous Supabase Auth and row-level security.
- Data collection script and benchmark script for real NHTSA complaint data; DistilBERT fine-tuning script for local GPU.
- Local model inference via joblib baseline or optional DistilBERT CPU inference.

**Not implemented or verified yet:** retrieval with FAISS/vector embeddings, trained weights, measured accuracy/latency, full RAG benchmark, MLflow tracking, quantized ONNX deployment, public Netlify/Railway deployment, screenshot/image multimodal analysis. **Do not use resume claims for those yet.**

## Run locally (Python 3.11 recommended)

From the repo root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate ; Linux/macOS: source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env   # Windows PowerShell: Copy-Item .env.example .env
# STORAGE_MODE=local by default: works WITHOUT Supabase, model weights, or LLM API key.
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal (Node.js 20.9+ is required):

```bash
cd frontend
npm install
cp .env.example .env.local  # Windows: Copy-Item .env.example .env.local
npm run dev
```

Open `http://localhost:3000`. API documentation: `http://localhost:8000/docs`.
The default classifier is a **keyword-only demo** until you train a model. Without OpenRouter credentials, responses are deterministic evidence-only text, not LLM output. NHTSA internet access is needed to retrieve actual recalls.

## Supabase Free (recommended for the public demo)

1. Create a Supabase project and run `supabase/schema.sql` in its SQL Editor.
2. Under **Authentication > Providers**, enable **Anonymous Sign-Ins**. Configure CAPTCHA/Turnstile and consider rate limits before public launch; anonymous user creation can be abused.
3. In root `.env`, set `STORAGE_MODE=supabase`, `SUPABASE_URL`, `SUPABASE_ANON_KEY` and the deployed Netlify URL in `CORS_ORIGINS`.
4. In `frontend/.env.local`, set `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` to **the same project**, plus `NEXT_PUBLIC_API_URL` to your backend.
5. Restart backend and frontend; changing `NEXT_PUBLIC_*` values on Netlify requires a rebuild. Users now get an anonymous auth session; Supabase RLS ensures chats are accessible to their owner only. Clearing browser storage/signing out loses access to an anonymous account.

**Security:** only use the public/anon key in the frontend. Never expose a `service_role` key or OpenRouter key in browser code. The backend uses the *user's JWT* for Supabase REST calls, and validates it before LLM calls. Local SQLite mode using `X-Demo-User` is **not authenticated** and must never be used for a public deployment.

Supabase Free allows a small database but can pause inactive projects. The FAQ and current quotas should be checked before applying to jobs. Chat metadata is a JSONB field for storing classification/evidence results. Do not store vehicle VINs or sensitive user data in this prototype.

## OpenRouter / Claude Haiku (optional)

In root `.env`:

```env
OPENROUTER_API_KEY=your_secret_key
OPENROUTER_MODEL=provider/model-slug
```

Choose an actual available model slug in your OpenRouter account; free models, quotas and provider availability can change. Your OpenRouter key remains on the FastAPI server. Failed or unconfigured model calls return an **explicitly labeled** evidence-only response. For public URLs, add deployment-layer request quotas and spending limits: anonymous auth by itself will not prevent API bill abuse.

## Real ML training on your GTX 1650 (4 GB VRAM, 16 GB RAM)

```bash
pip install -r ml/requirements-train.txt
python -m ml.prepare_nhtsa
python -m ml.train_baseline
```

The ingestion script retrieves real US-market NHTSA owner complaints and derives **noisy/weak training labels** from reported component categories; labels are **not mechanically verified faults**. `ml/artifacts/metrics.json` contains the baseline results from the held-out split, which you must inspect and verify. Files/artifacts are gitignored. The backend automatically loads `ml/artifacts/baseline.joblib` on the next startup in `MODEL_VARIANT=baseline` mode.

For transformer experimentation, install an appropriate official PyTorch CUDA build for your machine (refer to pytorch.org), plus:

```bash
pip install 'transformers>=4.46,<5' 'datasets>=3,<5' 'accelerate>=1,<2'
python -m ml.train_transformer --batch-size 2 --gradient-accumulation 8 --max-length 192 --epochs 2
```

The DistilBERT script saves weights locally to `ml/artifacts/distilbert/`. To serve that instead, set `MODEL_VARIANT=distilbert` and install PyTorch/Transformers in your backend environment/container. It will require substantially more CPU RAM; measure before deploying on a low-RAM free service. Baseline and transformer tests use the same initial seeded holdout; neither guarantees a metric improvement. Expand and audit dataset samples before using results in your resume.

## Docker — FastAPI + agent in one container

```bash
docker build -f backend/Dockerfile -t autoassist-api .
docker run --rm -p 8000:8000 --env-file .env autoassist-api
```

For cloud deployment use `STORAGE_MODE=supabase`; the default local SQLite path in a container is ephemeral. Backend instances need enough RAM for scikit-learn plus your selected ML model. If you select DistilBERT, update the container dependencies to include PyTorch and Transformers. Backend health check: `/api/health`.

## Deployment plan

- **Netlify:** base directory `frontend`, build `npm run build`, publish `.next` (Netlify OpenNext adapter); environment variables `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`.
- **Backend:** Railway / Cloud Run with root `backend/Dockerfile` built from repo root; env vars `STORAGE_MODE=supabase`, Supabase credentials, `CORS_ORIGINS` matching the Netlify domain, optional OpenRouter credentials.
- **ML model:** run inference inside the same backend (start with lightweight scikit-learn baseline; deploy DistilBERT after RAM/latency tests). No separate GPU service needed for inference.
- **Live demo:** publish working example inputs and a report of measured metrics; never present precomputed output as a live response.

## Next milestone checklist

1. Run NHTSA ingestion, inspect class balance/labels, deduplicate and identify data leakage.
2. Train and evaluate baseline; add reproducible model comparison (confusion matrix, macro-F1 by class).
3. Fine-tune DistilBERT on your local GTX 1650; compare on the exact held-out data and measure inference RAM/latency.
4. Add vector embeddings + FAISS in `agent/rag.py`, create a retrieval relevance dataset, measure Recall@K and grounded citation precision.
5. Add MLflow, safety cases, rate limits, proper incident logging, and cloud deployment. Record only metrics you actually measured.

## Known limitations

- NHTSA API is US-market oriented. A matching make/model/year does **not** confirm recall eligibility for a VIN. A missing recall result does not indicate a vehicle is safe.
- Vehicle complaints are owner-reported; categorizing them is not diagnosis. Human inspection remains essential.
- A public anonymous-demo service requires abuse prevention. Backend lacks production-grade distributed rate limiting.
- SQLite is only for local development. Supabase free projects may pause with inactivity. No backups are bundled.
- Agent RAG currently uses TF-IDF lexical similarity, not FAISS embeddings. Untrained fallback is transparently labeled.

## Phase 1 update — OpenRouter errors, intent routing, and conversational history

This release fixes three starter limitations:

- **Error visibility:** If OpenRouter fails, the API returns `generation_mode=llm_unavailable`
  and a **sanitized** `generation_error` such as HTTP 429 or timeout. The Next.js
  UI shows the error. The fallback answer labels itself **Non-LLM fallback**.
  The OpenRouter key and raw provider error body are not displayed.
- **Simple routing:** A transparent, keyword-based `agent/intent.py` guard prevents
  obviously off-topic prompts (e.g., tigers) from launching a vehicle recall query.
  Unclear car-count questions ask for clarification. This is **not** a trained
  intent model and cannot perfectly recognize every out-of-scope input.
- **Chat history:** For an existing conversation, the last eight stored user/assistant
  messages are now available to the LLM. A new conversation does not inherit them.
  Saved chat histories display the backend mode and any provider error.

**After replacing your repo, keep your existing root `.env` and
`frontend/.env.local`; they are deliberately NOT bundled in the update.**

Add this optionally to your root `.env`:

```env
OPENROUTER_TIMEOUT_SECONDS=90
```

The application still uses the keyword-only complaint classifier until a real
model is trained. RAG still uses lexical TF-IDF; embedding-based FAISS is planned.
No live OpenRouter credentials were used while verifying the changes: tests use
mock HTTP responses and verify success and 429 paths. Run with your own key to
verify the chosen Nemotron model's behavior.

Phase 1-specific tests:

```bash
python -m pytest -q backend/tests/test_phase1.py
```
