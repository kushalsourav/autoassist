from uuid import UUID
import json
import os
import httpx
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings, ROOT
from backend.app.schemas import AnalyzeRequest, AnalyzeResponse
from backend.app.storage import SQLiteStore, SupabaseStore
from backend.app.agent_analysis import run_analysis
from backend.app.agent_routes import mount_agent

app = FastAPI(title='AutoAssist', version='0.1.0',
              description='Automotive complaint decision-support prototype; not a mechanical diagnosis.')
app.add_middleware(CORSMiddleware, allow_origins=[s.strip() for s in settings.cors_origins.split(',')],
                   allow_credentials=False, allow_methods=['GET', 'POST'],
                   allow_headers=['Authorization', 'Content-Type', 'X-Demo-User'])

if settings.storage_mode not in ('local', 'supabase'):
    raise RuntimeError('STORAGE_MODE must be local or supabase')
if settings.storage_mode == 'supabase' and not (settings.supabase_url and settings.supabase_anon_key):
    raise RuntimeError('Supabase URL and anon key required for supabase mode')
local_store = SQLiteStore(settings.database_file) if settings.storage_mode == 'local' else None


async def current_user(authorization: str | None = Header(default=None),
                       x_demo_user: str | None = Header(default=None)) -> tuple[str, object]:
    if settings.storage_mode == 'local':
        if not x_demo_user:
            raise HTTPException(401, 'X-Demo-User UUID is required in local development')
        try:
            user_id = str(UUID(x_demo_user))
        except ValueError:
            raise HTTPException(401, 'Invalid demo user ID') from None
        return user_id, local_store

    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(401, 'Supabase bearer token required')
    jwt = authorization.removeprefix('Bearer ').strip()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f'{settings.supabase_url.rstrip("/")}/auth/v1/user',
                headers={'apikey': settings.supabase_anon_key, 'Authorization': f'Bearer {jwt}'})
            response.raise_for_status()
            user_id = str(UUID(response.json()['id']))
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise HTTPException(401, 'Invalid or expired Supabase session') from None
    return user_id, SupabaseStore(settings.supabase_url, settings.supabase_anon_key, jwt)


@app.get('/api/health')
def health():
    if settings.model_variant == 'multilabel_distilbert_v3':
        model_dir = settings.multilabel_v3_dir
        status = ('finetuned_multilabel_distilbert_v3'
                  if (model_dir / 'config.json').is_file()
                  and (model_dir / 'metrics.json').is_file()
                  and any((model_dir / name).is_file()
                          for name in ('model.safetensors', 'pytorch_model.bin'))
                  else 'multilabel_v3_artifact_missing')
    elif settings.model_variant == 'distilbert':
        status = ('finetuned_distilbert' if settings.transformer_dir.joinpath('config.json').exists()
                  else 'distilbert_artifact_missing')
    else:
        status = 'trained_tfidf_logreg' if settings.model_file.exists() else 'keyword_demo_only'
    provider = os.getenv('AUTOASSIST_PROVIDER', settings.autoassist_provider or 'anthropic').lower()
    if provider == 'anthropic':
        llm_configured = bool(os.getenv('ANTHROPIC_API_KEY', settings.anthropic_api_key))
    else:
        llm_configured = bool(settings.openrouter_api_key and settings.openrouter_model)
    return {'status': 'ok', 'model': status, 'storage': settings.storage_mode,
            'llm_provider': provider,
            'llm_model': (os.getenv('ANTHROPIC_MODEL', settings.anthropic_model)
                          if provider == 'anthropic' else settings.openrouter_model),
            'llm_configured': llm_configured}


@app.get('/api/metrics')
def metrics():
    path = ROOT / 'ml' / 'artifacts' / 'metrics.json'
    if not path.exists():
        return {'status': 'not_trained'}
    return {'status': 'evaluated', 'metrics': json.loads(path.read_text(encoding='utf-8'))}


@app.get('/api/chats')
async def chats(identity=Depends(current_user)):
    owner, store = identity
    return await store.list_chats(owner)


@app.get('/api/chats/{chat_id}/messages')
async def messages(chat_id: UUID, identity=Depends(current_user)):
    owner, store = identity
    if not await store.get_chat(owner, str(chat_id)):
        raise HTTPException(404, 'Conversation not found')
    return await store.list_messages(owner, str(chat_id))


@app.post('/api/analyze', response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, identity=Depends(current_user)):
    owner, store = identity
    chat_id = str(request.conversation_id) if request.conversation_id else None
    if chat_id and not await store.get_chat(owner, chat_id):
        raise HTTPException(404, 'Conversation not found')

    history = await store.list_messages(owner, chat_id) if chat_id else []
    state = await run_analysis(request, history, owner_id=owner, conversation_id=chat_id or '')
    if not chat_id:
        chat = await store.create_chat(owner, f'{request.year} {request.make} {request.model}: {request.message[:45]}')
        chat_id = chat['id']
    sources = [
        {'title': f'NHTSA recall {d["id"]} — {d["component"][:65]}',
         'url': d['url'], 'source_id': d['id'], 'excerpt': d['text'][:320]}
        for d in state['documents']
    ]
    await store.add_message(owner, chat_id, 'user', request.message,
                            {'vehicle': {'make': request.make, 'model': request.model, 'year': request.year}})
    await store.add_message(owner, chat_id, 'assistant', state['answer'],
                            {'sources': sources, 'classification': state['classification'],
                             'generation_mode': state['generation_mode'],
                             'generation_error': state.get('generation_error'),
                             'intent': state.get('intent', 'automotive')})
    return {'conversation_id': chat_id, 'answer': state['answer'],
            'classification': state['classification'], 'sources': sources,
            'recall_status': state['recall_status'], 'generation_mode': state['generation_mode'],
            'generation_error': state.get('generation_error'),
            'intent': state.get('intent', 'automotive')}

# Same /agent graph and REST API live on this one FastAPI app.
mount_agent(app)
