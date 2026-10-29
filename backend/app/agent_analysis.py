"""Compatibility adapter: /api/analyze uses the SAME LangChain agent as /agent.

This is not a second workflow or model client. REST chat history lives in the
existing SQLite/Supabase store; the LangGraph checkpoint is a short-term cache.
"""
from __future__ import annotations

from uuid import uuid4
from typing import Any


def _answer_text(state: dict[str, Any]) -> str:
    from langchain_core.messages import AIMessage

    text = state.get('answer')
    if isinstance(text, str) and text.strip():
        return text.strip()
    for msg in reversed(state.get('messages', [])):
        if not isinstance(msg, AIMessage) or msg.tool_calls:
            continue
        content = msg.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return ''.join(block.get('text', '') for block in content
                           if isinstance(block, dict) and block.get('type') == 'text').strip()
    return ''


async def run_analysis(request: Any, history: list[dict], *,
                       owner_id: str = '', conversation_id: str = '') -> dict[str, Any]:
    """Run the existing create_agent graph, preserving the old JSON response shape.

    - Only a server-authenticated owner may supply conversation_id.
    - Checkpoint names are namespaced by owner, never shared with AG-UI threads.
    - On cold start restore stored conversation history once; on warm calls append
      only the new user turn, so messages don't get duplicated.
    """
    from langchain_core.messages import AIMessage, HumanMessage
    from agent.story_agent import story_graph

    if not isinstance(request.message, str) or not request.message.strip():
        raise ValueError('A nonempty message is required')
    vehicle = {'make': request.make, 'model': request.model, 'year': request.year}
    # The form fields are user-provided, not hallucinated model attributes.
    current = HumanMessage(content=(
        f'{request.message}\n\n'
        f'User-provided vehicle details (for recall lookup): '
        f'{request.year} {request.make} {request.model}.'))
    # Stable only for authenticated, persisted REST chats. One-off calls use a
    # random ID; this prevents another user's memory from being loaded.
    thread_id = (f'rest:{owner_id}:{conversation_id}' if owner_id and conversation_id
                 else f'rest-ephemeral:{uuid4()}')
    config = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 20}
    saved = None
    if owner_id and conversation_id:
        # InMemorySaver is not durable: reconstruct from the user-owned store
        # after restarts instead of silently losing prior conversation context.
        saved = await story_graph.aget_state(config)
    warm = bool(saved is not None and getattr(saved, 'values', {}).get('messages'))
    messages = []
    if not warm:
        for item in history[-8:]:
            if item.get('role') == 'user' and isinstance(item.get('content'), str):
                messages.append(HumanMessage(content=item['content'][:2000]))
            elif item.get('role') == 'assistant' and isinstance(item.get('content'), str):
                messages.append(AIMessage(content=item['content'][:4000]))
    messages.append(current)

    try:
        state = await story_graph.ainvoke({'messages': messages, 'vehicle': vehicle}, config=config)
    finally:
        # The first request has no persisted conversation ID yet. Its history
        # is stored in SQLite/Supabase after success, so release the temporary
        # in-memory checkpoint to avoid one orphaned thread per new chat.
        if thread_id.startswith('rest-ephemeral:'):
            saver = getattr(story_graph, 'checkpointer', None)
            delete = getattr(saver, 'adelete_thread', None)
            if callable(delete):
                try:
                    await delete(thread_id)
                except Exception:
                    import logging
                    logging.getLogger(__name__).warning('Failed to release temporary checkpoint')
    answer = _answer_text(state)
    if not answer:
        raise RuntimeError('The agent returned no user-facing answer')
    prediction = state.get('classification') or {}
    if not prediction.get('category'):
        prediction = {'category': 'Not evaluated', 'confidence': None,
                      'method': 'not_requested_by_agent', 'labels': []}
    else:
        prediction = dict(prediction)
        prediction.setdefault('confidence', None)
        prediction.setdefault('labels', [])
    documents = state.get('documents') or []
    return {'answer': answer, 'classification': prediction,
            'documents': documents,
            'sources': state.get('sources') or [],
            'recall_status': state.get('recall_status', 'not_queried'),
            'generation_mode': 'langchain_agent', 'generation_error': None,
            'intent': 'automotive'}
