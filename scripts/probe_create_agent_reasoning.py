"""Optional LangChain create_agent + ChatOpenRouter reasoning probe.

Run: python -m scripts.probe_create_agent_reasoning
Requires: pip install -U langchain langchain-openrouter
Uses project OPENROUTER_MODEL and OPENROUTER_API_KEY environment variables.
NOT the deployed AutoAssist workflow. Verify model/provider support first.
"""
from __future__ import annotations

import asyncio
import os


def _reasoning_blocks(message):
    for block in getattr(message, 'content_blocks', []) or []:
        if isinstance(block, dict) and block.get('type') == 'reasoning':
            value = block.get('reasoning')
            if isinstance(value, str) and value and '[REDACTED]' not in value:
                yield value


async def main() -> None:
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_core.messages import AIMessageChunk
    from langchain_openrouter import ChatOpenRouter
    from langgraph.checkpoint.memory import InMemorySaver

    from backend.app.config import settings
    api_key = os.getenv('OPENROUTER_API_KEY') or settings.openrouter_api_key
    if not api_key:
        raise SystemExit('OPENROUTER_API_KEY must be set in .env or your environment (do not print it).')
    model_name = os.getenv('OPENROUTER_MODEL') or settings.openrouter_model
    if not model_name:
        raise SystemExit('Set OPENROUTER_MODEL to a specific reasoning-capable OpenRouter model.')

    @tool
    def autoassist_scope() -> str:
        """Return verified scope of the AutoAssist automotive AI prototype."""
        return ('AutoAssist has 6,052 cleaned labeled owner complaints used for ML experiments, '
                'with 67 unique recall campaigns indexed in a local research snapshot. '
                'Neither is the complete NHTSA database. VIN status is never verified.')

    # The partner integration reads OPENROUTER_API_KEY from the environment.
    os.environ.setdefault('OPENROUTER_API_KEY', api_key)
    model = ChatOpenRouter(
        model=model_name,
        temperature=0,
        max_tokens=4096,
        timeout=90,  # seconds, not milliseconds
        max_retries=0,
        reasoning={'effort': 'low', 'summary': 'auto'},
    )
    agent = create_agent(
        model=model,
        tools=[autoassist_scope],
        checkpointer=InMemorySaver(),
        system_prompt=('You are an automotive assistant. Answer directly. Use the scope tool when '
                       'asked about AutoAssist case counts. Never infer VIN recall eligibility.'),
    )
    print(f'Using model: {model_name} (does not guarantee readable reasoning or tool support)')
    # These are agent-level LangGraph streaming modes; tool steps come as updates.
    async for mode, packet in agent.astream(
        {'messages': [{'role': 'user', 'content':
                       'How many complaints does AutoAssist use? Please consult the available tool.'}]},
        config={'configurable': {'thread_id': 'local-create-agent-probe'}},
        stream_mode=['messages', 'updates'],
    ):
        if mode == 'messages':
            message, _metadata = packet
            if not isinstance(message, AIMessageChunk):
                continue
            for chunk in _reasoning_blocks(message):
                print(f'\n[PROVIDER REASONING] {chunk}', end='', flush=True)
            blocks = getattr(message, 'content_blocks', []) or []
            for block in blocks:
                if isinstance(block, dict) and block.get('type') == 'text':
                    print(block.get('text', ''), end='', flush=True)
        elif mode == 'updates':
            # Do not print raw tool inputs, secrets or full model state.
            for node in packet:
                print(f'\n[AGENT NODE] {node}', flush=True)
    print('\nDone.')


if __name__ == '__main__':
    asyncio.run(main())
