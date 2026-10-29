"""One-turn live smoke test of the exact same graph imported by the CopilotKit server.

Runs a simple metric question first, then emits *only* counts of streamed tokens and
whether provider reasoning was visible. Does not print reasoning or any API key.
"""
import asyncio
from uuid import uuid4
from langchain_core.messages import AIMessage
from agent.story_agent import story_graph


async def main():
    config = {'configurable': {'thread_id': str(uuid4())}}
    print('Probing real LangChain create_agent via ChatOpenRouter.')
    print('Note: OpenRouter free models may lack tool calling or visible reasoning.')
    event_counts = {}
    async for event in story_graph.astream_events(
        {'messages': [{'role': 'user', 'content': 'How many complaints trained AutoAssist and what was DistilBERT test macro-F1?'}]},
        config=config, version='v2',
    ):
        kind = str(event.get('event', ''))
        event_counts[kind] = event_counts.get(kind, 0) + 1
        if kind == 'on_chat_model_stream':
            chunk = event.get('data', {}).get('chunk')
            if chunk is not None:
                blocks = getattr(chunk, 'content_blocks', [])
                for block in blocks:
                    if isinstance(block, dict) and block.get('type') == 'reasoning':
                        event_counts['readable_reasoning_blocks'] = event_counts.get('readable_reasoning_blocks', 0) + 1
    print('Observed events:', {k: v for k, v in event_counts.items()
                               if k in {'on_chat_model_stream','on_tool_start','on_tool_end','readable_reasoning_blocks'}})
    state = await story_graph.aget_state(config)
    last = next((m for m in reversed(state.values['messages']) if isinstance(m, AIMessage) and not m.tool_calls), None)
    print('Last answer preview:', str(last.content)[:260] if last else '(no final answer)')
    print('Tool list:', [e['tool'] for e in state.values.get('tool_events', [])])


if __name__ == '__main__':
    asyncio.run(main())
