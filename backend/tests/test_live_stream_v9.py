"""V9 viewer updated by V10 to use native AG-UI protocol events first."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONT = ROOT / 'frontend/src/components'
WORKSPACE = FRONT / 'StoryWorkspace.tsx'


def test_v9_v10_model_metadata_remains_safe():
    src = (FRONT / 'modelStreamEvents.ts').read_text(encoding='utf8')
    assert "event === 'on_chat_model_start'" in src
    assert "event === 'on_chat_model_end'" in src
    assert 'output.usage_metadata' in src
    assert 'JSON.stringify(data.input)' not in src  # never display provider prompts
    assert "type === 'REASONING_MESSAGE_CONTENT'" in src
    assert "type === 'TOOL_CALL_RESULT'" in src


def test_v9_v10_one_in_chat_viewer():
    src = WORKSPACE.read_text(encoding='utf8')
    assert 'NEXT_PUBLIC_AUTOASSIST_STREAM_DEBUG' in src
    assert 'parseLangChainStream(payload)' in src
    assert 'parseAgUiStreamEvent(payload)' in src
    assert src.count('<ModelStreamViewer items={streamEvents} running={running}/>') == 1
    assert src.count('onReasoningMessageContentEvent:') == 1
    assert src.count('onToolCallResultEvent:') == 1
    assert src.count('setStreamEvents([])') == 3
    assert 'return create_agent(' in (ROOT / 'agent/story_agent.py').read_text(encoding='utf8')


def test_v9_v10_css_remains_additive():
    css = (ROOT / 'frontend/src/app/globals.css').read_text(encoding='utf8')
    assert '.aa-v9-stream-viewport' in css
    assert '.aa-v10-live' in css
