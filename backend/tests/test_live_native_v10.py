"""V10 frontend regressions: native AG-UI handling and no duplicate UI surfaces."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONT = ROOT / 'frontend/src/components'


def test_reasoning_and_tools_are_native_protocol_events():
    src = (FRONT / 'modelStreamEvents.ts').read_text(encoding='utf8')
    assert 'REASONING_MESSAGE_CONTENT' in src
    assert 'TOOL_CALL_START' in src
    assert 'TOOL_CALL_ARGS' in src
    assert 'TOOL_CALL_RESULT' in src
    assert 'const delta = visible(str(event.delta))' in src
    assert 'last.text + item.text' in src
    assert 'JSON.stringify(data.input)' not in src


def test_ui_is_live_and_local_first():
    src = (FRONT / 'StoryWorkspace.tsx').read_text(encoding='utf8')
    assert "process.env.NODE_ENV === 'development'" in src
    assert src.count('onReasoningMessageContentEvent:') == 1
    assert src.count('onToolCallStartEvent:') == 1
    assert src.count('onToolCallArgsEvent:') == 1
    assert src.count('onToolCallResultEvent:') == 1
    assert src.count('<ModelStreamViewer items={streamEvents} running={running}/>') == 1
    assert '{running && !SHOW_STREAM_DEBUG && <LiveActivity' in src
    assert '{!SHOW_STREAM_DEBUG && providerReasoning' in src
    assert 'setStreamEvents([])' in src


def test_viewer_does_not_duplicate_answer_or_expose_system_prompt():
    src = (FRONT / 'ModelStreamViewer.tsx').read_text(encoding='utf8')
    assert 'Provider reasoning' in src
    assert 'Tool activity' in src
    assert 'No tool' in src
    assert "e.kind==='reasoning'" in src
    assert "e.kind==='tool-request'" in src
    assert "e.kind==='answer'" not in src
    assert 'JSON.stringify(data.input)' not in src
