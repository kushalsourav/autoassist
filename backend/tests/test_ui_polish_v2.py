"""Self-contained checks for the narrow fallback text patch and diagnostics."""
from pathlib import Path

from scripts.polish_generation_fallback import apply
from scripts.diagnose_story_llm import explain_status


def test_fallback_patch_is_narrow_and_idempotent(tmp_path: Path):
    target = tmp_path / 'generation.py'
    target.write_text('abc "**Non-LLM fallback summary (generation was not available).**\\n\\n" xyz')
    assert 'Updated' in apply(target)
    assert 'Non-LLM fallback summary' not in target.read_text()
    assert target.with_suffix('.py.before_ui_polish.bak').exists()
    assert 'unchanged' in apply(target)


def test_fallback_patch_does_not_modify_unknown_code(tmp_path: Path):
    target = tmp_path / 'generation.py'
    target.write_text('def other(): pass\n')
    assert 'unchanged' in apply(target)
    assert target.read_text() == 'def other(): pass\n'


def test_safe_status_messages():
    assert 'quota' in explain_status(429)
    assert 'MODEL' in explain_status(404)
    assert 'KEY' in explain_status(401)


def test_story_patcher_changes_only_expected_blocks():
    from scripts.apply_ui_polish_v2 import update_story
    original = (
        "MessageCircle, ShieldCheck, Sparkles, StopCircle,\n"
        "function WorkflowPanel({ state, running }: { state: UiState; running: boolean }) {\n"
        "  const awaiting = state.stage === 'awaiting_vehicle'\n"
        "<span className={running ? 'aa-running' : 'aa-idle'}><span/>{running ? 'Running' : 'Ready'}</span>\n"
        "          <div><strong>{step.label}</strong><p>{step.detail}</p></div>\n"
        "<div className=\"aa-message-avatar\">{m.role === 'assistant' ? <Sparkles size={17}/> : 'You'}</div>\n"
        "            {!running && state.stage === 'complete' && <ResultsUI state={state}/>}\n"
        "<WorkflowPanel state={state} running={running}/>"
    )
    once = update_story(original)
    twice = update_story(once)
    assert once == twice
    assert '<UserRound size={16}/>' in once
    assert "? 'Completed'" in once
    assert 'aa-generation-notice' in once
