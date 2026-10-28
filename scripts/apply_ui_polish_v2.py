"""Apply narrow improvements to the existing StoryWorkspace, without replacing pages/CSS.

Usage (project root): python -m scripts.apply_ui_polish_v2
Idempotent on repeated runs. A source mismatch stops before writing the file.
"""
from __future__ import annotations

from pathlib import Path
import shutil

from scripts.polish_generation_fallback import apply as polish_generation

CHANGES = [
    ("MessageCircle, ShieldCheck, Sparkles, StopCircle,",
     "MessageCircle, ShieldCheck, Sparkles, StopCircle, UserRound,"),
    ("function WorkflowPanel({ state, running }: { state: UiState; running: boolean }) {",
     "function WorkflowPanel({ state, running, failed }: { state: UiState; running: boolean; failed: boolean }) {"),
    ("  const awaiting = state.stage === 'awaiting_vehicle'\n",
     "  const awaiting = state.stage === 'awaiting_vehicle'\n"
     "  const status = running ? 'Investigating' : failed ? 'Interrupted' :\n"
     "    state.stage === 'complete' ? 'Completed' : awaiting ? 'Need vehicle' : 'Ready'\n"),
    ("<span className={running ? 'aa-running' : 'aa-idle'}><span/>{running ? 'Running' : 'Ready'}</span>",
     "<span className={running ? 'aa-running' : state.stage === 'complete' ? 'aa-finished' : 'aa-idle'}><span/>{status}</span>"),
    ("          <div><strong>{step.label}</strong><p>{step.detail}</p></div>",
     "          <div><strong>{step.label}</strong><p>{step.detail}</p>\n"
     "            <span className=\"aa-step-state\">{done ? 'Completed' : active ? 'In progress' : 'Pending'}</span>\n"
     "          </div>"),
    ("<div className=\"aa-message-avatar\">{m.role === 'assistant' ? <Sparkles size={17}/> : 'You'}</div>",
     "<div className=\"aa-message-avatar\" aria-hidden=\"true\">{m.role === 'assistant' ? <Sparkles size={17}/> : <UserRound size={16}/>}</div>"),
    ("            {!running && state.stage === 'complete' && <ResultsUI state={state}/>}\n",
     "            {!running && state.stage === 'complete' && state.generation_mode && state.generation_mode !== 'llm' &&\n"
     "              <div className=\"aa-generation-notice\" role=\"status\">\n"
     "                <ShieldCheck size={16}/>\n"
     "                <div><strong>Evidence-only response</strong><p>The AI explanation service was unavailable; this answer uses the classifier and retrieved records only.\n"
     "                  {state.generation_error && <span> {state.generation_error}</span>}</p></div>\n"
     "              </div>}\n"
     "            {!running && state.stage === 'complete' && <ResultsUI state={state}/>}\n"),
    ("<WorkflowPanel state={state} running={running}/>",
     "<WorkflowPanel state={state} running={running} failed={Boolean(error)}/>"),
]

CSS_APPEND = """
/* Polished chat execution / fallback states — scoped to the story workspace. */
.aa-finished { font-size:9px;font-weight:800; color:#247e66; display:flex; align-items:center; gap:6px; }
.aa-finished span { width:6px;height:6px;border-radius:50%;background:currentColor; }
.aa-step-state { display:inline-block; margin-top:5px; color:#a3adb5; font-size:9px; }
.aa-step.done .aa-step-state { color:#30856f; }
.aa-step.active .aa-step-state { color:#9b7846; }
.aa-generation-notice { display:flex;align-items:flex-start;gap:10px;margin-left:48px;padding:13px 15px;border:1px solid #f0dab1;border-radius:11px;background:#fff9ed;color:#74572e; }
.aa-generation-notice svg { flex-shrink:0; margin-top:1px; }
.aa-generation-notice strong { font-size:11px; }
.aa-generation-notice p { font-size:10px;line-height:1.7;margin:5px 0 0; }
@media(max-width:660px) { .aa-generation-notice { margin-left:0; } }
"""


def update_story(source: str) -> str:
    content = source
    for old, new in CHANGES:
        # A replacement can contain the original snippet, so check for a
        # previously applied replacement BEFORE looking for the old text.
        if new in content:
            continue
        if old not in content or content.count(old) != 1:
            raise RuntimeError(f"Unexpected StoryWorkspace revision; no changes written. Missing: {old[:70]}")
        content = content.replace(old, new, 1)
    return content


def apply(root: Path) -> list[str]:
    story = root / 'frontend/src/components/StoryWorkspace.tsx'
    css = root / 'frontend/src/app/globals.css'
    gen = root / 'agent/generation.py'
    text = story.read_text(encoding='utf-8')
    edited = update_story(text)  # Validate all snippets before overwriting anything
    css_text = css.read_text(encoding='utf-8')
    if '.aa-generation-notice {' not in css_text:
        css_text += CSS_APPEND
    actions = []
    if edited != text:
        backup = story.with_suffix('.tsx.before_ui_polish.bak')
        if not backup.exists(): shutil.copy2(story, backup)
        story.write_text(edited, encoding='utf-8')
        actions.append('Updated story UI (backup created).')
    if css_text != css.read_text(encoding='utf-8'):
        backup = css.with_suffix('.css.before_ui_polish.bak')
        if not backup.exists(): shutil.copy2(css, backup)
        css.write_text(css_text, encoding='utf-8')
        actions.append('Appended CSS (backup created).')
    if gen.exists():
        actions.append(polish_generation(gen))
    else:
        actions.append('agent/generation.py not found: skipped optional fallback text polish.')
    return actions


if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[1]
    for event in apply(repo): print(event)
