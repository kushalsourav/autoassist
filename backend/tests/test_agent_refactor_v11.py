"""Static/offline architecture tests; live LangChain integration runs in WSL."""
from __future__ import annotations

import ast
from pathlib import Path
import runpy
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / 'agent'
RUNTIME = AGENT / 'runtime'


def text(name):
    return (RUNTIME / name).read_text(encoding='utf-8')


def test_single_live_graph_factory_and_public_entrypoint():
    source = text('factory.py')
    wrapper = (AGENT / 'story_agent.py').read_text(encoding='utf-8')
    assert 'return create_agent(' in source
    assert 'ChatOpenRouter(**model_kwargs)' in source
    assert source.count('story_graph = build_story_graph()') == 1
    assert 'from agent.runtime.factory import build_story_graph, story_graph' in wrapper
    assert 'StateGraph(' not in source + wrapper
    assert 'httpx' not in source


def test_real_callable_tools_stay_registered():
    source = text('tools.py')
    for name in ('classify_symptoms', 'search_nhtsa_recalls',
                 'retrieve_recall_evidence', 'get_project_metrics'):
        assert f'def {name}(' in source
        assert name in text('factory.py')
    assert 'predict_category(' in source
    assert 'await fetch_recalls(' in source
    assert 'retrieve(symptoms, records,' in source
    assert 'runtime.tool_call_id' in source
    assert 'HumanMessage' in source  # regression: missing import after split


def test_grounding_and_state_are_preserved():
    ground = text('grounding.py')
    state = text('state.py')
    audit = text('lifecycle.py')
    assert '_vehicle_authorized(' in ground
    assert 'ensure_unverified_recall_status(' in ground
    assert 'class StoryState(AgentState):' in state
    assert "'vin_eligibility_checked': False" in audit
    assert "'repair_status_checked': False" in audit
    assert "if e.get('turn') == turn" in audit
    assert 'begin_investigation' in audit and 'finish_investigation' in audit


def test_policy_retains_recall_safety_and_measured_metrics():
    source = text('constants.py')
    assert 'Never assert a mechanical cause' in source
    assert 'If NHTSA isn\'t queried' in source
    assert 'no VIN verification tool exists' in source
    constants = ast.parse(source)
    assign = next(n for n in constants.body if isinstance(n, ast.Assign) and
                  any(isinstance(t, ast.Name) and t.id == 'PUBLIC_PROJECT_SCOPE' for t in n.targets))
    metrics = ast.literal_eval(assign.value)
    assert metrics['train'] == 4235
    assert metrics['distilbert_macro_f1'] == 0.8054
    assert metrics['indexed_unique_recall_campaigns'] == 67


def test_only_one_model_factory_in_active_runtime():
    imports = text('factory.py') + text('tools.py') + text('lifecycle.py')
    assert 'from agent.agentic_llm' not in imports
    assert 'from agent.agentic_planner' not in imports
    assert 'from agent.generation' not in imports
    assert 'from agent.workflow' not in imports
    assert sum('ChatOpenRouter(' in p.read_text() for p in RUNTIME.glob('*.py')) == 1


def test_cleanup_is_dry_run_and_preserves_skills_and_referenced_modules(tmp_path):
    root = tmp_path / 'project'
    (root / 'scripts').mkdir(parents=True)
    (root / 'agent' / 'skills' / 'diagnose_symptoms').mkdir(parents=True)
    (root / 'backend').mkdir()
    src = ROOT / 'scripts/cleanup_agent_folder_v11.py'
    tool = root / 'scripts' / src.name
    tool.write_text(src.read_text())
    (root / 'agent' / 'story_agent.py').write_text('"""shim"""')
    (root / 'agent' / 'old.py.bak').write_text('old contents')
    (root / 'agent' / 'skills' / 'diagnose_symptoms' / 'SKILL.md').write_text('KEEP')
    (root / 'agent' / 'agentic_llm.py').write_text('OLD')
    (root / 'backend' / 'check.py').write_text('from agent.agentic_llm import chat')
    cmd = [sys.executable, str(tool), '--prune-unused']
    result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    assert 'DRY RUN' in result.stdout
    assert (root / 'agent' / 'old.py.bak').exists()
    assert (root / 'agent' / 'agentic_llm.py').exists()
    subprocess.run(cmd + ['--apply'], text=True, capture_output=True, check=True)
    assert not (root / 'agent' / 'old.py.bak').exists()
    assert list((root / '.agent-cleanup-archive').rglob('old.py.bak'))
    assert (root / 'agent' / 'agentic_llm.py').exists()
    assert (root / 'agent' / 'skills' / 'diagnose_symptoms' / 'SKILL.md').read_text() == 'KEEP'
