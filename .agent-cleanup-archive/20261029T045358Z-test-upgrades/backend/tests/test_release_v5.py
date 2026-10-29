"""Release V5 offline tests. No live LLM, NHTSA, GPU, or CopilotKit required."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / 'agent/story_agent.py'
CHAT = ROOT / 'frontend/src/components/StoryWorkspace.tsx'
LAB = ROOT / 'frontend/src/app/model-lab/page.tsx'
HISTORY = ROOT / 'frontend/src/components/localThreads.ts'


def _snapshot_fn():
    tree = ast.parse(AGENT.read_text(encoding='utf8'))
    function = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                    and item.name == '_snapshot')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, '<isolated-result-snapshot>', 'exec'), namespace)
    return namespace['_snapshot']


def test_evidence_metadata_is_observed_not_model_estimated():
    snapshot_fn = _snapshot_fn()
    state = {'messages': [], 'recall_status': 'ok',
             'recalls': [{'id': 'A'}, {'id': 'B'}],
             'documents': [{'id': 'B'}],
             'sources': [{'source_id': 'B'}], 'classification': {'category': 'Brakes'}}
    snapshot = snapshot_fn(state, turn=2)
    assert snapshot['turn'] == 2
    assert snapshot['evidence_summary']['model_year_records'] == 2
    assert snapshot['evidence_summary']['retrieved_candidates'] == 1
    assert snapshot['evidence_summary']['lookup_scope'] == 'make_model_year'
    assert snapshot['evidence_summary']['vin_eligibility_checked'] is False
    assert snapshot['evidence_summary']['repair_status_checked'] is False


def test_no_lookup_is_not_misrepresented_as_empty_recall_database():
    snapshot_fn = _snapshot_fn()
    snapshot = snapshot_fn({'messages': []}, turn=1)
    assert snapshot['evidence_summary']['lookup_status'] == 'not_queried'
    assert snapshot['evidence_summary']['model_year_records'] == 0
    assert snapshot['evidence_summary']['retrieved_candidates'] == 0


def test_agent_guidance_keeps_recall_and_diagnosis_claims_separate():
    source = AGENT.read_text(encoding='utf8')
    assert 'Do not claim access to manufacturer service bulletins' in source
    assert 'Distinguish reported symptoms' in source
    assert 'ensure_unverified_recall_status' in source
    assert 'ChatOpenRouter(**model_kwargs)' in source  # native model streaming


def test_chat_renders_only_observed_steps_and_preserves_per_turn_history():
    source = CHAT.read_text(encoding='utf8')
    history = HISTORY.read_text(encoding='utf8')
    assert 'onStepStartedEvent:' in source
    assert 'onStepFinishedEvent:' in source
    assert 'View agent steps' in source
    assert 'stepsByTurnRef.current' in source
    assert '<TurnTrace steps={stepsByTurn[String(userTurn)] || []}' in source
    assert 'stepsByTurn?:' in history
    assert 'const FLOW =' not in source  # no fabricated static stage list
    assert 'aa-v5-draft-caption' in source


def test_model_lab_separates_real_and_retrospective_retrieval_scores():
    source = LAB.read_text(encoding='utf8')
    assert '.8054' in source and '.7754' in source
    assert '0.222' in source and '0.611' in source and '0.278' in source
    assert 'not deployed' in source
    assert 'Assistant-reviewed' in source
    assert '204.89 → 13.79 ms' in source
    assert abs((.8054 - .7754) - .03) < 1e-10
