"""Offline migration contract checks; real integration requires WSL dependencies."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = '\n'.join((ROOT / part).read_text(encoding='utf-8') for part in ('agent/runtime/state.py', 'agent/runtime/factory.py', 'agent/runtime/tools.py', 'agent/runtime/lifecycle.py', 'agent/runtime/grounding.py'))
TREE = ast.parse((ROOT / 'agent/runtime/tools.py').read_text(encoding='utf-8'))


def test_live_agent_is_built_by_langchain_factory():
    assert 'from langchain.agents import create_agent' in SOURCE and 'class StoryState(AgentState)' in SOURCE
    assert 'return create_agent(' in SOURCE
    assert 'StateGraph(' not in SOURCE
    assert 'from agent.agentic_llm import' not in SOURCE
    assert 'from agent.agentic_planner import' not in SOURCE
    assert 'from agent.generation import' not in SOURCE


def test_bounded_tools_have_real_backend_calls():
    assert 'predict_category(' in SOURCE
    assert 'await fetch_recalls(' in SOURCE
    assert 'retrieve(' in SOURCE
    for name in ['classify_symptoms', 'search_nhtsa_recalls', 'retrieve_recall_evidence', 'get_project_metrics']:
        assert any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in TREE.body)


def test_model_uses_single_openrouter_integration_and_ms_timeout():
    assert 'ChatOpenRouter(**model_kwargs)' in SOURCE
    assert 'timeout=int(settings.openrouter_timeout_seconds * 1000)' in SOURCE
    assert 'httpx' not in SOURCE
    assert 'OPENROUTER_API_KEY' in SOURCE


def test_turn_snapshots_are_from_actual_tools():
    assert "if e.get('turn') == turn" in SOURCE
    assert "'repair_status_checked': False" in SOURCE
    assert "'vin_eligibility_checked': False" in SOURCE


def test_no_provider_reasoning_in_final_chat_history():
    assert "b.get('type') == 'text'" in SOURCE
    assert "result['messages'] = [final.model_copy(update={'content': grounded})]" in SOURCE


def test_recall_lookups_have_explicit_user_grounding():
    assert '_vehicle_authorized(make, model, year, runtime.state)' in SOURCE
    assert 'if state.get(\'recall_status\') != \'ok\':' in SOURCE


def test_root_dotenv_reasoning_flag_is_loaded_not_only_shell_env():
    # pydantic_settings does not export variables loaded from .env to os.environ.
    assert "dotenv_values(settings.model_config.get('env_file'))" in SOURCE
    assert 'reasoning_enabled' in SOURCE
