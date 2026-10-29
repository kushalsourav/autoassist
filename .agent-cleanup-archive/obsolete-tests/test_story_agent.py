"""Tests for the semantic story agent (replaces obsolete regex-parser tests)."""
from __future__ import annotations

import json
import pytest
from pydantic import ValidationError

from agent.agentic_planner import AgentPlan, VehicleIdentity, make_plan, merge_vehicle, skill_descriptions
from agent.agentic_llm import AgentLLMError, decode_json_object


BASE = {
    'skill': 'diagnose_symptoms', 'action': 'classify',
    'vehicle': {'make': None, 'model': None, 'year': None},
    'summary': 'My brake pedal is very hard and the stopping distance increased',
    'response_guidance': 'Give immediate safety advice before asking a question.',
    'followups': ['Is the problem constant or intermittent?'],
}


def test_json_fenced_response():
    assert decode_json_object('```json\n{"a": 1}\n```') == {'a': 1}
    with pytest.raises(AgentLLMError):
        decode_json_object('analysis {"action":"recall"}')


def test_plan_schema_blocks_unknown_actions_and_extra_fields():
    assert AgentPlan.model_validate(BASE).action == 'classify'
    with pytest.raises(ValidationError):
        AgentPlan.model_validate({**BASE, 'action': 'run_shell'})
    with pytest.raises(ValidationError):
        AgentPlan.model_validate({**BASE, 'secret': 'x'})


def test_arbitrary_make_model_without_catalog():
    data = {**BASE, 'vehicle': {'make': 'Škoda', 'model': 'Kushaq', 'year': 2023}}
    assert AgentPlan.model_validate(data).vehicle.model == 'Kushaq'


def test_no_invented_vehicle_default():
    assert merge_vehicle({}, VehicleIdentity()) == {}
    assert not VehicleIdentity().complete()


def test_switch_vehicle_does_not_inherit_old_year():
    result = merge_vehicle({'make': 'Honda', 'model': 'Accord', 'year': 2018}, VehicleIdentity(make='Toyota'))
    assert result == {'make': 'Toyota'}
    result = merge_vehicle({'make': 'Ford', 'model': 'Escape', 'year': 2018}, VehicleIdentity(model='Mustang'))
    assert result == {'make': 'Ford', 'model': 'Mustang'}


def test_skills_are_real_trusted_documents():
    descriptions = skill_descriptions()
    assert 'diagnose_symptoms' in descriptions
    assert 'recall_research' in descriptions
    assert 'automotive_education' in descriptions


@pytest.mark.asyncio
async def test_general_questions_answer_without_vehicle(monkeypatch):
    async def reply(*args, **kwargs):
        return json.dumps({**BASE, 'skill': 'automotive_education', 'action': 'answer',
                           'summary': 'Explain what OBD-II codes are'})
    monkeypatch.setattr('agent.agentic_planner.chat', reply)
    plan = await make_plan([{'role': 'user', 'content': 'What is an OBD-II code?'}])
    assert plan.action == 'answer'
    assert plan.vehicle.make is None


@pytest.mark.asyncio
async def test_incomplete_recall_action_downgraded_safely(monkeypatch):
    async def reply(*args, **kwargs):
        return json.dumps({**BASE, 'skill': 'recall_research', 'action': 'recall',
                           'vehicle': {'make': 'Skoda', 'model': None, 'year': None}})
    monkeypatch.setattr('agent.agentic_planner.chat', reply)
    plan = await make_plan([{'role': 'user', 'content': 'Any recalls for my Skoda?'}])
    assert plan.action == 'answer'
    assert plan.vehicle.make == 'Skoda'


@pytest.mark.asyncio
async def test_multi_turn_vehicle_resolution_keeps_context(monkeypatch):
    async def reply(*args, **kwargs):
        return json.dumps({**BASE, 'skill': 'diagnose_symptoms', 'action': 'classify_and_recall',
                           'vehicle': {'make': 'Honda', 'model': 'Accord', 'year': 2018},
                           'summary': 'Hard brake pedal and increased stopping distance on 2018 Honda Accord'})
    monkeypatch.setattr('agent.agentic_planner.chat', reply)
    plan = await make_plan([{'role': 'user', 'content': 'My brakes feel hard.'},
                            {'role': 'assistant', 'content': 'What car is it?'},
                            {'role': 'user', 'content': '2018 Honda Accord'}],
                           previous_summary='Hard brake pedal')
    assert plan.action == 'classify_and_recall'
    assert plan.vehicle.complete()
    assert 'brake' in plan.summary.lower()


@pytest.mark.asyncio
async def test_invalid_json_retried_then_explicit_error(monkeypatch):
    calls = []
    async def reply(*args, **kwargs):
        calls.append(1)
        return 'This is not a JSON object'
    monkeypatch.setattr('agent.agentic_planner.chat', reply)
    with pytest.raises(AgentLLMError):
        await make_plan([{'role': 'user', 'content': 'Please help'}])
    assert len(calls) == 2


def test_live_story_graph_uses_langchain_create_agent():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[2] / 'agent/runtime/factory.py').read_text()
    assert 'create_agent(' in source
    assert 'ChatOpenRouter(' in source
    assert 'StateGraph(' not in source
    assert 'story_vehicle' not in source
    assert '_SYMPTOM' not in source


def test_hallucinated_vehicle_fields_are_rejected():
    from agent.agentic_planner import ground_vehicle_claims
    result = ground_vehicle_claims(
        VehicleIdentity(make='Mercedes-Benz', model='C-Class', year=2024), {},
        [{'role': 'user', 'content': 'My car brakes are hard. Can you help?'}])
    assert result.model_dump(exclude_none=True) == {}


def test_explicit_vehicle_source_is_accepted():
    from agent.agentic_planner import ground_vehicle_claims
    result = ground_vehicle_claims(
        VehicleIdentity(make='Honda', model='Accord', year=2018), {},
        [{'role': 'user', 'content': 'It is my 2018 Honda Accord'}])
    assert result.complete()
