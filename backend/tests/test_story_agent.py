from agent.story_vehicle import extract_vehicle, missing_vehicle_fields, vehicle_question


def test_no_defaults_or_guesses():
    assert extract_vehicle('My car suddenly lost power on the highway') == {}
    assert missing_vehicle_fields({}) == ['make', 'model', 'year']


def test_story_extracts_vehicle():
    assert extract_vehicle('My 2018 Honda Accord wipers fail') == {
        'make': 'Honda', 'model': 'Accord', 'year': 2018}


def test_followup_fills_missing_year():
    previous = extract_vehicle('I drive a Honda Civic')
    assert extract_vehicle('It is a 2018', previous) == {'make': 'Honda', 'model': 'Civic', 'year': 2018}


def test_switching_make_discards_stale_model():
    previous = extract_vehicle('2018 Honda Accord')
    assert extract_vehicle('Actually a Toyota', previous) == {'make': 'Toyota', 'year': 2018}


def test_clarification_is_natural():
    assert 'model year' in vehicle_question({'make': 'Honda', 'model': 'Civic'})


def test_parser_handles_variants():
    assert extract_vehicle('2019 Nissan Rogue')['model'] == 'Rogue'
    assert extract_vehicle('My 2018 Ford F-150')['model'] == 'F-150'


def test_story_flow_intake_reuses_symptom():
    import pytest
    pytest.importorskip('langgraph')
    from langchain_core.messages import HumanMessage
    from agent.story_agent import intake_node
    state = {'messages': [HumanMessage(content='My windshield wipers do not work'),
                          HumanMessage(content='It is a 2018 Honda Accord')]}
    result = intake_node(state)
    assert result['vehicle']['model'] == 'Accord'
    assert result['symptom'] == 'My windshield wipers do not work'
    assert result['stage'] == 'vehicle_identified'
