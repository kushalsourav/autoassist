"""Offline regression checks for V6; no cloud calls or frontend dependencies."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAT = ROOT / 'frontend/src/components/StoryWorkspace.tsx'
LAB = ROOT / 'frontend/src/app/model-lab/page.tsx'
CHARTS = ROOT / 'frontend/src/components/ModelCharts.tsx'
AGENT = ROOT / 'agent/story_agent.py'
CSS = ROOT / 'scripts/assets/autoassist-v6.css'


def test_one_live_trace_surface_and_per_turn_audit():
    code = CHAT.read_text()
    assert '<LiveActivity steps={liveSteps}/>' in code
    assert '<TurnTrace steps={stepsByTurn[String(userTurn)] || []}' in code
    assert 'onStepStartedEvent:' in code and 'onStepFinishedEvent:' in code
    assert '<WorkflowPanel' not in code
    assert 'aa-v5-live-work' not in code
    assert '`Agent: ${readableStep' not in code
    assert 'onRawEvent:' in code
    assert 'saved' not in code or 'stepsByTurnRef.current' in code


def test_charts_are_measured_not_synthetic():
    lab = LAB.read_text()
    charts = CHARTS.read_text()
    assert all(n in charts for n in ('77.54', '80.54', '80.67', '83.63', '67.38', '72.25'))
    assert all(n in charts for n in ('87.50', '84.94', '67.46', '90.61', '69.06', '83.65'))
    assert all(n in charts for n in ('22.2', '61.1', '27.8', '204.89', '13.79'))
    assert 'not deployed' in lab
    assert 'retrospectively' in lab
    assert 'ClassifierComparisonChart' in lab and 'RetrievalLatencyChart' in lab
    assert 'ResponsiveContainer' in charts and 'Tooltip' in charts


def test_public_project_facts_use_scoped_counts():
    tree = ast.parse(AGENT.read_text())
    facts = next(node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'PUBLIC_PROJECT_SCOPE' for t in node.targets))
    result = ast.literal_eval(facts.value)
    assert result['cleaned_complaints'] == 6052
    assert result['indexed_unique_recall_campaigns'] == 67
    assert result['train'] == 4235 and result['test'] == 1211
    assert 'no external validation' in result['scope']
    assert 'get_project_metrics' in AGENT.read_text()


def test_css_does_not_override_fonts_or_theme():
    css = CSS.read_text()
    assert 'AutoAssist V6: one activity surface' in css
    assert 'next/font/google' not in css
    assert ':root{' not in css
    assert '@import' not in css
    assert '.aa-v6-live-activity' in css
