"""Update *known* V5-V8 source-location assertions after agent/runtime split.

Dry-run by default. Exact-match edits only; unknown tests are never overwritten.
Every changed original is archived outside agent/. The new V11 tests are separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'backend/tests'
COMMON = "Path(__file__).resolve().parents[2]"

REPLACEMENTS = {
    'test_release_v5.py': [
        ("AGENT = ROOT / 'agent/story_agent.py'", "AGENT = ROOT / 'agent/runtime/lifecycle.py'"),
        ("source = AGENT.read_text(encoding='utf8')", "source = '\\n'.join((ROOT / item).read_text(encoding='utf8') for item in ('agent/runtime/constants.py', 'agent/runtime/grounding.py', 'agent/runtime/factory.py'))"),
    ],
    'test_release_v6.py': [
        ("AGENT = ROOT / 'agent/story_agent.py'", "AGENT = ROOT / 'agent/runtime/constants.py'"),
        ("assert 'get_project_metrics' in AGENT.read_text()", "assert 'get_project_metrics' in (ROOT / 'agent/runtime/tools.py').read_text()"),
    ],
    'test_provider_reasoning_v7.py': [
        ("agent = (ROOT / 'agent/story_agent.py').read_text()", "agent = (ROOT / 'agent/runtime/factory.py').read_text()"),
    ],
    'test_story_agent.py': [
        ("/ 'agent/story_agent.py').read_text()", "/ 'agent/runtime/factory.py').read_text()"),
    ],
    'test_create_agent_v8.py': [
        ("SOURCE = (ROOT / 'agent' / 'story_agent.py').read_text(encoding='utf-8')", "SOURCE = '\\n'.join((ROOT / part).read_text(encoding='utf-8') for part in ('agent/runtime/state.py', 'agent/runtime/factory.py', 'agent/runtime/tools.py', 'agent/runtime/lifecycle.py', 'agent/runtime/grounding.py'))"),
        ('TREE = ast.parse(SOURCE)', "TREE = ast.parse((ROOT / 'agent/runtime/tools.py').read_text(encoding='utf-8'))"),
        ("assert 'from langchain.agents import AgentState, create_agent' in SOURCE", "assert 'from langchain.agents import create_agent' in SOURCE and 'class StoryState(AgentState)' in SOURCE"),
    ],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply', action='store_true', help='Archive originals and update matching tests')
    args = p.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for name, edits in REPLACEMENTS.items():
        path = TESTS / name
        if not path.exists():
            print('ABSENT:', name)
            continue
        original = path.read_text(encoding='utf-8')
        updated = original
        for old, new in edits:
            if old in updated:
                updated = updated.replace(old, new, 1)
            elif new in updated:
                continue  # idempotent
            else:
                print('UNKNOWN PATTERN — review manually:', name, repr(old[:65]))
        if updated == original:
            print('UNCHANGED:', name)
            continue
        print(('UPDATE:' if args.apply else 'WOULD UPDATE:'), name)
        if args.apply:
            archive = ROOT / '.agent-cleanup-archive' / (stamp + '-test-upgrades') / 'backend/tests'
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, archive / name)
            path.write_text(updated, encoding='utf-8')
    if not args.apply:
        print('No test changes. Run with --apply after reviewing the preview.')


if __name__ == '__main__':
    main()
