"""Update one callsite in agent/workflow.py, preserving any other user edits."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / 'agent' / 'workflow.py'
text = path.read_text(encoding='utf-8')
old = "route_intent(state['message'], state.get('history'))"
new = "route_intent(state['message'], state.get('history'), vehicle=state.get('vehicle'))"
if new in text:
    print('Workflow already passes vehicle context; no change needed.')
elif text.count(old) == 1:
    backup = path.with_name('workflow.py.before_intent_fix.bak')
    if backup.exists():
        raise SystemExit(f'Backup exists: {backup}; review it before running again.')
    backup.write_text(text, encoding='utf-8')
    path.write_text(text.replace(old, new, 1), encoding='utf-8')
    print(f'Updated {path.relative_to(root)}; backup: {backup.relative_to(root)}')
else:
    raise SystemExit('Could not safely find expected callsite in agent/workflow.py; no changes made.')
