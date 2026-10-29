"""Append optional V7 styles while preserving local-font fixes and V4/V6 theme."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
css = ROOT / 'frontend/src/app/globals.css'
extra = ROOT / 'scripts/assets/autoassist-v7-reasoning.css'
marker = '/* V7: Optional provider-supplied reasoning'
if not css.exists():
    raise SystemExit(f'Cannot find {css}; install patch at AutoAssist project root.')
content = css.read_text(encoding='utf-8')
if marker not in content:
    css.write_text(content.rstrip() + '\n\n' + extra.read_text(encoding='utf-8'), encoding='utf-8')
    print('Added V7 reasoning panel styles')
else:
    print('V7 styles already installed; unchanged')
