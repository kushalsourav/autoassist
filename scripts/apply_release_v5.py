"""Idempotently add AutoAssist V5 component styling after the existing V4 CSS.

This intentionally does not replace globals.css, layout.tsx or local font setup.
Run from the project root: python -m scripts.apply_release_v5
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / 'frontend/src/app/globals.css'
PATCH = ROOT / 'scripts/assets/autoassist-v5.css'
MARKER = 'AutoAssist release V5: evidence hierarchy'


def main() -> None:
    if not CSS.exists() or not PATCH.exists():
        raise SystemExit('Missing CSS or V5 patch asset; run this from a populated AutoAssist repository.')
    existing = CSS.read_text(encoding='utf8')
    if MARKER in existing:
        print('V5 CSS is already installed; nothing changed.')
        return
    if 'AutoAssist · Material-inspired visual system V4' not in existing:
        raise SystemExit('Material UI V4 stylesheet not found; apply the previous V4 patch before V5.')
    content = PATCH.read_text(encoding='utf8')
    CSS.write_text(existing.rstrip() + '\n\n' + content + '\n', encoding='utf8')
    print('V5 CSS appended; existing Google Font variables and V4 theme kept intact.')


if __name__ == '__main__':
    main()
