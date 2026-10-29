"""Append AutoAssist V6 CSS without replacing V4/V5 typography or global styles.

Run after extraction: python -m scripts.apply_release_v6
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / 'frontend/src/app/globals.css'
ASSET = ROOT / 'scripts/assets/autoassist-v6.css'
MARKER = 'AutoAssist V6: one activity surface'


def main() -> None:
    if not CSS.exists() or not ASSET.exists():
        raise SystemExit('Missing AutoAssist V6 CSS files; extract the ZIP at the project root.')
    content = CSS.read_text(encoding='utf8')
    if MARKER in content:
        print('V6 styling already applied. No changes.')
        return
    if 'AutoAssist release V5: evidence hierarchy' not in content:
        raise SystemExit('V5 stylesheet is missing. Apply V5 before V6 to avoid styling regressions.')
    CSS.write_text(content.rstrip() + '\n\n' + ASSET.read_text(encoding='utf8').rstrip() + '\n', encoding='utf8')
    print('V6 CSS installed. Existing V4 theme and local Google Font setup preserved.')


if __name__ == '__main__':
    main()
