"""Remove the technical fallback banner from answer text; UI shows status separately.

This is a narrow, idempotent patch. It never changes the OpenRouter call, safety
prompt, recall guardrails or generation failure handling.
"""
from __future__ import annotations
from pathlib import Path
import shutil

OLD = "**Non-LLM fallback summary (generation was not available).**\\n\\n"


def apply(source: Path) -> str:
    content = source.read_text(encoding="utf-8")
    if OLD not in content:
        if "Non-LLM fallback summary" not in content:
            return "Already polished or a different fallback implementation; unchanged."
        raise RuntimeError("Fallback text has changed; inspect agent/generation.py manually.")
    if content.count(OLD) != 1:
        raise RuntimeError("Unexpected duplicate fallback marker; unchanged.")
    backup = source.with_suffix(".py.before_ui_polish.bak")
    if not backup.exists():
        shutil.copy2(source, backup)
    source.write_text(content.replace(OLD, "", 1), encoding="utf-8")
    return f"Updated {source}; original backed up to {backup}"


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(apply(root / "agent" / "generation.py"))
