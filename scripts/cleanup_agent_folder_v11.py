"""Safely archive stale agent backups; optionally prune provably unused Python modules.

Dry-run by default. Never touches agent/skills, tracked live/legacy runtime modules,
or files referenced anywhere in Python code. Archives instead of permanent deletion.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / 'agent'
ARCHIVE = ROOT / '.agent-cleanup-archive'
# Never auto-delete files that define either a live or still-supported legacy path.
PROTECTED = {
    'story_agent.py', 'workflow.py', 'generation.py', 'intent.py',
    'nhtsa.py', 'rag.py', 'recall_guardrails.py', 'corpus_retrieval.py',
    '__init__.py',
}
# Offline/test-only candidates. These are removed ONLY when no other Python source
# refers to them; an old test import blocks their removal.
CANDIDATES = {
    'agentic_llm.py', 'agentic_planner.py', 'story_vehicle.py',
    'evidence_verifier_v4.py', 'rerank_experiment.py',
}
IGNORED = {'.git', '.venv', 'venv', 'node_modules', '__pycache__',
           '.agent-cleanup-archive', '.pytest_cache', '.mypy_cache',
           '.next', '.tox', 'dist', 'build'}


def backup(path: Path) -> bool:
    if path.is_symlink():
        return False
    name = path.name.casefold()
    return (name.endswith(('.bak', '.orig', '.old', '.save', '~', '.rej'))
            or any(part in name for part in ('.bak.', '.orig.', '.old.')))


def python_sources():
    for p in ROOT.rglob('*.py'):
        if not any(part in IGNORED for part in p.relative_to(ROOT).parts) and not p.is_symlink():
            yield p


def mentions(source: str, filename: str) -> bool:
    """Conservative static scan; strings/dynamic imports also prevent removal."""
    stem = filename.removesuffix('.py')
    return (f'agent.{stem}' in source
            or f'from agent import {stem}' in source
            or f'from . import {stem}' in source
            or f'from .{stem} import' in source
            or f'import {stem}' in source)


def candidates_not_referenced():
    existing = {n for n in CANDIDATES if (AGENT / n).is_file()}
    # Fixed point: start at modules referenced by protected source or tests/scripts.
    occupied = set()
    for name in existing:
        for p in python_sources():
            if p.resolve() == (AGENT / name).resolve():
                continue
            if p.name in existing and p.parent == AGENT:
                continue  # examine cross-candidate references below
            if mentions(p.read_text(encoding='utf-8', errors='replace'), name):
                occupied.add(name)
                break
    while True:
        next_occupied = set(occupied)
        for name in occupied:
            src = (AGENT / name).read_text(encoding='utf-8', errors='replace')
            next_occupied |= {n for n in existing if mentions(src, n)}
        if next_occupied == occupied:
            break
        occupied = next_occupied
    return sorted(existing - occupied), sorted(occupied)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Archive instead of previewing')
    parser.add_argument('--prune-unused', action='store_true',
                        help='Also archive unreferenced experiment/legacy candidate modules')
    args = parser.parse_args()
    if not AGENT.is_dir():
        parser.error('Run from a project with an agent/ directory')
    backups = [p for p in AGENT.rglob('*') if p.is_file() and
               'skills' not in p.relative_to(AGENT).parts and backup(p)]
    old_cache = [p for p in AGENT.rglob('__pycache__') if p.is_dir() and
                 'skills' not in p.relative_to(AGENT).parts]
    removable, retained = candidates_not_referenced() if args.prune_unused else ([], [])
    print('Agent cleanup:', 'APPLY (reversible archive)' if args.apply else 'DRY RUN')
    for p in sorted(backups):
        print('BACKUP:', p.relative_to(ROOT))
    for p in old_cache:
        print('BYTECODE CACHE:', p.relative_to(ROOT))
    for name in removable:
        print('UNREFERENCED CANDIDATE:', 'agent/' + name)
    for name in retained:
        print('KEEP (imported/referenced):', 'agent/' + name)
    if not args.apply:
        print('No changes made. Rerun with --apply to archive and clean.')
        return
    items = backups + [AGENT / name for name in removable]
    if not items and not old_cache:
        print('Nothing to clean.')
        return
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = ARCHIVE / timestamp
    for p in items:
        target = folder / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.move(str(p), str(target))
        print('ARCHIVED:', p.relative_to(ROOT))
    for p in sorted(old_cache, key=lambda x: len(x.parts), reverse=True):
        if not p.is_symlink() and p.exists():
            shutil.rmtree(p)
            print('REMOVED CACHE:', p.relative_to(ROOT))
    print('Archive:', folder, '(restore by copying files back)')
    print('Skills, legacy analysis API and active runtime modules were not touched.')


if __name__ == '__main__':
    main()
