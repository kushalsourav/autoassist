"""Deterministic, v3-compatible data preparation for controlled experiments.

Reconstructs the same split as ml.train_multilabel_v3 (seed=42), without
changing prior model outputs. Keep the original NHTSA CSV unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from ml.taxonomy_v3 import LABELS_V3, components_to_labels_v3, normalize_summary


def load_v3_splits(csv_path: Path, seed: int = 42):
    df = pd.read_csv(csv_path)
    required = {'summary', 'category', 'reported_components'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing dataset fields: {sorted(missing)}')

    df = df.dropna(subset=sorted(required)).copy()
    df = df[df['summary'].astype(str).str.strip().str.len() > 0].copy()
    df['norm_summary'] = df['summary'].map(normalize_summary)
    df = df[df['norm_summary'].str.len() > 0].copy()
    df = df.drop_duplicates(subset=['norm_summary']).reset_index(drop=True)
    df['labels_v3'] = df['reported_components'].map(components_to_labels_v3)
    df = df[df['labels_v3'].map(len) > 0].reset_index(drop=True)

    mlb = MultiLabelBinarizer(classes=LABELS_V3)
    y = mlb.fit_transform(df['labels_v3']).astype('float32')

    indices = np.arange(len(df))
    develop, test = train_test_split(
        indices, test_size=0.2, stratify=df['category'], random_state=seed)
    train, valid = train_test_split(
        develop, test_size=0.125, stratify=df.iloc[develop]['category'], random_state=seed)

    # One fingerprint for the input and ordering; immutable across experiments.
    sha = hashlib.sha256()
    for row in df.itertuples(index=False):
        sha.update(json.dumps([row.norm_summary, row.category,
                               row.reported_components, row.labels_v3],
                              ensure_ascii=False).encode('utf-8'))
        sha.update(b'\n')

    manifest = {
        'taxonomy': 'v3', 'seed': seed,
        'n_rows': len(df), 'n_train': len(train),
        'n_validation': len(valid), 'n_test': len(test),
        'labels': list(LABELS_V3), 'cleaned_dataset_sha256': sha.hexdigest(),
        'split_strategy': 'legacy-primary-stratified 70/10/20 (identical to v3 baseline)',
        'split_index_sha256': {
            split: hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()
            for split, idx in [('train', train), ('validation', valid), ('test', test)]
        }
    }
    return df, y, {'train': train, 'validation': valid, 'test': test}, manifest


def verify_against_baseline(manifest: dict, baseline_metrics: Path) -> None:
    if not baseline_metrics.exists():
        raise FileNotFoundError(
            f'V3 baseline metrics not found: {baseline_metrics}. '
            'Run python -m ml.train_multilabel_v3 first.')
    v3 = json.loads(baseline_metrics.read_text(encoding='utf-8'))
    expected = {
        'n_rows': v3['rows_after_cleaning'],
        'n_train': v3['n_train'],
        'n_validation': v3['n_validation'],
        'n_test': v3['n_test'],
        'labels': v3['labels'], 'seed': v3['seed'],
    }
    mismatches = {k: (manifest[k], value) for k, value in expected.items()
                  if manifest[k] != value}
    if mismatches:
        raise ValueError(f'Dataset/split does not match v3 baseline: {mismatches}')
