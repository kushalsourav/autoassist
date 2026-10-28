"""Independent multi-label experiment on NHTSA owner-reported component groups.

This experiment DOES NOT replace or alter the original single-label dataset, model,
or test scores. Metrics are not comparable with the old single-label macro-F1.

Usage:
    python -m ml.train_multilabel_baseline
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, hamming_loss,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from ml.prepare_nhtsa import COMPONENT_MAP


def components_to_labels(raw: str) -> list[str]:
    """Map comma-separated NHTSA components to all applicable groups.

    Match exact component names or subcomponents after ':'; avoid general substring
    matching. This is still weak supervision, NOT verified fault diagnoses.
    """
    parts = [re.sub(r'\s+', ' ', s.strip().upper()) for s in str(raw).split(',')]
    matches: set[str] = set()
    for component in parts:
        for tokens, category in COMPONENT_MAP:
            if any(component == token or component.startswith(token + ':')
                   or component.startswith(token + ' :') for token in tokens):
                matches.add(category)
    return sorted(matches)


def metric_block(actual: np.ndarray, predicted: np.ndarray, labels: list[str]) -> dict:
    per_label = classification_report(
        actual, predicted, target_names=labels, output_dict=True, zero_division=0
    )
    return {
        'macro_f1': float(f1_score(actual, predicted, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(actual, predicted, average='micro', zero_division=0)),
        'sample_f1': float(f1_score(actual, predicted, average='samples', zero_division=0)),
        'micro_precision': float(precision_score(actual, predicted, average='micro', zero_division=0)),
        'micro_recall': float(recall_score(actual, predicted, average='micro', zero_division=0)),
        'subset_accuracy': float(accuracy_score(actual, predicted)),
        'hamming_loss': float(hamming_loss(actual, predicted)),
        'per_label': {lab: per_label[lab] for lab in labels},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=Path('ml/data/nhtsa_complaints.csv'))
    parser.add_argument('--output', type=Path, default=Path('ml/artifacts/multilabel_v2'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Fixed threshold; do not optimize against the test set')
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error('--threshold must be strictly between 0 and 1')

    df = pd.read_csv(args.csv)
    required = {'summary', 'category', 'reported_components'}
    if not required.issubset(df.columns):
        raise SystemExit(f'Missing required CSV columns: {sorted(required - set(df.columns))}')
    df = df.dropna(subset=['summary', 'category', 'reported_components']).copy()
    df = df.drop_duplicates('summary').reset_index(drop=True)
    df['labels'] = df['reported_components'].map(components_to_labels)
    unmapped = int((df['labels'].map(len) == 0).sum())
    if unmapped:
        print(f'WARNING: {unmapped} complaints did not match the exact-component rules; excluding.')
        df = df[df['labels'].map(len) > 0].reset_index(drop=True)
    if len(df) < 100:
        raise SystemExit('Insufficient mapped complaints; review component matching rules.')

    labels = sorted({category for _, category in COMPONENT_MAP})
    mlb = MultiLabelBinarizer(classes=labels)
    y = mlb.fit_transform(df['labels'])
    x = df['summary'].astype(str)

    # Same primary-label stratification as the existing models. This is not a
    # fully multilabel-stratified split; call that limitation out in reporting.
    indices = np.arange(len(df))
    trainval, test = train_test_split(indices, test_size=0.2, random_state=args.seed,
                                      stratify=df['category'])
    train, valid = train_test_split(trainval, test_size=0.125,
                                    random_state=args.seed,
                                    stratify=df.loc[trainval, 'category'])

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=30000,
                                   strip_accents='unicode', sublinear_tf=True)),
        ('classifier', OneVsRestClassifier(LogisticRegression(
            max_iter=1200, class_weight='balanced', random_state=args.seed))),
    ])
    pipeline.fit(x.iloc[train], y[train])

    def infer(split: np.ndarray) -> np.ndarray:
        proba = pipeline.predict_proba(x.iloc[split])
        return (proba >= args.threshold).astype(int)

    validation_results = metric_block(y[valid], infer(valid), labels)
    test_predictions = infer(test)
    test_results = metric_block(y[test], test_predictions, labels)
    results = {
        'task': 'multi-label component-group classification (not diagnosis)',
        'model': 'TF-IDF + OneVsRest LogisticRegression',
        'seed': args.seed,
        'threshold': args.threshold,
        'rows': int(len(df)),
        'unmapped_removed': unmapped,
        'multi_category_records': int((df['labels'].map(len) > 1).sum()),
        'multi_category_fraction': float((df['labels'].map(len) > 1).mean()),
        'n_train': len(train), 'n_validation': len(valid), 'n_test': len(test),
        'labels': labels,
        'label_support': {label: int(y[:, i].sum()) for i, label in enumerate(labels)},
        'validation': validation_results,
        'test': test_results,
        'caveats': [
            'NHTSA owner-reported components are weak labels, not confirmed diagnoses.',
            'POWER TRAIN is grouped as Transmission in the original taxonomy; the name is an approximation.',
            'Split is stratified by original primary category, not iterative multilabel stratification.',
            'This is a different task from the existing single-label benchmark: do not compare F1 values directly.',
            'Exact duplicate texts removed; near-duplicate and cross-vehicle leakage still require auditing.',
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': pipeline, 'classes': labels, 'threshold': args.threshold},
                args.output / 'model.joblib')
    (args.output / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f"Mapped {len(df)} records; {results['multi_category_records']} have multiple labels")
    print(f"Split: train={len(train)}, valid={len(valid)}, test={len(test)}")
    for split_name, m in [('validation', validation_results), ('test', test_results)]:
        print(f"{split_name}: macro-F1={m['macro_f1']:.4f}, micro-F1={m['micro_f1']:.4f}, "
              f"subset accuracy={m['subset_accuracy']:.4f}")
        for label in labels:
            print(f"  {label:<24} F1={m['per_label'][label]['f1-score']:.4f}")
    print(f'Saved {args.output / "metrics.json"} and {args.output / "model.joblib"}')
    print('These results are NOT directly comparable to the previous single-label F1.')


if __name__ == '__main__':
    main()
