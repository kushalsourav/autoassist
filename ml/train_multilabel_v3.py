"""New independent multi-label experiment with corrected weak-label taxonomy.

- No edits to previously trained models, scripts, or metrics.
- Splits after normalized text deduplication to prevent identical text leakage.
- Optimizes ONE global probability threshold using validation only.
- Evaluates the held-out test split ONCE after selection.

Run: python -m ml.train_multilabel_v3
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from ml.taxonomy_v3 import (LABELS_V3, LEGACY_PRIMARY_NAME_TO_V3,
                            components_to_labels_v3, normalize_summary)
from ml.train_multilabel_baseline import metric_block


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=Path('ml/data/nhtsa_complaints.csv'))
    parser.add_argument('--output', type=Path, default=Path('ml/artifacts/multilabel_v3'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = {'summary', 'category', 'reported_components'}
    missing = required - set(df.columns)
    if missing:
        parser.error(f'Missing columns: {sorted(missing)}')
    raw_count = len(df)
    df = df.dropna(subset=sorted(required)).copy()
    df = df[df['summary'].astype(str).str.strip().str.len() > 0].copy()
    df['norm_summary'] = df['summary'].map(normalize_summary)
    df = df[df['norm_summary'].str.len() > 0].copy()
    normalized_duplicates_removed = int(df.duplicated(subset=['norm_summary']).sum())
    df = df.drop_duplicates(subset=['norm_summary']).reset_index(drop=True)
    df['labels'] = df['reported_components'].map(components_to_labels_v3)
    unknown = df.loc[df['labels'].map(len) == 0]
    if not unknown.empty:
        print(f'WARNING: {len(unknown)} rows remain unmapped; exclude from this experiment.')
        df = df.loc[df['labels'].map(len) > 0].copy().reset_index(drop=True)

    # An old primary label was decided by substring precedence. Diagnose but
    # never silently force that old label into the new multi-label truth set.
    def primary_mismatch(row) -> bool:
        primary = LEGACY_PRIMARY_NAME_TO_V3.get(row['category'])
        return primary is None or primary not in row['labels']
    mismatch = df.loc[df.apply(primary_mismatch, axis=1)]
    if len(df) < 100:
        raise SystemExit('Too few mapped rows to run the experiment')
    support = df['category'].value_counts()
    if support.min() < 5:
        raise SystemExit(f'Insufficient primary-category examples for stratification: {support.to_dict()}')

    mlb = MultiLabelBinarizer(classes=LABELS_V3)
    y = mlb.fit_transform(df['labels'])
    indices = np.arange(len(df))
    develop, test = train_test_split(indices, test_size=0.2,
                                     stratify=df['category'], random_state=args.seed)
    train, valid = train_test_split(develop, test_size=0.125,
                                    stratify=df.iloc[develop]['category'], random_state=args.seed)
    x = df['summary'].astype(str)
    clf = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=30000,
                                   strip_accents='unicode', sublinear_tf=True)),
        ('classifier', OneVsRestClassifier(LogisticRegression(
            max_iter=1200, class_weight='balanced', random_state=args.seed)))
    ])
    clf.fit(x.iloc[train], y[train])
    p_valid = clf.predict_proba(x.iloc[valid])

    # Tune on validation data only; test must remain untouched until selection.
    sweep = []
    for threshold in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        metrics = metric_block(y[valid], (p_valid >= threshold).astype(int), LABELS_V3)
        sweep.append({'threshold': threshold,
                      'validation_macro_f1': metrics['macro_f1'],
                      'validation_micro_f1': metrics['micro_f1'],
                      'validation_subset_accuracy': metrics['subset_accuracy']})
    # If tied, choose the value closest to the conventional 0.5 default.
    best = max(sweep, key=lambda item: (item['validation_macro_f1'], -abs(item['threshold'] - 0.5)))
    threshold = float(best['threshold'])
    validation = metric_block(y[valid], (p_valid >= threshold).astype(int), LABELS_V3)
    p_test = clf.predict_proba(x.iloc[test])
    test_metrics = metric_block(y[test], (p_test >= threshold).astype(int), LABELS_V3)

    report = {
        'task': 'multi-label classification of owner-reported component groups (NOT diagnosis)',
        'experiment_version': 'v3',
        'model': 'TF-IDF + OneVsRest LogisticRegression', 'seed': args.seed,
        'input_rows': raw_count, 'rows_after_cleaning': int(len(df)),
        'normalized_duplicate_rows_removed': normalized_duplicates_removed,
        'unmapped_rows_removed': int(len(unknown)),
        'primary_label_mismatches': int(len(mismatch)),
        'multi_category_records': int((df['labels'].map(len) > 1).sum()),
        'n_train': len(train), 'n_validation': len(valid), 'n_test': len(test),
        'labels': LABELS_V3,
        'label_support': {lab: int(y[:, i].sum()) for i, lab in enumerate(LABELS_V3)},
        'selected_threshold': threshold,
        'validation_threshold_sweep': sweep, 'validation': validation, 'test': test_metrics,
        'limitations': [
            'Owner-reported components are weak labels; no mechanical diagnosis is verified.',
            'POWER TRAIN is modeled as Powertrain / Drivetrain, not exclusively Transmission.',
            'ENGINE AND ENGINE COOLING is now an explicit Engine / Fuel synonym.',
            'Split is stratified by the legacy primary category, not multilabel iterative stratification.',
            'Normalized text deduplicated before splitting; semantic near-duplicates are not audited.',
            'The v3 cohort/taxonomy differs from v2: performance numbers are not a controlled model comparison.',
            'Repeated test-set inspection can bias later experiments; select future changes on validation.',
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': clf, 'classes': LABELS_V3, 'threshold': threshold,
                 'taxonomy': 'v3'}, args.output / 'model.joblib')
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(sweep).to_csv(args.output / 'threshold_validation.csv', index=False)
    if not mismatch.empty:
        cols = [c for c in ['category', 'reported_components', 'summary'] if c in df.columns]
        mismatch[cols].to_csv(args.output / 'primary_label_mismatches.csv', index=False)

    print(f'Input {raw_count} | mapped {len(df)} | normalized duplicates removed {normalized_duplicates_removed}')
    print(f'Unmapped {len(unknown)} | legacy-primary mismatches {len(mismatch)} | multi-label {report["multi_category_records"]}')
    print(f'Train={len(train)} validation={len(valid)} test={len(test)}')
    print(f'Validation-selected threshold={threshold:.2f}')
    print(f'Validation: macro-F1={validation["macro_f1"]:.4f} micro-F1={validation["micro_f1"]:.4f}')
    print(f'Test:       macro-F1={test_metrics["macro_f1"]:.4f} micro-F1={test_metrics["micro_f1"]:.4f} exact-match={test_metrics["subset_accuracy"]:.4f}')
    for lab in LABELS_V3:
        print(f'  {lab:<26} F1={test_metrics["per_label"][lab]["f1-score"]:.4f}')
    print(f'Artifacts saved to {args.output}')
    print('Do not directly compare v3 numbers with the earlier single-label or v2 scores.')


if __name__ == '__main__':
    main()
