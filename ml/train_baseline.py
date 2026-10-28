"""Reproducible TF-IDF + Logistic Regression benchmark using real labeled complaints."""
import argparse
import json
from collections import Counter
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=Path('ml/data/nhtsa_complaints.csv'))
    parser.add_argument('--output', type=Path, default=Path('ml/artifacts'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    data = pd.read_csv(args.csv).dropna(subset=['summary', 'category'])
    data = data.drop_duplicates(subset=['summary']).copy()
    data['summary'] = data['summary'].astype(str)
    data['category'] = data['category'].astype(str)
    counts = Counter(data['category'])
    if len(counts) < 2 or min(counts.values()) < 10:
        raise SystemExit(f'Need at least 2 categories with >=10 unique complaints each; got {counts}.')
    development, holdout = train_test_split(
        data, test_size=0.2, stratify=data['category'], random_state=args.seed)
    train, _validation = train_test_split(
        development, test_size=0.125, stratify=development['category'], random_state=args.seed)
    train_x, train_y = train['summary'], train['category']
    test_x, test_y = holdout['summary'], holdout['category']
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000,
                                  strip_accents='unicode', sublinear_tf=True)),
        ('classifier', LogisticRegression(max_iter=1200, class_weight='balanced', random_state=args.seed)),
    ])
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    labels = sorted(counts)
    metrics = {
        'model': 'TF-IDF + LogisticRegression', 'seed': args.seed,
        'n_rows': len(data), 'n_train': len(train_x), 'n_test': len(test_x),
        'label_distribution': dict(counts),
        'accuracy': round(float(accuracy_score(test_y, pred)), 4),
        'macro_f1': round(float(f1_score(test_y, pred, average='macro')), 4),
        'classification_report': classification_report(test_y, pred, output_dict=True, zero_division=0),
        'labels': labels,
        'confusion_matrix': confusion_matrix(test_y, pred, labels=labels).tolist(),
        'caveat': 'Owner-reported component labels are not verified fault diagnoses.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / 'baseline.joblib')
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps({key: metrics[key] for key in ('n_rows', 'n_train', 'n_test', 'accuracy', 'macro_f1')}, indent=2))
    print(f'Saved model and evaluation in {args.output}')


if __name__ == '__main__':
    main()
