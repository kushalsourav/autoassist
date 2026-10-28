"""Compare saved baseline and DistilBERT models on the exact held-out NHTSA split.

Run from repository root:
    python -m ml.evaluate_models --batch-size 4 --max-length 128

Writes class-level reports, confusion matrices, and examples of misclassification.
This is evaluation only: it does NOT retrain either model.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=Path('ml/data/nhtsa_complaints.csv'))
    parser.add_argument('--baseline', type=Path, default=Path('ml/artifacts/baseline.joblib'))
    parser.add_argument('--transformer', type=Path, default=Path('ml/artifacts/distilbert'))
    parser.add_argument('--output', type=Path, default=Path('ml/artifacts/model_comparison'))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=128,
                        help='Use the same truncation length as the fine-tuning run.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()

    data = pd.read_csv(args.csv).dropna(subset=['summary', 'category'])
    data = data.drop_duplicates(subset=['summary']).copy()
    data['summary'] = data['summary'].astype(str)
    data['category'] = data['category'].astype(str)
    _, holdout = train_test_split(
        data, test_size=0.2, stratify=data['category'], random_state=42
    )
    texts = holdout['summary'].tolist()
    actual = holdout['category'].tolist()
    labels = sorted(data['category'].unique().tolist())

    print(f'Evaluating {len(texts)} untouched test records across {len(labels)} categories...')
    baseline = joblib.load(args.baseline)
    baseline_predictions = baseline.predict(texts).tolist()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else (
        'cpu' if args.device == 'auto' else args.device
    )
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was selected but is unavailable. Use --device cpu.')
    tokenizer = AutoTokenizer.from_pretrained(str(args.transformer), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.transformer), local_files_only=True
    ).to(device).eval()
    transformer_predictions = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch = tokenizer(
                texts[start:start + args.batch_size],
                return_tensors='pt', padding=True, truncation=True,
                max_length=args.max_length
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            indices = model(**batch).logits.argmax(dim=-1).cpu().tolist()
            transformer_predictions.extend(model.config.id2label[int(idx)] for idx in indices)
            if (start // args.batch_size) % 50 == 0:
                print(f'  DistilBERT: {min(start + args.batch_size, len(texts))}/{len(texts)}')

    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    error_rows = []
    for name, predictions in [('baseline', baseline_predictions), ('distilbert', transformer_predictions)]:
        report = classification_report(
            actual, predictions, labels=labels, output_dict=True, zero_division=0
        )
        metrics = {
            'accuracy': float(accuracy_score(actual, predictions)),
            'macro_f1': float(f1_score(actual, predictions, labels=labels, average='macro')),
            'classification_report': report,
            'labels': labels,
            'confusion_matrix': confusion_matrix(actual, predictions, labels=labels).tolist(),
        }
        results[name] = metrics
        print(f"\n{name.upper()}: accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f}")
        for category in labels:
            entry = report[category]
            print(f"  {category:<24} F1={entry['f1-score']:.4f} | P={entry['precision']:.4f} | R={entry['recall']:.4f} | n={int(entry['support'])}")
        for text, true_label, predicted in zip(texts, actual, predictions):
            if true_label != predicted:
                error_rows.append({
                    'model': name, 'actual': true_label, 'predicted': predicted,
                    'summary': text[:500]
                })

    results['comparison'] = {
        'test_records': len(texts),
        'macro_f1_delta_distilbert_minus_baseline': (
            results['distilbert']['macro_f1'] - results['baseline']['macro_f1']
        ),
        'accuracy_delta_distilbert_minus_baseline': (
            results['distilbert']['accuracy'] - results['baseline']['accuracy']
        ),
        'seed': 42,
        'max_length': args.max_length,
        'caveat': 'Labels are derived from owner-reported complaint components, not confirmed diagnoses.',
    }
    (args.output / 'evaluation.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    pd.DataFrame(error_rows).to_csv(args.output / 'misclassified_examples.csv', index=False)
    print(f'\nSaved {args.output / "evaluation.json"} and {args.output / "misclassified_examples.csv"}')


if __name__ == '__main__':
    main()
