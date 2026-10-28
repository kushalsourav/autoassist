"""Fine-tune DistilBERT for multi-label NHTSA component classification.

Runs on the same v3 cohort/split as TF-IDF+OvR baseline. Validation chooses
best epoch and a global sigmoid threshold. Do not tune using test metrics.

Usage:
  python -m ml.train_multilabel_distilbert_v3 --prepare-only
  python -m ml.train_multilabel_distilbert_v3 --batch-size 1 \\
      --gradient-accumulation 16 --max-length 128 --epochs 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, hamming_loss)
from ml.v3_split import load_v3_splits, verify_against_baseline


def probabilities(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))


def metric_block(truth: np.ndarray, predicted: np.ndarray, labels: list[str]):
    truth = np.asarray(truth).astype(int)
    predicted = np.asarray(predicted).astype(int)
    return {
        'macro_f1': float(f1_score(truth, predicted, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(truth, predicted, average='micro', zero_division=0)),
        'subset_accuracy': float(accuracy_score(truth, predicted)),
        'hamming_loss': float(hamming_loss(truth, predicted)),
        'micro_precision': float(precision_score(truth, predicted, average='micro', zero_division=0)),
        'micro_recall': float(recall_score(truth, predicted, average='micro', zero_division=0)),
        'per_label': {
            label: {
                'f1': float(f1_score(truth[:, i], predicted[:, i], zero_division=0)),
                'precision': float(precision_score(truth[:, i], predicted[:, i], zero_division=0)),
                'recall': float(recall_score(truth[:, i], predicted[:, i], zero_division=0)),
                'support': int(truth[:, i].sum()),
            }
            for i, label in enumerate(labels)
        },
    }


def select_threshold(validation_probs, validation_truth, labels):
    sweep = []
    for threshold in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        metrics = metric_block(validation_truth,
                               (validation_probs >= threshold).astype(int), labels)
        sweep.append({
            'threshold': threshold,
            'macro_f1': metrics['macro_f1'],
            'micro_f1': metrics['micro_f1'],
            'subset_accuracy': metrics['subset_accuracy'],
        })
    best = max(sweep, key=lambda x: (x['macro_f1'], -abs(x['threshold'] - 0.50)))
    return best['threshold'], sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=Path('ml/data/nhtsa_complaints.csv'))
    parser.add_argument('--v3-metrics', type=Path,
                        default=Path('ml/artifacts/multilabel_v3/metrics.json'))
    parser.add_argument('--output', type=Path,
                        default=Path('ml/artifacts/multilabel_distilbert_v3'))
    parser.add_argument('--epochs', type=float, default=2.0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()

    df, y, splits, manifest = load_v3_splits(args.csv, args.seed)
    verify_against_baseline(manifest, args.v3_metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'split_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print('Verified identical v3 cohort, label order, split seed and split sizes.')
    print(f"Rows={manifest['n_rows']} train={manifest['n_train']} "
          f"validation={manifest['n_validation']} test={manifest['n_test']}")
    print('Fingerprint:', manifest['cleaned_dataset_sha256'])
    if args.prepare_only:
        return

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              Trainer, TrainingArguments, set_seed)

    set_seed(args.seed)
    base_model = 'distilbert/distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    labels = manifest['labels']

    def as_hf(indices):
        texts = df.iloc[indices]['summary'].astype(str).tolist()
        arrays = y[indices].astype('float32').tolist()
        dataset = Dataset.from_dict({'text': texts, 'labels': arrays})
        return dataset.map(
            lambda batch: tokenizer(batch['text'], truncation=True,
                                    max_length=args.max_length),
            batched=True, remove_columns=['text'])

    train_ds = as_hf(splits['train'])
    val_ds = as_hf(splits['validation'])
    test_ds = as_hf(splits['test'])

    def collator(features):
        gold = torch.tensor([f['labels'] for f in features], dtype=torch.float32)
        features = [{k: v for k, v in f.items() if k != 'labels'} for f in features]
        batch = tokenizer.pad(features, padding=True, return_tensors='pt')
        batch['labels'] = gold
        return batch

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(labels), problem_type='multi_label_classification',
        label2id={v: i for i, v in enumerate(labels)},
        id2label={i: v for i, v in enumerate(labels)})
    model.gradient_checkpointing_enable()

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        predicted = (probabilities(logits) >= 0.5).astype(int)
        truth = np.asarray(eval_pred.label_ids).astype(int)
        return {'macro_f1': float(f1_score(truth, predicted,
                                           average='macro', zero_division=0)),
                'micro_f1': float(f1_score(truth, predicted,
                                           average='micro', zero_division=0))}

    training_args = TrainingArguments(
        output_dir=str(args.output / 'checkpoints'),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, weight_decay=0.01,
        eval_strategy='epoch', save_strategy='epoch',
        load_best_model_at_end=True, metric_for_best_model='macro_f1',
        greater_is_better=True, save_total_limit=1,
        fp16=torch.cuda.is_available(), seed=args.seed,
        report_to='none', logging_steps=50,
    )
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      processing_class=tokenizer, data_collator=collator,
                      compute_metrics=compute_metrics)
    trainer.train()
    pred_val = trainer.predict(val_ds)
    logits_val = pred_val.predictions
    if isinstance(logits_val, tuple):
        logits_val = logits_val[0]
    val_probs = probabilities(logits_val)
    threshold, sweep = select_threshold(val_probs, y[splits['validation']], labels)
    val_metrics = metric_block(y[splits['validation']],
                               (val_probs >= threshold).astype(int), labels)
    print(f'Validation-selected threshold: {threshold:.2f} ' 
          f'(macro-F1={val_metrics["macro_f1"]:.4f})')

    # Test is evaluated only after the model checkpoint and threshold are selected.
    pred_test = trainer.predict(test_ds)
    logits_test = pred_test.predictions
    if isinstance(logits_test, tuple):
        logits_test = logits_test[0]
    test_probs = probabilities(logits_test)
    test_metrics = metric_block(y[splits['test']],
                                (test_probs >= threshold).astype(int), labels)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    report = {
        'task': 'multi-label classification of reported components, NOT diagnosis',
        'architecture': 'DistilBERT', 'taxonomy': 'v3', 'seed': args.seed,
        'model_source': base_model, 'labels': labels,
        'n_train': len(splits['train']), 'n_validation': len(splits['validation']),
        'n_test': len(splits['test']), 'selected_threshold': threshold,
        'epochs': args.epochs, 'batch_size': args.batch_size,
        'gradient_accumulation': args.gradient_accumulation,
        'max_length': args.max_length, 'learning_rate': args.learning_rate,
        'validation_threshold_sweep': sweep, 'validation': val_metrics,
        'test': test_metrics,
        'limitations': [
            'Owner-reported components are weak labels, not verified diagnoses.',
            'The V3 test set has been viewed in prior experiments. Treat comparisons as exploratory; '
            'use an independent new dataset for final external validation.',
            'Near-duplicate complaints have not been audited.',
        ],
    }
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Validation macro-F1:', round(val_metrics['macro_f1'], 4))
    print('Test macro-F1:', round(test_metrics['macro_f1'], 4))
    print('Test micro-F1:', round(test_metrics['micro_f1'], 4))
    print('Test exact-match accuracy:', round(test_metrics['subset_accuracy'], 4))
    print('Saved model and metrics in:', args.output)


if __name__ == '__main__':
    main()
