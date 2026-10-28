"""Optional local GTX 1650 experiment: fine-tune DistilBERT with small batches.

Install from README only when ready; artifacts are NOT committed. Evaluate on a
fixed test split for side-by-side comparison with the classical baseline.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def main():
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=Path('ml/data/nhtsa_complaints.csv'))
    parser.add_argument('--output', type=Path, default=Path('ml/artifacts/distilbert'))
    parser.add_argument('--epochs', type=float, default=2.0)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--gradient-accumulation', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=192)
    args = parser.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=['summary', 'category']).drop_duplicates('summary')
    labels = sorted(df['category'].unique().tolist())
    if len(labels) < 2 or df['category'].value_counts().min() < 10:
        raise SystemExit('Insufficient data for stratified training: >=5 complaints per category required.')
    label2id = {v: i for i, v in enumerate(labels)}
    df['label'] = df['category'].map(label2id)
    train, holdout = train_test_split(df, test_size=.2, random_state=42, stratify=df['category'])
    train, valid = train_test_split(train, test_size=.125, random_state=42, stratify=train['category'])
    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

    def to_dataset(frame):
        data = Dataset.from_dict({'text': frame['summary'].tolist(), 'label': frame['label'].tolist()})
        return data.map(lambda batch: tokenizer(batch['text'], truncation=True, max_length=args.max_length), batched=True)

    train_ds, valid_ds, test_ds = map(to_dataset, (train, valid, holdout))
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert/distilbert-base-uncased', num_labels=len(labels),
        id2label={i: value for i, value in enumerate(labels)}, label2id=label2id)
    model.gradient_checkpointing_enable()  # Reduce activation memory on 4 GB VRAM

    def metrics(eval_pred):
        logits, gold = eval_pred
        prediction = np.argmax(logits, axis=-1)
        return {'accuracy': float(accuracy_score(gold, prediction)),
                'macro_f1': float(f1_score(gold, prediction, average='macro'))}

    args.output.mkdir(parents=True, exist_ok=True)
    training = TrainingArguments(
        output_dir=str(args.output / 'checkpoints'), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation, learning_rate=2e-5,
        weight_decay=.01, eval_strategy='epoch', save_strategy='epoch',
        load_best_model_at_end=True, metric_for_best_model='macro_f1', greater_is_better=True,
        fp16=torch.cuda.is_available(), seed=42, report_to='none',
    )
    trainer = Trainer(model=model, args=training, train_dataset=train_ds, eval_dataset=valid_ds,
                      processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=metrics)
    trainer.train()
    result = trainer.evaluate(test_ds)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    (args.output / 'test_metrics.json').write_text(json.dumps({
        'test_metrics': result, 'labels': labels, 'seed': 42,
        'n_train': len(train), 'n_valid': len(valid), 'n_test': len(holdout),
        'caveat': 'NHTSA complaint component labels are noisy proxy labels.',
    }, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
