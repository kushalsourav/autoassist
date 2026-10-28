# AutoAssist: multi-label classification experiment (v2)

The original six-category model assigned only one label even when a complaint listed multiple components. In your audit, 1,139/6,060 (18.8%) matched multiple category rules. Rather than delete these complaints, this separate experiment predicts every matching reported component group.

## Apply

From your WSL project root, unzip this patch into the repository. It adds `ml/train_multilabel_baseline.py` and `tests/test_component_multilabel.py` without replacing existing models.

## Run

```bash
cd ~/projects/autoassist
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_component_multilabel.py' -v
python -m ml.train_multilabel_baseline
```

Read `ml/artifacts/multilabel_v2/metrics.json` and share the console output.

## Interpretation

- Macro-F1 averages per-label F1; micro-F1 pools label decisions.
- Subset accuracy requires an exact match across all six labels and is stricter than per-label accuracy.
- Multi-label metrics are **not directly comparable** with the old single-label baseline (0.7882) or DistilBERT (0.7861).
- Do not use this preliminary test set for repeated hyperparameter selection. Tune on validation, then freeze the model and evaluate a fresh holdout for final reporting.
- Next, audit taxonomy names (particularly POWER TRAIN -> Transmission), near duplicates and cross-vehicle generalization. Only then decide whether to fine-tune a multi-label transformer or deploy.

This is an **experiment**; don't switch the production FastAPI classifier to the new output until the UX and inference contracts have been updated to support multiple categories.
