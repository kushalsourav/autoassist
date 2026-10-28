# Label quality audit — AutoAssist

Purpose: Audit why the multi-label baseline excluded 319 rows. It does not modify the original data, train new models, or change FastAPI.

## Install into existing WSL project

```bash
cd ~/projects/autoassist
unzip -o /mnt/d/autoassist-label-audit-patch.zip -d .
source .venv/bin/activate
python -m ml.audit_component_labels
```

## Outputs

- `ml/artifacts/label_audit/audit.json` — summary counts and component frequencies.
- `ml/artifacts/label_audit/unmapped_examples.csv` — sample of raw unmatched components, for local review.
- `ml/artifacts/label_audit/label_disagreements.csv` — rows where old category disagrees with all new labels.

The CSV examples include public complaint text; review before sharing publicly.

This audit helps decide whether to repair component matching, rename the `POWER TRAIN` category, and/or create a fresh evaluation split. It cannot establish diagnosis correctness.
