"""Read-only NHTSA label quality audit; does NOT modify datasets or models.

From repository root:
    python -m ml.audit_component_labels

Outputs are LOCAL and can include real public complaint text; review before sharing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import re

import pandas as pd

from ml.prepare_nhtsa import categorize
from ml.train_multilabel_baseline import components_to_labels


def fingerprint(value: str) -> str:
    """Normalize punctuation/spacing for a conservative duplicate check."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", str(value).casefold())).strip()


def audit(csv_path: Path, output_dir: Path, example_count: int = 15) -> dict:
    df = pd.read_csv(csv_path)
    required = {"summary", "category", "reported_components"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    df = df.dropna(subset=sorted(required)).copy()
    # Consistent with original modeling experiments: exact summary deduplication.
    original_rows = len(df)
    df = df.drop_duplicates(subset=["summary"]).reset_index(drop=True)

    df["candidate_labels"] = df["reported_components"].map(components_to_labels)
    df["legacy_category_from_rules"] = df["reported_components"].map(categorize)
    df["normalized_summary"] = df["summary"].map(fingerprint)
    df["label_count"] = df["candidate_labels"].map(len)

    unmapped = df[df["label_count"] == 0].copy()
    mapped = df[df["label_count"] > 0].copy()
    mismatch = mapped[mapped.apply(lambda r: r["category"] not in r["candidate_labels"], axis=1)]
    legacy_mismatch = df[df["category"] != df["legacy_category_from_rules"]]

    component_freq = Counter()
    for component_string in unmapped["reported_components"]:
        for value in str(component_string).split(","):
            token = value.strip().upper()
            if token:
                component_freq[token] += 1

    cooccurrences = Counter()
    for labels in mapped["candidate_labels"]:
        for pair in combinations(sorted(labels), 2):
            cooccurrences[pair] += 1

    normalized_duplicate_groups = (
        df.groupby("normalized_summary", dropna=False)
        .size()
        .loc[lambda s: s > 1]
    )
    report = {
        "task": "read-only label taxonomy audit (not model evaluation)",
        "input": str(csv_path),
        "rows_before_exact_dedup": original_rows,
        "rows_after_exact_dedup": len(df),
        "mapped_rows": len(mapped),
        "unmapped_rows": len(unmapped),
        "multi_label_rows": int((df["label_count"] > 1).sum()),
        "original_primary_category_not_in_exact_labels": len(mismatch),
        "original_primary_category_mismatches_legacy_mapping": len(legacy_mismatch),
        "unmapped_primary_category_distribution": {str(k): int(v) for k, v in unmapped["category"].value_counts().items()},
        "unmapped_top_components": [{"component": k, "rows": v} for k, v in component_freq.most_common(30)],
        "top_label_pairs": [{"labels": list(pair), "rows": int(count)} for pair, count in cooccurrences.most_common(12)],
        "normalized_text_duplicate_groups": int(len(normalized_duplicate_groups)),
        "normalized_text_duplicate_extra_rows": int(sum(normalized_duplicate_groups - 1)),
        "cautions": [
            "Reported components are weak labels, not confirmed fault diagnoses.",
            "Exact component matching may exclude records included by the original substring mapper.",
            "Normalized duplicate detection is conservative, not semantic near-duplicate detection.",
            "POWER TRAIN and Transmission are not interchangeable; consider taxonomy review.",
            "Do not optimize or select models based on repeated test-set inspection.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    sample_columns = [c for c in ["make", "model", "year", "category", "reported_components", "summary"] if c in df.columns]
    unmapped[sample_columns].head(example_count).to_csv(output_dir / "unmapped_examples.csv", index=False)
    mismatch[sample_columns].head(example_count).to_csv(output_dir / "label_disagreements.csv", index=False)

    print(f"Rows: {len(df)} | mapped: {len(mapped)} | unmapped: {len(unmapped)} | multi-label: {report['multi_label_rows']}")
    print(f"Original primary category missing from exact label set: {len(mismatch)}")
    print(f"Original primary category disagrees with old mapper: {len(legacy_mismatch)}")
    print(f"Normalized text duplicates: {report['normalized_text_duplicate_extra_rows']} extra rows")
    print("\nTop unmatched component strings:")
    for obj in report["unmapped_top_components"][:12]:
        print(f"  {obj['component']:<48} {obj['rows']}")
    print("\nTop co-occurring category pairs:")
    for obj in report["top_label_pairs"][:8]:
        print(f"  {' + '.join(obj['labels']):<52} {obj['rows']}")
    print(f"\nSaved read-only audit outputs to {output_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("ml/data/nhtsa_complaints.csv"))
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/label_audit"))
    parser.add_argument("--examples", type=int, default=15)
    args = parser.parse_args()
    audit(args.csv, args.output, args.examples)


if __name__ == "__main__":
    main()
