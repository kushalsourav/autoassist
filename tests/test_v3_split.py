"""Deterministic preprocessing test without requiring transformers or torch."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml.v3_split import load_v3_splits
from ml.train_multilabel_distilbert_v3 import select_threshold


class V3SplitTests(unittest.TestCase):
    def test_reproducible_split_and_multilabel(self):
        rows = []
        groups = [
            ('ENGINE', 'Engine / Powertrain'),
            ('POWER TRAIN,ENGINE', 'Transmission'),
            ('SERVICE BRAKES', 'Brakes'),
            ('ELECTRICAL SYSTEM', 'Electrical'),
            ('AIR BAGS', 'Airbags / Safety'),
            ('STEERING', 'Steering / Suspension'),
        ]
        for i in range(120):
            raw, primary = groups[i % len(groups)]
            rows.append({'summary': f'Complaint unique text {i} issue',
                         'reported_components': raw, 'category': primary})
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / 'small.csv'
            pd.DataFrame(rows).to_csv(csv, index=False)
            df1, y1, splits1, manifest1 = load_v3_splits(csv, 42)
            df2, y2, splits2, manifest2 = load_v3_splits(csv, 42)
            self.assertEqual(manifest1, manifest2)
            self.assertEqual(len(df1), 120)
            self.assertEqual(y1.shape, (120, 6))
            self.assertTrue((y1.sum(axis=1) > 1).any())
            for name in ('train', 'validation', 'test'):
                self.assertEqual(splits1[name].tolist(), splits2[name].tolist())
            self.assertEqual(len(splits1['train']), 84)
            self.assertEqual(len(splits1['validation']), 12)
            self.assertEqual(len(splits1['test']), 24)

    def test_threshold_selection(self):
        import numpy as np
        gold = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=int)
        probs = np.asarray([[0.8, .2], [.1, .9], [.7, .6]])
        threshold, sweep = select_threshold(probs, gold, ['A', 'B'])
        self.assertEqual(threshold, 0.5)
        self.assertEqual(len(sweep), 7)


if __name__ == '__main__':
    unittest.main()
