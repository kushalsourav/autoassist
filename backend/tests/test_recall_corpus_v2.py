from scripts.build_recall_corpus_v2 import (
    merge_campaigns, normalize_campaign, normalize_vehicle, vehicle_documents,
)
import unittest


class RecallCorpusV2Tests(unittest.TestCase):
    def setUp(self):
        self.a = {'make': 'Honda', 'model': 'Accord', 'year': 2018}
        self.b = {'make': 'Honda', 'model': 'Civic', 'year': 2018}
        self.c = {'make': 'Toyota', 'model': 'Camry', 'year': 2019}
        self.one = {'id': '20V771000', 'component': 'ELECTRICAL SYSTEM',
                    'text': 'Body control module and windshield wipers',
                    'url': 'https://api.nhtsa.gov/recalls/recallsByVehicle'}
        self.two = {'id': '18V629000', 'component': 'BACK OVER PREVENTION',
                    'text': 'Backup camera display malfunction', 'url': self.one['url']}

    def test_same_campaign_merges_vehicle_memberships(self):
        docs = merge_campaigns([(self.a, [self.one, self.two]), (self.b, [self.one])])
        self.assertEqual(len(docs), 2)
        self.assertEqual(len(next(d for d in docs if d['id'] == '20V771000')['vehicles']), 2)

    def test_deduplicate_same_campaign_in_same_vehicle(self):
        docs = merge_campaigns([(self.a, [self.one, self.one])])
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(docs[0]['vehicles']), 1)

    def test_strict_vehicle_filter(self):
        docs = merge_campaigns([(self.a, [self.one, self.two]), (self.b, [self.one])])
        self.assertEqual([d['id'] for d in vehicle_documents(docs, 'honda', 'accord', 2018)],
                         ['18V629000', '20V771000'])
        self.assertEqual([d['id'] for d in vehicle_documents(docs, 'Honda', 'Civic', 2018)],
                         ['20V771000'])
        self.assertEqual(vehicle_documents(docs, **self.c), [])

    def test_invalid_vehicle_rejected(self):
        with self.assertRaises(ValueError):
            normalize_vehicle({'make': 'Honda', 'model': '', 'year': 2018})
        with self.assertRaises(ValueError):
            normalize_vehicle({'make': 'Honda', 'model': 'Accord', 'year': '2018'})

    def test_bad_campaign_id_excluded(self):
        self.assertIsNone(normalize_campaign({'id': 'Unknown recall', 'text': 'abc'}))


if __name__ == '__main__':
    unittest.main()
