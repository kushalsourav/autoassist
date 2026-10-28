import unittest
from ml.taxonomy_v3 import components_to_labels_v3, normalize_summary

class TaxonomyV3Tests(unittest.TestCase):
    def test_engine_and_engine_cooling_alias(self):
        self.assertEqual(components_to_labels_v3('ENGINE AND ENGINE COOLING'), ['Engine / Fuel'])

    def test_both_engine_and_powertrain(self):
        self.assertEqual(components_to_labels_v3('POWER TRAIN,ENGINE'),
                         ['Engine / Fuel', 'Powertrain / Drivetrain'])

    def test_electrical_and_engine(self):
        self.assertEqual(components_to_labels_v3('ELECTRICAL SYSTEM,ENGINE AND ENGINE COOLING'),
                         ['Electrical', 'Engine / Fuel'])

    def test_unknown_component_not_forced(self):
        self.assertEqual(components_to_labels_v3('STRUCTURE,UNKNOWN OR OTHER'), [])

    def test_exact_tokens_only(self):
        self.assertEqual(components_to_labels_v3('ENGINEERING,POWER TRAINING'), [])

    def test_component_children(self):
        self.assertEqual(components_to_labels_v3('STEERING: ELECTRIC POWER ASSIST SYSTEM'),
                         ['Steering / Suspension'])

    def test_normalized_duplicate_detection(self):
        self.assertEqual(normalize_summary('Brake-failure!!'), normalize_summary('brake failure'))

if __name__ == '__main__':
    unittest.main()
