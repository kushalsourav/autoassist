import unittest
from ml.train_multilabel_baseline import components_to_labels


class ComponentMappingTests(unittest.TestCase):
    def test_engine_and_power_train_are_both_kept(self):
        self.assertEqual(
            components_to_labels('POWER TRAIN,ENGINE'),
            ['Engine / Powertrain', 'Transmission'],
        )

    def test_electrical_engine(self):
        self.assertEqual(
            components_to_labels('ELECTRICAL SYSTEM,ENGINE,FUEL/PROPULSION SYSTEM'),
            ['Electrical', 'Engine / Powertrain'],
        )

    def test_prefix_child_component(self):
        self.assertEqual(components_to_labels('POWER TRAIN: AUTOMATIC TRANSMISSION'),
                         ['Transmission'])

    def test_no_partial_word_match(self):
        self.assertEqual(components_to_labels('AN ENGINEER COMPLAINT'), [])


if __name__ == '__main__':
    unittest.main()
