import unittest

from numcompute_stream.ensemble import EnsembleClassifier


class EnsembleTests(unittest.TestCase):
    def test_ensemble_partial_fit(self):
        model = EnsembleClassifier(n_estimators=3, max_depth=2)
        returned = model.partial_fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        self.assertIs(returned, model)

    def test_ensemble_predict_length(self):
        model = EnsembleClassifier(n_estimators=3, max_depth=2)
        model.partial_fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        self.assertEqual(len(model.predict([[0], [3]])), 2)

    def test_ensemble_has_expected_number_of_trees(self):
        model = EnsembleClassifier(n_estimators=4)
        self.assertEqual(len(model.trees), 4)


if __name__ == "__main__":
    unittest.main()
