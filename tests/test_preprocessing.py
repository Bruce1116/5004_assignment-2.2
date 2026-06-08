import unittest

import numpy as np

from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler


class PreprocessingTests(unittest.TestCase):
    def test_imputer_fills_nan(self):
        imputer = Imputer()
        X = np.array([[1.0], [np.nan], [3.0]])
        out = imputer.fit_transform(X)
        self.assertFalse(np.isnan(out).any())

    def test_imputer_updates_fill_value(self):
        imputer = Imputer()
        imputer.partial_fit([[1.0], [3.0]])
        self.assertTrue(np.allclose(imputer.fill_values, [2.0]))

    def test_scaler_partial_fit(self):
        scaler = StandardScaler()
        scaler.partial_fit([[1, 2], [3, 4]])
        self.assertEqual(scaler.count, 2)

    def test_scaler_transform_shape(self):
        scaler = StandardScaler()
        scaler.partial_fit([[1, 2], [3, 4]])
        self.assertEqual(scaler.transform([[1, 2]]).shape, (1, 2))

    def test_scaler_fit_transform(self):
        scaler = StandardScaler()
        out = scaler.fit_transform([[1], [2], [3]])
        self.assertEqual(out.shape, (3, 1))

    def test_one_hot_encoder_detects_categories(self):
        encoder = OneHotEncoder()
        encoder.partial_fit([["red"], ["blue"]])
        self.assertEqual(encoder.categories[0], ["red", "blue"])

    def test_one_hot_encoder_detects_new_categories(self):
        encoder = OneHotEncoder()
        encoder.partial_fit([["red"]])
        encoder.partial_fit([["green"]])
        self.assertEqual(encoder.categories[0], ["red", "green"])

    def test_one_hot_encoder_transform_shape(self):
        encoder = OneHotEncoder()
        encoder.partial_fit([["a"], ["b"]])
        self.assertEqual(encoder.transform([["a"]]).shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
