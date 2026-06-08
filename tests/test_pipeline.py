import unittest

import numpy as np

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


class PipelineTests(unittest.TestCase):
    def make_pipeline(self):
        return Pipeline([
            ("imputer", Imputer()),
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=2)),
        ])

    def test_pipeline_partial_fit_works(self):
        pipe = self.make_pipeline()
        pipe.partial_fit([[1.0], [2.0], [3.0]], [0, 0, 1])
        self.assertIsNotNone(pipe.steps[-1][1].tree)

    def test_pipeline_predict_length(self):
        pipe = self.make_pipeline()
        pipe.partial_fit([[1.0], [2.0], [3.0]], [0, 0, 1])
        pred = pipe.predict([[1.5], [2.5]])
        self.assertEqual(len(pred), 2)

    def test_pipeline_fit_chunk(self):
        pipe = self.make_pipeline()
        returned = pipe.fit_chunk([[1.0], [2.0]], [0, 1])
        self.assertIs(returned, pipe)

    def test_pipeline_score_chunk(self):
        pipe = self.make_pipeline()
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1])
        pipe.partial_fit(X, y)
        self.assertGreaterEqual(pipe.score_chunk(X, y), 0.0)


if __name__ == "__main__":
    unittest.main()
