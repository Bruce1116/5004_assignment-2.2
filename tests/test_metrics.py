import unittest

import numpy as np

from numcompute_stream.metrics import (
    StreamingClassificationMetrics,
    accuracy,
    auc_score,
    confusion_matrix,
    f1_score,
    precision,
    recall,
)


class MetricsTests(unittest.TestCase):
    def test_accuracy(self):
        self.assertAlmostEqual(accuracy([1, 0, 1], [1, 1, 1]), 2 / 3)

    def test_accuracy_empty(self):
        self.assertEqual(accuracy([], []), 0.0)

    def test_precision(self):
        self.assertAlmostEqual(precision([1, 0, 1], [1, 1, 0]), 0.5)

    def test_recall(self):
        self.assertAlmostEqual(recall([1, 0, 1], [1, 1, 0]), 0.5)

    def test_f1_score(self):
        self.assertAlmostEqual(f1_score([1, 0, 1], [1, 1, 0]), 0.5)

    def test_confusion_matrix_shape(self):
        matrix = confusion_matrix([0, 1, 1], [0, 0, 1])
        self.assertEqual(matrix.shape, (2, 2))

    def test_auc_score(self):
        score = auc_score([0, 1, 0, 1], [0.1, 0.8, 0.3, 0.7])
        self.assertGreater(score, 0.9)

    def test_streaming_metrics_update(self):
        metric = StreamingClassificationMetrics()
        metric.update([1, 0, 1], [1, 0, 0])
        self.assertAlmostEqual(metric.result()["accuracy"], 2 / 3)

    def test_streaming_metrics_reset(self):
        metric = StreamingClassificationMetrics()
        metric.update([1], [1])
        metric.reset()
        self.assertEqual(metric.result()["accuracy"], 0.0)

    def test_streaming_metrics_window(self):
        metric = StreamingClassificationMetrics(window_size=2)
        metric.update([1, 1, 0], [0, 1, 0])
        self.assertEqual(metric.result()["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
