import unittest

import numpy as np

from numcompute_stream.stats import (
    StreamingStats,
    chunk_histogram,
    chunk_mean,
    chunk_quantiles,
    chunk_variance,
    update_stats,
)


class StatsTests(unittest.TestCase):
    def test_stats_count_update(self):
        stats = StreamingStats()
        stats.update_stats([[1, 2], [3, 4]])
        self.assertEqual(stats.count, 2)

    def test_stats_mean(self):
        stats = StreamingStats()
        stats.update_stats([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(np.allclose(stats.mean, [3, 4]))

    def test_stats_variance(self):
        stats = StreamingStats()
        stats.update_stats([[1], [3], [5]])
        self.assertTrue(np.allclose(stats.variance(), [4]))

    def test_stats_quantiles(self):
        stats = StreamingStats()
        stats.update_stats([[1], [2], [3]])
        self.assertTrue(np.allclose(stats.quantiles((0.5,)), [[2]]))

    def test_stats_histogram(self):
        stats = StreamingStats(bins=2)
        stats.update_stats([[1], [2], [3]])
        self.assertEqual(len(stats.histogram()), 1)

    def test_update_stats_function(self):
        stats = update_stats([[1], [2]])
        self.assertEqual(stats.count, 2)

    def test_chunk_mean(self):
        self.assertTrue(np.allclose(chunk_mean([[1, 2], [3, 4]]), [2, 3]))

    def test_chunk_variance(self):
        self.assertTrue(np.allclose(chunk_variance([[1], [3]]), [1]))

    def test_chunk_quantiles(self):
        self.assertTrue(np.allclose(chunk_quantiles([[1], [3]], (0.5,)), [[2]]))

    def test_chunk_histogram(self):
        hist = chunk_histogram([[1], [2], [3]], bins=2)
        self.assertEqual(len(hist), 1)


if __name__ == "__main__":
    unittest.main()
