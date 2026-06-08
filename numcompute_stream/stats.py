import numpy as np


class StreamingStats:
    """Keep simple statistics for data chunks."""

    def __init__(self, bins=10):
        self.bins = bins
        self.count = 0
        self.mean = None
        self.m2 = None
        self.values = []

    def update_stats(self, X_chunk):
        X = np.asarray(X_chunk, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.mean is None:
            self.mean = np.zeros(X.shape[1])
            self.m2 = np.zeros(X.shape[1])

        for row in X:
            self.count += 1
            diff = row - self.mean
            self.mean += diff / self.count
            diff2 = row - self.mean
            self.m2 += diff * diff2

        self.values.extend(X.tolist())
        return self

    def variance(self):
        if self.count < 2:
            return np.zeros_like(self.mean)
        return self.m2 / (self.count - 1)

    def quantiles(self, q=(0.25, 0.5, 0.75)):
        if not self.values:
            return None
        return np.quantile(np.asarray(self.values), q, axis=0)

    def histogram(self):
        if not self.values:
            return None
        values = np.asarray(self.values)
        output = []
        for i in range(values.shape[1]):
            counts, edges = np.histogram(values[:, i], bins=self.bins)
            output.append((counts, edges))
        return output


def update_stats(X_chunk, stats=None):
    if stats is None:
        stats = StreamingStats()
    return stats.update_stats(X_chunk)


def chunk_mean(X_chunk):
    return np.mean(np.asarray(X_chunk, dtype=float), axis=0)


def chunk_variance(X_chunk):
    return np.var(np.asarray(X_chunk, dtype=float), axis=0)


def chunk_quantiles(X_chunk, q=(0.25, 0.5, 0.75)):
    return np.quantile(np.asarray(X_chunk, dtype=float), q, axis=0)


def chunk_histogram(X_chunk, bins=10):
    X = np.asarray(X_chunk, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return [np.histogram(X[:, i], bins=bins) for i in range(X.shape[1])]

