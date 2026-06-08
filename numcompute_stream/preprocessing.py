import numpy as np


class StandardScaler:
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def partial_fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
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
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.mean is None:
            return X
        std = np.sqrt(self.m2 / max(self.count - 1, 1))
        std[std == 0] = 1
        return (X - self.mean) / std

    def fit_transform(self, X, y=None):
        self.partial_fit(X, y)
        return self.transform(X)


class Imputer:
    def __init__(self):
        self.count = None
        self.total = None
        self.fill_values = None

    def partial_fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.count is None:
            self.count = np.zeros(X.shape[1])
            self.total = np.zeros(X.shape[1])

        mask = ~np.isnan(X)
        self.count += np.sum(mask, axis=0)
        self.total += np.nansum(X, axis=0)
        self.fill_values = np.divide(
            self.total,
            np.maximum(self.count, 1),
        )
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fill_values is None:
            return np.nan_to_num(X)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(self.fill_values, inds[1])
        return X

    def fit_transform(self, X, y=None):
        self.partial_fit(X, y)
        return self.transform(X)


class OneHotEncoder:
    def __init__(self):
        self.categories = []

    def partial_fit(self, X, y=None):
        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if not self.categories:
            self.categories = [[] for _ in range(X.shape[1])]

        for col in range(X.shape[1]):
            for value in X[:, col]:
                if value not in self.categories[col]:
                    self.categories[col].append(value)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        rows = []
        for row in X:
            new_row = []
            for col, value in enumerate(row):
                for category in self.categories[col]:
                    new_row.append(1 if value == category else 0)
            rows.append(new_row)
        return np.asarray(rows, dtype=float)

    def fit_transform(self, X, y=None):
        self.partial_fit(X, y)
        return self.transform(X)

