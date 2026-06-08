import numpy as np

from .tree import DecisionTreeClassifier


class EnsembleClassifier:
    def __init__(self, n_estimators=5, max_depth=5, sample_ratio=0.8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sample_ratio = sample_ratio
        self.trees = [
            DecisionTreeClassifier(max_depth=max_depth)
            for _ in range(n_estimators)
        ]

    def partial_fit(self, X_chunk, y_chunk):
        X = np.asarray(X_chunk, dtype=float)
        y = np.asarray(y_chunk)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        size = max(1, int(len(X) * self.sample_ratio))
        for tree in self.trees:
            # train each tree on a bootstrap sample
            choices = np.random.choice(len(X), size=size, replace=True)
            tree.partial_fit(X[choices], y[choices])
        return self

    def predict(self, X):
        predictions = np.asarray([tree.predict(X) for tree in self.trees])
        output = []
        for col in predictions.T:
            values, counts = np.unique(col, return_counts=True)
            output.append(values[np.argmax(counts)])
        return np.asarray(output)

