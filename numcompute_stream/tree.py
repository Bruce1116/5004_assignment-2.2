from collections import Counter
import numpy as np


class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.X_seen = None
        self.y_seen = None
        self.tree = None

    def partial_fit(self, X_chunk, y_chunk):
        X = np.asarray(X_chunk, dtype=float)
        y = np.asarray(y_chunk)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.X_seen is None:
            self.X_seen = X
            self.y_seen = y
        else:
            self.X_seen = np.vstack([self.X_seen, X])
            self.y_seen = np.concatenate([self.y_seen, y])

        self.tree = self._build_tree(self.X_seen, self.y_seen, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.asarray([self._predict_one(row, self.tree) for row in X])

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1 or depth >= self.max_depth or len(y) < self.min_samples_split:
            return {"label": self._majority_class(y)}

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {"label": self._majority_class(y)}

        left = X[:, feature] <= threshold
        right = ~left
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left], y[left], depth + 1),
            "right": self._build_tree(X[right], y[right], depth + 1),
        }

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_score = float("inf")

        features = list(range(X.shape[1]))
        if self.max_features is not None:
            features = features[: self.max_features]

        for feature in features:
            values = np.unique(X[:, feature])
            for threshold in values:
                left = y[X[:, feature] <= threshold]
                right = y[X[:, feature] > threshold]
                if len(left) == 0 or len(right) == 0:
                    continue
                score = self._split_score(left, right)
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _split_score(self, left, right):
        total = len(left) + len(right)
        return (len(left) / total) * self._impurity(left) + (len(right) / total) * self._impurity(right)

    def _impurity(self, y):
        counts = np.asarray(list(Counter(y).values()), dtype=float)
        probs = counts / np.sum(counts)
        if self.criterion == "entropy":
            return float(-np.sum(probs * np.log2(probs + 1e-9)))
        return float(1 - np.sum(probs ** 2))

    def _majority_class(self, y):
        return Counter(y).most_common(1)[0][0]

    def _predict_one(self, row, node):
        if node is None:
            return 0
        if "label" in node:
            return node["label"]
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_one(row, node["left"])
        return self._predict_one(row, node["right"])

