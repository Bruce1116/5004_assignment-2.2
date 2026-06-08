from collections import deque
import numpy as np


class StreamingClassificationMetrics:
    """Store classification results across chunks."""

    def __init__(self, positive_label=1, window_size=None):
        self.positive_label = positive_label
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.window = deque(maxlen=self.window_size)

    def update(self, y_true_chunk, y_pred_chunk):
        y_true = list(y_true_chunk)
        y_pred = list(y_pred_chunk)
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)
        if self.window_size is not None:
            for a, b in zip(y_true, y_pred):
                self.window.append((a, b))
        return self

    def _data(self):
        if self.window_size is None:
            return self.y_true, self.y_pred
        if not self.window:
            return [], []
        y_true, y_pred = zip(*self.window)
        return list(y_true), list(y_pred)

    def result(self):
        y_true, y_pred = self._data()
        return {
            "accuracy": accuracy(y_true, y_pred),
            "precision": precision(y_true, y_pred, self.positive_label),
            "recall": recall(y_true, y_pred, self.positive_label),
            "f1": f1_score(y_true, y_pred, self.positive_label),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }


def accuracy(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred, positive_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def recall(y_true, y_pred, positive_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def f1_score(y_true, y_pred, positive_label=1):
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def confusion_matrix(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_index = {label: i for i, label in enumerate(labels)}
    for a, b in zip(y_true, y_pred):
        matrix[label_index[a], label_index[b]] += 1
    return matrix


def auc_score(y_true, scores):
    # Simple AUC calculation using ranking
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pos = scores[y_true == 1]
    neg = scores[y_true != 1]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    total = 0
    for p in pos:
        total += np.sum(p > neg)
        total += 0.5 * np.sum(p == neg)
    return float(total / (len(pos) * len(neg)))

