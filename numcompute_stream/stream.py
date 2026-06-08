import os
import sys

from .metrics import StreamingClassificationMetrics


class StreamTrainer:
    def __init__(self, model, metric=None):
        self.model = model
        self.metric = metric or StreamingClassificationMetrics()
        self.history = []

    def fit_chunk(self, X, y):
        self.model.partial_fit(X, y)
        return self

    def score_chunk(self, X, y):
        pred = self.model.predict(X)
        self.metric.update(y, pred)
        result = self.metric.result()
        log_row = {
            "chunk": len(self.history) + 1,
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "memory_mb": self._memory_mb(),
        }
        self.history.append(log_row)
        return log_row

    def fit_stream(self, chunks):
        for X, y in chunks:
            self.fit_chunk(X, y)
            self.score_chunk(X, y)
        return self.history

    def _memory_mb(self):
        # rough memory size for logging
        return sys.getsizeof(self.model) / (1024 * 1024)

    def save_log(self, path):
        lines = ["chunk,accuracy,precision,recall,memory_mb"]
        for row in self.history:
            lines.append(
                f"{row['chunk']},{row['accuracy']},{row['precision']},"
                f"{row['recall']},{row['memory_mb']}"
            )
        with open(path, "w", encoding="utf-8") as f:
            f.write(os.linesep.join(lines))

