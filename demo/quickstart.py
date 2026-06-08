import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.stream import StreamTrainer


def make_data(n_samples=120):
    np.random.seed(1)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # add a few missing values
    X[5, 0] = np.nan
    X[30, 1] = np.nan
    return X, y


def make_chunks(X, y, chunk_size=20):
    for start in range(0, len(X), chunk_size):
        end = start + chunk_size
        yield X[start:end], y[start:end]


def main():
    X, y = make_data()

    pipe = Pipeline([
        ("imputer", Imputer()),
        ("scale", StandardScaler()),
        ("model", EnsembleClassifier(n_estimators=3, max_depth=4)),
    ])

    trainer = StreamTrainer(pipe)

    for X_chunk, y_chunk in make_chunks(X, y):
        trainer.fit_chunk(X_chunk, y_chunk)
        row = trainer.score_chunk(X_chunk, y_chunk)
        print(
            "chunk",
            row["chunk"],
            "accuracy:",
            round(row["accuracy"], 3),
            "precision:",
            round(row["precision"], 3),
            "recall:",
            round(row["recall"], 3),
        )


if __name__ == "__main__":
    main()
