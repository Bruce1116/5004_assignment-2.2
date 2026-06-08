import os
import tempfile
import unittest

import numpy as np

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


class StreamTests(unittest.TestCase):
    def make_trainer(self):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=2)),
        ])
        return StreamTrainer(pipe)

    def test_stream_trainer_fit_chunk(self):
        trainer = self.make_trainer()
        returned = trainer.fit_chunk([[0], [1], [2]], [0, 0, 1])
        self.assertIs(returned, trainer)

    def test_stream_trainer_score_chunk_accuracy(self):
        trainer = self.make_trainer()
        X = np.array([[0], [1], [2]])
        y = np.array([0, 0, 1])
        trainer.fit_chunk(X, y)
        row = trainer.score_chunk(X, y)
        self.assertIn("accuracy", row)

    def test_stream_trainer_history_updates(self):
        trainer = self.make_trainer()
        X = np.array([[0], [1], [2]])
        y = np.array([0, 0, 1])
        trainer.fit_chunk(X, y)
        trainer.score_chunk(X, y)
        self.assertEqual(len(trainer.history), 1)

    def test_stream_trainer_fit_stream(self):
        trainer = self.make_trainer()
        chunks = [
            (np.array([[0], [1]]), np.array([0, 0])),
            (np.array([[2], [3]]), np.array([1, 1])),
        ]
        history = trainer.fit_stream(chunks)
        self.assertEqual(len(history), 2)

    def test_stream_trainer_save_log(self):
        trainer = self.make_trainer()
        X = np.array([[0], [1], [2]])
        y = np.array([0, 0, 1])
        trainer.fit_chunk(X, y)
        trainer.score_chunk(X, y)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "log.csv")
            trainer.save_log(path)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
