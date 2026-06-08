# Assignment 2.2 - Streaming Machine Learning

This project is a small streaming machine learning package. It trains and
evaluates models on chunks of data instead of fitting everything only once.
The code is kept simple so the streaming idea is easy to follow.

## Project structure

```text
numcompute_stream/
  __init__.py
  stream.py
  tree.py
  ensemble.py
  preprocessing.py
  stats.py
  metrics.py
  pipeline.py
  visualise.py
tests/
  test_metrics.py
  test_stats.py
  test_preprocessing.py
  test_pipeline.py
  test_tree.py
  test_ensemble.py
  test_stream.py
demo/
  quickstart.py
  stream_demo.ipynb
benchmark/
  benchmark_loop_vs_vectorised.py
README.md
Assignment2.2-specs.md
```

## How to run the demo

Run the Python demo:

```bash
python demo/quickstart.py
```

Open the notebook demo:

```bash
jupyter notebook demo/stream_demo.ipynb
```

## How to run the tests

```bash
python -m unittest discover -s tests
```

## How to run the benchmark

```bash
python benchmark/benchmark_loop_vs_vectorised.py
```

The benchmark compares a normal Python loop with a NumPy vectorised version
for calculating a mean over data chunks.

## Streaming design

Most classes use `partial_fit` so they can update with each new chunk. The
`Pipeline` first updates and applies preprocessing steps such as the
`Imputer` and `StandardScaler`, then it trains the model.

`StreamTrainer` controls the chunk loop. It calls `fit_chunk`, predicts on the
chunk, updates `StreamingClassificationMetrics`, and stores a simple history
with accuracy, precision, recall, and memory use.

The decision tree stores the data seen so far and rebuilds a small tree when a
new chunk arrives. This is not the fastest online tree method, but it is clear
and suitable for this assignment. The ensemble trains several small trees and
uses majority voting for predictions.
