import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))


def loop_chunk_mean(chunks):
    total = 0.0
    count = 0
    for chunk in chunks:
        for value in chunk:
            total += value
            count += 1
    return total / count


def vectorised_chunk_mean(chunks):
    total = 0.0
    count = 0
    for chunk in chunks:
        total += np.sum(chunk)
        count += len(chunk)
    return total / count


def main():
    np.random.seed(2)
    data = np.random.randn(200000)
    chunks = np.array_split(data, 20)

    start = time.perf_counter()
    loop_result = loop_chunk_mean(chunks)
    loop_time = time.perf_counter() - start

    start = time.perf_counter()
    vector_result = vectorised_chunk_mean(chunks)
    vector_time = time.perf_counter() - start

    print("Loop mean:", round(loop_result, 6))
    print("Vectorised mean:", round(vector_result, 6))
    print("Loop time:", round(loop_time, 6), "seconds")
    print("Vectorised time:", round(vector_time, 6), "seconds")
    print("Difference:", abs(loop_result - vector_result))


if __name__ == "__main__":
    main()
