
import time
import statistics

import numpy as np
from sklearn.neighbors import KDTree

from PyIF import helper, CPU_TE, te_compute


N = 2000
N_SOURCES = 10
EMBEDDING = 2
K = 2
REPEATS = 5

rng = np.random.default_rng(2026)

target = rng.normal(size=N)
sources = [rng.normal(size=N) for _ in range(N_SOURCES)]

# Compile the existing Numba functions before timing.
te_compute.te_compute(
    target, sources[0],
    k=K, embedding=EMBEDDING,
    GPU=False, safetyCheck=False
)

results = {
    "histories": [],
    "trees": [],
    "estimator": [],
}

for _ in range(REPEATS):
    totals = {name: 0.0 for name in results}

    for source in sources:
        start = time.perf_counter()

        xky, ky, xk, history, n_points = helper.make_spaces(
            target, source, embedding=EMBEDDING
        )

        t1 = time.perf_counter()

        trees = [
            KDTree(space, metric="chebyshev")
            for space in (xky, ky, xk, history)
        ]

        t2 = time.perf_counter()

        CPU_TE.compute(
            *trees,
            xky, ky, xk, history,
            n_points, target,
            embedding=EMBEDDING,
            k=K
        )

        t3 = time.perf_counter()

        totals["histories"] += t1 - start
        totals["trees"] += t2 - t1
        totals["estimator"] += t3 - t2

    for name in results:
        results[name].append(totals[name])

medians = {
    name: statistics.median(values)
    for name, values in results.items()
}

total = sum(medians.values())

print(f"\nN={N}, pairs={N_SOURCES}, k={K}, embedding={EMBEDDING}")
print("Median times across repeated batches:\n")

for name, seconds in medians.items():
    percentage = 100 * seconds / total
    print(f"{name:12s}: {seconds:.4f}s ({percentage:.1f}%)")

print(f"\nTotal: {total:.4f}s")
