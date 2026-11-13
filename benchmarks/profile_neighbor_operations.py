
import statistics
import time

import numpy as np
from sklearn.neighbors import KDTree

from PyIF import CPU_TE, helper


N = 5000
N_SOURCES = 10
EMBEDDING = 2
K = 2
REPEATS = 7

STAGES = (
    "joint_xky",
    "joint_xk",
    "range_ky",
    "range_k",
    "manual_x",
)


class TimedTree:
    """Measure KD-tree operations without changing their results."""

    def __init__(self, tree, name, timings):
        self.tree = tree
        self.name = name
        self.timings = timings

    def query(self, *args, **kwargs):
        start = time.perf_counter()
        result = self.tree.query(*args, **kwargs)
        self.timings[self.name] += time.perf_counter() - start
        return result

    def query_radius(self, *args, **kwargs):
        start = time.perf_counter()
        result = self.tree.query_radius(*args, **kwargs)
        self.timings[self.name] += time.perf_counter() - start
        return result


rng = np.random.default_rng(2026)

target = rng.normal(size=N)
sources = [
    rng.normal(size=N)
    for _ in range(N_SOURCES)
]

# Construct histories and trees outside the timed estimator stage.
prepared_pairs = []

for source in sources:
    spaces = helper.make_spaces(
        target, source, embedding=EMBEDDING
    )

    trees = tuple(
        KDTree(space, metric="chebyshev")
        for space in spaces[:4]
    )

    prepared_pairs.append((trees, spaces))


# Warm up Numba before recording measurements.
first_trees, first_spaces = prepared_pairs[0]

CPU_TE.compute(
    *first_trees,
    *first_spaces,
    target,
    embedding=EMBEDDING,
    k=K,
)

# Record uninstrumented reference values.
reference_results = [
    CPU_TE.compute(
        *trees,
        *spaces,
        target,
        embedding=EMBEDDING,
        k=K,
    )
    for trees, spaces in prepared_pairs
]


original_counts = CPU_TE.countsInKD
runs = []

for repeat in range(REPEATS):
    timings = {stage: 0.0 for stage in STAGES}
    timings["total"] = 0.0

    def timed_counts(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_counts(*args, **kwargs)
        finally:
            timings["manual_x"] += time.perf_counter() - start

    # Temporarily instrument the existing Numba counting call.
    CPU_TE.countsInKD = timed_counts

    try:
        for i, (trees, spaces) in enumerate(prepared_pairs):
            traced_trees = [
                TimedTree(trees[0], "joint_xky", timings),
                TimedTree(trees[1], "range_ky", timings),
                TimedTree(trees[2], "joint_xk", timings),
                TimedTree(trees[3], "range_k", timings),
            ]

            start = time.perf_counter()

            result = CPU_TE.compute(
                *traced_trees,
                *spaces,
                target,
                embedding=EMBEDDING,
                k=K,
            )

            timings["total"] += time.perf_counter() - start

            np.testing.assert_allclose(
                result,
                reference_results[i],
                rtol=0,
                atol=0,
            )

    finally:
        # Always restore the original function.
        CPU_TE.countsInKD = original_counts

    timings["other"] = (
        timings["total"]
        - sum(timings[stage] for stage in STAGES)
    )

    runs.append(timings)


def median_of(stage):
    return statistics.median(run[stage] for run in runs)


total = median_of("total")

groups = {
    "Joint neighbor searches":
        median_of("joint_xky") + median_of("joint_xk"),
    "Marginal KD-tree counts":
        median_of("range_ky") + median_of("range_k"),
    "Numba target counts":
        median_of("manual_x"),
    "Other operations":
        median_of("other"),
}

print("\nNumerical comparison: PASSED")
print(f"N={N}, pairs={N_SOURCES}, k={K}, embedding={EMBEDDING}")
print("\nMedian estimator-stage timings:")

for name, seconds in groups.items():
    print(
        f"{name:25s}: {seconds:.4f}s "
        f"({100 * seconds / total:.1f}%)"
    )

print(f"\nTotal estimator time: {total:.4f}s")

print("\nDetailed KD-tree operations:")
for name in STAGES:
    print(f"{name:15s}: {median_of(name):.4f}s")
