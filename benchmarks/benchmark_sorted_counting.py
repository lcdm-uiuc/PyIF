
import statistics
import time

import numpy as np
from sklearn.neighbors import KDTree

from PyIF import CPU_TE, helper
from PyIF.counting import counts_sorted


SIZES = [2000, 5000, 10000]
EMBEDDING = 2
K = 2
REPEATS = 7


def median_runtime(function):
    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)

    return statistics.median(times)


def benchmark(n):
    rng = np.random.default_rng(2026)

    target = rng.normal(size=n)
    source = rng.normal(size=n)

    xky, ky, xk, history, n_points = helper.make_spaces(
        target, source, embedding=EMBEDDING
    )

    trees = tuple(
        KDTree(space, metric="chebyshev")
        for space in (xky, ky, xk, history)
    )

    # Calculate the same radii used by the existing estimator.
    idx_xky = trees[0].query(xky, k=K + 1)[1][:, K]
    idx_xk = trees[2].query(xk, k=K + 1)[1][:, K]

    radii_xky = np.abs(xky[:, 0] - xky[idx_xky, 0])
    radii_xk = np.abs(xk[:, 0] - xk[idx_xk, 0])

    sorted_target = np.sort(target[EMBEDDING:])

    n_counted = len(xky) - EMBEDDING

    legacy_a = np.zeros(n_counted)
    legacy_b = np.zeros(n_counted)
    sorted_a = np.zeros(n_counted)
    sorted_b = np.zeros(n_counted)

    def legacy_counts():
        CPU_TE.countsInKD(
            legacy_a, legacy_b,
            radii_xky, radii_xk,
            xky, EMBEDDING, target
        )

    def optimized_counts():
        counts_sorted(
            sorted_a, sorted_b,
            radii_xky, radii_xk,
            xky, sorted_target
        )

    def legacy_estimator():
        return CPU_TE.compute(
            *trees, xky, ky, xk, history,
            n_points, target,
            embedding=EMBEDDING, k=K
        )

    def optimized_estimator():
        return CPU_TE.compute(
            *trees, xky, ky, xk, history,
            n_points, target,
            embedding=EMBEDDING, k=K,
            sorted_target=sorted_target
        )

    # Warm up both implementations, including Numba.
    legacy_counts()
    optimized_counts()

    np.testing.assert_array_equal(legacy_a, sorted_a)
    np.testing.assert_array_equal(legacy_b, sorted_b)

    legacy_te = legacy_estimator()
    optimized_te = optimized_estimator()

    assert (
        np.float64(legacy_te).tobytes()
        == np.float64(optimized_te).tobytes()
    ), "TE estimates differ at the bit level"

    count_old = median_runtime(legacy_counts)
    count_new = median_runtime(optimized_counts)

    te_old = median_runtime(legacy_estimator)
    te_new = median_runtime(optimized_estimator)

    print(f"\nN = {n:,}")
    print("Numerical comparison: EXACT MATCH")
    print(
        f"Target counting: {count_old:.5f}s -> "
        f"{count_new:.5f}s "
        f"({count_old / count_new:.2f}x)"
    )
    print(
        f"Full estimator:  {te_old:.5f}s -> "
        f"{te_new:.5f}s "
        f"({te_old / te_new:.2f}x)"
    )


if __name__ == "__main__":
    for n in SIZES:
        benchmark(n)
