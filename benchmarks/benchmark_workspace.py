
import time
import statistics

import numpy as np

from PyIF import te_compute
from PyIF.workspace import TEWorkspace


# Same data and parameters for both implementations
N = 2000
N_SOURCES = 10
EMBEDDING = 2
K = 2
REPEATS = 1

rng = np.random.default_rng(2026)

target = rng.normal(size=N)
sources = {
    f"source_{i}": rng.normal(size=N)
    for i in range(N_SOURCES)
}


def run_legacy():
    results = {}

    for name, source in sources.items():
        results[name] = te_compute.te_compute(
            X=target,
            Y=source,
            k=K,
            embedding=EMBEDDING,
            GPU=False,
            safetyCheck=False
        )

    return results


def run_cached():
    workspace = TEWorkspace({
        "target": target,
        **sources
    })

    results = {}

    for name in sources:
        results[name] = workspace.estimate_ksg(
            source_name=name,
            target_name="target",
            k=K,
            embedding=EMBEDDING
        )

    return results


# Warm up Numba before measuring runtime
run_legacy()
run_cached()

# Verify numerical equivalence
original = run_legacy()
cached = run_cached()

for name in sources:
    np.testing.assert_allclose(
        original[name],
        cached[name],
        rtol=0,
        atol=1e-12
    )

print("Numerical comparison: PASSED")

legacy_times = []
cached_times = []

for _ in range(REPEATS):
    start = time.perf_counter()
    run_legacy()
    legacy_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    run_cached()
    cached_times.append(time.perf_counter() - start)

legacy_median = statistics.median(legacy_times)
cached_median = statistics.median(cached_times)

print(f"Sample size: {N}")
print(f"Number of pairs: {N_SOURCES}")
print(f"Embedding: {EMBEDDING}, k: {K}")
print(f"Legacy median: {legacy_median:.4f} seconds")
print(f"Cached median: {cached_median:.4f} seconds")
print(f"Speedup: {legacy_median / cached_median:.2f}x")
