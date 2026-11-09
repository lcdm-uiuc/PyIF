
import numpy as np
import pytest

from PyIF import CPU_TE
from PyIF.counting import counts_sorted
from PyIF.workspace import TEWorkspace
from PyIF import te_compute
from PyIF import helper


@pytest.mark.parametrize("embedding", [1, 2, 3])
@pytest.mark.parametrize("kind", ["normal", "ties", "constant"])
def test_sorted_counts_match_legacy(embedding, kind):
    rng = np.random.default_rng(1234)

    if kind == "normal":
        target = rng.normal(size=100)
    elif kind == "ties":
        target = rng.integers(0, 8, size=100).astype(float)
    else:
        target = np.ones(100)

    workspace = TEWorkspace({
        "target": target,
        "source": rng.normal(size=100),
    })

    xky, ky, xk, history, n_points = workspace.make_spaces(
        "source", "target", embedding
    )

    n_counted = len(xky) - embedding

    radii_xky = np.empty(len(xky))
    radii_xk = np.empty(len(xky))

    for i in range(len(xky)):
        point = xky[i, 0]
        neighbor = target[embedding + (i + 7) % n_points]

        distance = abs(point - neighbor)

        # Include exact and adjacent floating-point boundaries.
        radii_xky[i] = distance
        radii_xk[i] = np.nextafter(distance, np.inf)

    legacy_a = np.zeros(n_counted)
    legacy_b = np.zeros(n_counted)

    sorted_a = np.zeros(n_counted)
    sorted_b = np.zeros(n_counted)

    CPU_TE.countsInKD(
        legacy_a, legacy_b,
        radii_xky, radii_xk,
        xky, embedding, target
    )

    counts_sorted(
        sorted_a, sorted_b,
        radii_xky, radii_xk,
        xky,
        workspace.get_sorted_target("target", embedding)
    )

    np.testing.assert_array_equal(sorted_a, legacy_a)
    np.testing.assert_array_equal(sorted_b, legacy_b)


@pytest.mark.parametrize("embedding,k", [(1, 1), (2, 2), (3, 1)])
def test_sorted_workspace_te_matches_legacy(embedding, k):
    rng = np.random.default_rng(2026)

    target = rng.normal(size=200)
    source = rng.normal(size=200)

    workspace = TEWorkspace({
        "target": target,
        "source": source,
    })

    original = te_compute.te_compute(
        X=target, Y=source,
        k=k, embedding=embedding,
        GPU=False, safetyCheck=False
    )

    optimized = workspace.estimate_ksg(
        "source", "target",
        k=k, embedding=embedding
    )

    np.testing.assert_array_equal(optimized, original)


def test_sorted_target_is_reused():
    rng = np.random.default_rng(456)

    workspace = TEWorkspace({
        "target": rng.normal(size=100),
        "source": rng.normal(size=100),
    })

    first = workspace.get_sorted_target("target", embedding=2)
    second = workspace.get_sorted_target("target", embedding=2)

    assert first is second
    assert not first.flags.writeable


@pytest.mark.parametrize(
    "kind",
    ["normal", "ties", "constant", "adjacent", "large", "small"]
)
@pytest.mark.parametrize("embedding", [1, 2, 4, 8])
def test_sorted_counts_randomized_stress(kind, embedding):
    """Compare optimized counts with legacy counts on difficult inputs."""

    for seed in range(20):
        rng = np.random.default_rng(seed)
        n = 80 + seed * 3

        if kind == "normal":
            target = rng.normal(size=n)

        elif kind == "ties":
            target = rng.integers(-3, 4, size=n).astype(float)

        elif kind == "constant":
            target = np.ones(n)

        elif kind == "adjacent":
            target = rng.choice([
                np.nextafter(1.0, -np.inf),
                1.0,
                np.nextafter(1.0, np.inf),
            ], size=n)

        elif kind == "large":
            target = rng.normal(size=n) * 1e150

        else:
            target = rng.normal(size=n) * 1e-150

        source = rng.normal(size=n)

        xky, ky, xk, history, n_points = (
            helper.make_spaces(
                target, source, embedding=embedding
            )
        )

        n_counted = len(xky) - embedding

        radius_a = np.zeros(n_points)
        radius_b = np.zeros(n_points)

        for i in range(n_counted):
            point = xky[i, 0]
            neighbor = target[rng.integers(embedding, n)]

            distance = abs(point - neighbor)

            options = [
                distance,
                np.nextafter(distance, np.inf),
                np.nextafter(distance, 0.0),
                0.0,
            ]

            radius_a[i] = options[i % 4]
            radius_b[i] = options[(i + seed + 1) % 4]

        legacy_a = np.zeros(n_counted)
        legacy_b = np.zeros(n_counted)

        optimized_a = np.zeros(n_counted)
        optimized_b = np.zeros(n_counted)

        CPU_TE.countsInKD(
            legacy_a, legacy_b,
            radius_a, radius_b,
            xky, embedding, target
        )

        counts_sorted(
            optimized_a, optimized_b,
            radius_a, radius_b,
            xky,
            np.sort(target[embedding:])
        )

        np.testing.assert_array_equal(
            optimized_a, legacy_a
        )

        np.testing.assert_array_equal(
            optimized_b, legacy_b
        )
