import numpy as np
import pytest

from PyIF.ms import conditional_mi


def brute_force_cmi(x, y, z, k):
    """Small-sample reference for MS equations 18-19."""

    x, y, z = [
        np.asarray(a, dtype=float).reshape(len(a), -1)
        for a in (x, y, z)
    ]

    spaces = [
        np.column_stack([x, y, z]),
        np.column_stack([x, z]),
        np.column_stack([y, z]),
        z,
    ]

    distances = [
        np.max(
            np.abs(a[:, None, :] - a[None, :, :]),
            axis=2,
        )
        for a in spaces
    ]

    # Exclude each observation from its own neighbors.
    for d in distances:
        np.fill_diagonal(d, np.inf)

    radius = np.sort(distances[0], axis=1)[:, k - 1]

    # Include all observations on the boundary.
    counts = np.stack(
        [(d <= radius[:, None]).sum(axis=1)
         for d in distances],
        axis=1,
    )

    local = []

    for m, a, b, c in counts:
        if m > k:
            f = np.log
        else:
            # Digamma at integer n equals H_(n-1) - gamma.
            # Euler's constant cancels in the CMI expression.
            def f(n):
                return sum(1 / j for j in range(1, n))

        local.append(f(m) - f(a) - f(b) + f(c))

    return radius, counts, np.array(local)


def sample(case):
    rng = np.random.default_rng(42)

    if case == "continuous":
        return (
            rng.normal(size=(20, 2)),
            rng.normal(size=20),
            rng.normal(size=(20, 2)),
            3,
        )

    if case == "duplicates":
        return (
            [0, 0, 0, 1, 2],
            [0, 0, 0, 1, 2],
            [0] * 5,
            1,
        )

    if case == "positive_radius_ties":
        return (
            [0, 1, -1, 3],
            [0, 0, 0, 2],
            [0] * 4,
            1,
        )

    if case == "exactly_k_duplicates":
        return (
            [0, 0, 1, 3],
            [0, 0, 2, 5],
            [0] * 4,
            1,
        )

    return [1] * 10, [1] * 10, [1] * 10, 3


@pytest.mark.parametrize(
    "case",
    [
        "continuous",
        "duplicates",
        "positive_radius_ties",
        "exactly_k_duplicates",
        "constant",
    ],
)
def test_ms_against_brute_force(case):
    x, y, z, k = sample(case)

    radius, counts, local = brute_force_cmi(x, y, z, k)

    actual = conditional_mi(
        x, y, z, k=k, return_details=True
    )

    np.testing.assert_array_equal(
        actual["radius"], radius
    )

    for j, key in enumerate(
        ["joint_count", "nxz", "nyz", "nz"]
    ):
        np.testing.assert_array_equal(
            actual[key], counts[:, j]
        )

    np.testing.assert_array_equal(
        actual["log_branch"], counts[:, 0] > k
    )

    np.testing.assert_allclose(
        actual["local"], local, rtol=0, atol=1e-12
    )

    assert actual["value"] == pytest.approx(
        local.mean(), abs=1e-12, rel=0
    )

    if case == "positive_radius_ties":
        assert actual["radius"][0] == 1
        assert actual["joint_count"][0] == 2
        assert actual["log_branch"][0]

    if case == "constant":
        assert actual["value"] == 0