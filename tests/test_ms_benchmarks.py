"""Known-answer TE benchmarks adapted from Jingyuan Chen's Week 6 experiments."""

import numpy as np
import pytest
from scipy.signal import lfilter

from PyIF import estimate_te
from PyIF.config import MSConfig


ACTIVE_PROBABILITY = 0.2
CHANNEL_STRENGTH = 0.8
ECHO_COEFFICIENT = 0.35
BURN_IN = 500

GAUSSIAN_TE = 0.5 * np.log1p(CHANNEL_STRENGTH**2)


def benchmark_sample(case, seed, n=5000):
    """Generate source and target with known population TE."""

    rng = np.random.default_rng(seed)

    # Preserve the original Week 6 random-number generation order.
    u, noise = rng.normal(size=(2, 80500))
    active = rng.random(len(u)) < ACTIVE_PROBABILITY

    if case in ("zeros_null", "echo_null"):
        source = u * (
            rng.random(len(u)) < ACTIVE_PROBABILITY
        )
        target = noise * active

    else:
        source = u
        target = (
            noise
            + CHANNEL_STRENGTH * np.r_[0.0, u[:-1]]
        )

        if case == "gated_link":
            target *= active

    if case in ("echo_null", "gated_link"):
        target = lfilter(
            [1], [1, -ECHO_COEFFICIENT], target
        )

    if case == "echo_null":
        source = lfilter(
            [1], [1, -ECHO_COEFFICIENT], source
        )

    return (
        source[BURN_IN:BURN_IN + n],
        target[BURN_IN:BURN_IN + n],
    )


@pytest.mark.parametrize(
    "case, expected, tolerance",
    [
        ("gaussian", GAUSSIAN_TE, 0.03),
        ("zeros_null", 0.0, 0.01),
        ("echo_null", 0.0, 0.01),
        (
            "gated_link",
            ACTIVE_PROBABILITY * GAUSSIAN_TE,
            0.015,
        ),
    ],
)
def test_ms_known_answer(case, expected, tolerance):
    """MS should recover known TE on continuous and mixed data."""

    estimates = []

    for seed in (1, 2, 3):
        source, target = benchmark_sample(case, seed)

        result = estimate_te(
            source=source,
            target=target,
            estimator="ms",
            config=MSConfig(
                k_neighbors=16,
                standardize=True,
            ),
        )

        assert np.isfinite(result.raw_te)
        assert result.units == "nats"
        assert result.n_history_rows == len(target) - 1
        assert result.resolved_settings["estimator"] == "ms"

        estimates.append(result.raw_te)

    median = np.median(estimates)

    assert median == pytest.approx(
        expected,
        abs=tolerance,
        rel=0,
    )