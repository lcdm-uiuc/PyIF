"""Numerical robustness tests for the MS estimator."""

import numpy as np
import pytest

from PyIF import estimate_te
from PyIF.config import MSConfig
from PyIF.ms.numerics import prepare_collection


def ms_te(source, target, k=16, standardize=True):
    return estimate_te(
        source=source,
        target=target,
        estimator="ms",
        config=MSConfig(
            k_neighbors=k,
            standardize=standardize,
        ),
    ).raw_te


def test_constant_target():
    """A constant target contains no transferable information."""

    rng = np.random.default_rng(23)
    source = rng.normal(size=256)
    target = np.zeros(256)

    assert ms_te(source, target) == pytest.approx(
        0.0, abs=1e-12
    )


def test_constant_source():
    """A constant source supplies no predictive information."""

    rng = np.random.default_rng(23)
    source = np.zeros(256)
    target = rng.normal(size=256)

    assert ms_te(source, target) == pytest.approx(
        0.0, abs=1e-12
    )


@pytest.mark.parametrize("power", [-500, 500, 1024])
@pytest.mark.parametrize("standardize", [True, False])
def test_common_binary_scaling(power, standardize):
    """Lossless common scaling must preserve neighbor geometry."""

    target, source = np.random.default_rng(0).uniform(
        -1, 1, size=(2, 64)
    )

    scaled_source = np.ldexp(source, power)
    scaled_target = np.ldexp(target, power)

    np.testing.assert_array_equal(
        np.ldexp(scaled_source, -power), source
    )

    np.testing.assert_array_equal(
        np.ldexp(scaled_target, -power), target
    )

    expected = ms_te(
        source, target, standardize=standardize
    )

    actual = ms_te(
        scaled_source,
        scaled_target,
        standardize=standardize,
    )

    assert actual == pytest.approx(
        expected, abs=1e-12, rel=0
    )


@pytest.mark.parametrize("power", [-600, 600])
def test_extreme_relative_scaling(power):
    """Standardization must remain stable across coordinates."""

    rng = np.random.default_rng(42)

    source = rng.normal(size=2000)
    target = rng.normal(size=2000)
    target[1:] += 0.8 * source[:-1]

    expected = ms_te(source, target)
    actual = ms_te(source, np.ldexp(target, power))

    assert actual == pytest.approx(
        expected, abs=1e-12, rel=0
    )


def test_large_integer_precision():
    """Large integer offsets must not collapse distinct values."""

    rng = np.random.default_rng(23)
    source = rng.permutation(256).astype(np.int64)
    target = np.roll(source, 1)

    offset = np.int64(2**62)

    large_source = source + offset
    large_target = target + offset

    prepared, _ = prepare_collection({
        "source": large_source,
        "target": large_target,
    })

    np.testing.assert_array_equal(
        prepared["source"], source
    )

    np.testing.assert_array_equal(
        prepared["target"], target
    )

    expected = ms_te(source, target)
    actual = ms_te(large_source, large_target)

    assert actual == pytest.approx(
        expected, abs=1e-12, rel=0
    )


def test_near_perfect_prediction():
    """Strong directional dependence should remain finite."""

    rng = np.random.default_rng(42)

    source = rng.normal(size=1000)

    target = (
        np.r_[0.0, source[:-1]]
        + 1e-6 * rng.normal(size=1000)
    )

    forward = ms_te(source, target)
    reverse = ms_te(target, source)

    assert np.isfinite(forward)
    assert np.isfinite(reverse)
    assert forward > 1.0
    assert forward > reverse


def test_unrepresentable_integer_range():
    """Reject integer inputs that cannot be preserved exactly."""

    target = np.array(
        [0, 2**62, 2**62 + 1, 3],
        dtype=np.int64,
    )

    source = np.arange(4, dtype=float)

    with pytest.raises(
        ValueError, match="integer range"
    ):
        ms_te(source, target, k=1)