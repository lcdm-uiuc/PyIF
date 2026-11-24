"""Public interface tests for Mesner-Shalizi estimation."""

import numpy as np
import pytest

from PyIF import estimate_te
from PyIF.config import KSGConfig, MSConfig
from PyIF.ms import conditional_mi


@pytest.mark.parametrize("value", [1, 16, np.int32(4), np.int64(8)])
def test_ms_accepts_integer_types(value):
    config = MSConfig(k_neighbors=value)

    assert config.k_neighbors == int(value)
    assert type(config.k_neighbors) is int


@pytest.mark.parametrize(
    "value", [True, False, 0, -1, 1.5, "2", None]
)
def test_ms_rejects_invalid_k(value):
    with pytest.raises(ValueError):
        MSConfig(k_neighbors=value)


@pytest.mark.parametrize("value", [1, "True", None])
def test_ms_rejects_invalid_standardize(value):
    with pytest.raises(TypeError):
        MSConfig(standardize=value)


def test_ms_default_settings():
    rng = np.random.default_rng(42)

    source = rng.normal(size=100)
    target = rng.normal(size=100)

    result = estimate_te(source, target, estimator="ms")

    assert result.units == "nats"
    assert result.n_input == 100
    assert result.n_history_rows == 99
    assert result.resolved_settings["estimator"] == "ms"

    assert result.resolved_settings["config"] == {
        "k_neighbors": 16,
        "standardize": True,
    }

    assert result.inference.status == "not_computed"
    assert len(result.data_hashes["source"]) == 64
    assert len(result.data_hashes["target"]) == 64


@pytest.mark.parametrize("standardize", [True, False])
def test_ms_multiple_histories_and_delay(standardize):
    rng = np.random.default_rng(42)

    source = rng.normal(size=40)
    target = rng.normal(size=40)

    # Construct the aligned observations independently.
    data = np.column_stack([
        target[4:],
        source[1:-3],
        source[:-4],
        target[3:-1],
        target[2:-2],
    ])

    if standardize:
        data = (
            (data - data.mean(axis=0))
            / data.std(axis=0)
        )

    expected = conditional_mi(
        data[:, 0],
        data[:, 1:3],
        data[:, 3:],
        k=4,
    )

    result = estimate_te(
        source,
        target,
        estimator="ms",
        target_history=2,
        source_history=2,
        delay=3,
        config=MSConfig(
            k_neighbors=4,
            standardize=standardize,
        ),
    )

    assert result.raw_te == pytest.approx(
        expected, abs=1e-12, rel=0
    )

    assert result.n_history_rows == 36
    assert result.resolved_settings["target_history"] == 2
    assert result.resolved_settings["source_history"] == 2
    assert result.resolved_settings["delay"] == 3


def test_ms_rejects_wrong_config():
    source = np.arange(100, dtype=float)

    with pytest.raises(TypeError, match="MSConfig"):
        estimate_te(
            source,
            source,
            estimator="ms",
            config=KSGConfig(),
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_ms_rejects_nonfinite_inputs(invalid):
    source = np.arange(100, dtype=float)
    target = source.copy()
    target[10] = invalid

    with pytest.raises(ValueError, match="finite"):
        estimate_te(source, target, estimator="ms")


def test_ms_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="length"):
        estimate_te(
            np.arange(100),
            np.arange(99),
            estimator="ms",
        )


def test_ms_rejects_complex_inputs():
    source = np.ones(100, dtype=complex)
    target = np.ones(100)

    with pytest.raises(TypeError, match="real numeric"):
        estimate_te(source, target, estimator="ms")


def test_ms_checks_k_against_history_rows():
    source = np.arange(10, dtype=float)

    with pytest.raises(ValueError, match="history rows"):
        estimate_te(
            source,
            source,
            estimator="ms",
            target_history=2,
            config=MSConfig(k_neighbors=8),
        )


def test_ms_does_not_modify_inputs():
    rng = np.random.default_rng(42)

    source = rng.normal(size=100)
    target = rng.normal(size=100)

    original_source = source.copy()
    original_target = target.copy()

    estimate_te(source, target, estimator="ms")

    np.testing.assert_array_equal(source, original_source)
    np.testing.assert_array_equal(target, original_target)