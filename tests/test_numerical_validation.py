import numpy as np
import pytest

from PyIF import te_compute as te


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_rejects_nonfinite_x(bad_value):
    x = np.arange(20, dtype=float)
    y = np.arange(20, dtype=float)
    x[5] = bad_value

    with pytest.raises(
        ValueError,
        match="X must contain only finite values",
    ):
        te.te_compute(
            x,
            y,
            safetyCheck=False,
            GPU=False,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_rejects_nonfinite_y(bad_value):
    x = np.arange(20, dtype=float)
    y = np.arange(20, dtype=float)
    y[5] = bad_value

    with pytest.raises(
        ValueError,
        match="Y must contain only finite values",
    ):
        te.te_compute(
            x,
            y,
            safetyCheck=False,
            GPU=False,
        )


def test_rejects_embedding_equal_to_series_length():
    """Embedding must leave at least one usable sample."""
    x = np.arange(10, dtype=float)
    y = np.arange(10, dtype=float)

    with pytest.raises(
        ValueError,
        match="embedding must be less than the length",
    ):
        te.te_compute(
            x,
            y,
            embedding=10,
            safetyCheck=False,
            GPU=False,
        )


def test_rejects_k_larger_than_available_neighbors():
    """k must leave enough usable neighbors after embedding."""
    x = np.arange(10, dtype=float)
    y = np.arange(10, dtype=float) * 2

    with pytest.raises(
        ValueError,
        match="k must be less than the number of usable samples",
    ):
        te.te_compute(
            x,
            y,
            k=20,
            embedding=1,
            safetyCheck=False,
            GPU=False,
        )


def test_largest_valid_k_executes():
    """k = usable_samples - 1 should be allowed."""
    rng = np.random.RandomState(seed=50)
    x = rng.randn(10)
    y = rng.randn(10)

    # len=10, embedding=1 -> 9 usable points -> largest valid k is 8
    result = te.te_compute(
        x,
        y,
        k=8,
        embedding=1,
        safetyCheck=False,
        GPU=False,
    )

    assert np.isfinite(result)


def test_k_equal_to_usable_samples_is_rejected():
    """k cannot equal the number of usable samples."""
    x = np.arange(10, dtype=float)
    y = np.arange(10, dtype=float) * 2

    # len=10, embedding=1 -> 9 usable points
    with pytest.raises(
        ValueError,
        match="k must be less than the number of usable samples",
    ):
        te.te_compute(
            x,
            y,
            k=9,
            embedding=1,
            safetyCheck=False,
            GPU=False,
        )