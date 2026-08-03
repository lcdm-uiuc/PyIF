import numpy as np
import pytest

from PyIF import te_compute as te


def valid_inputs():
    """Return two valid arrays that can be modified by individual tests."""
    x = np.arange(10, dtype=float)
    y = np.arange(10, dtype=float)
    return x, y


def test_rejects_k_below_one():
    x, y = valid_inputs()

    with pytest.raises(
        AssertionError,
        match="K should be greater than or equal to 1",
    ):
        te.te_compute(x, y, k=0, GPU=False)


def test_rejects_embedding_below_one():
    x, y = valid_inputs()

    with pytest.raises(
        AssertionError,
        match="The embedding must be greater than or equal to 1",
    ):
        te.te_compute(x, y, embedding=0, GPU=False)


def test_rejects_non_numpy_x():
    _, y = valid_inputs()
    x = list(range(10))

    with pytest.raises(
        AssertionError,
        match="X should be a numpy array",
    ):
        te.te_compute(x, y, GPU=False)


def test_rejects_non_numpy_y():
    x, _ = valid_inputs()
    y = list(range(10))

    with pytest.raises(
        AssertionError,
        match="Y should be a numpy array",
    ):
        te.te_compute(x, y, GPU=False)


def test_rejects_unequal_lengths():
    x = np.arange(10, dtype=float)
    y = np.arange(8, dtype=float)

    with pytest.raises(
        AssertionError,
        match="The length of X & Y are not equal",
    ):
        te.te_compute(x, y, GPU=False)