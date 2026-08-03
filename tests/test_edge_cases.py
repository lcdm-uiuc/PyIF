import numpy as np
import pytest

from PyIF import helper
from PyIF import te_compute as te


def test_make_spaces_embedding_one():
    """Document the exact spaces currently built for embedding=1."""
    x = np.array([10.0, 20.0, 30.0, 40.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    xky, ky, xk, k_space, n_points = helper.make_spaces(
        x,
        y,
        embedding=1,
    )

    np.testing.assert_array_equal(
        xky,
        np.array(
            [
                [20.0, 10.0, 1.0],
                [30.0, 20.0, 2.0],
                [40.0, 30.0, 3.0],
            ]
        ),
    )

    np.testing.assert_array_equal(
        ky,
        np.array(
            [
                [10.0, 1.0],
                [20.0, 2.0],
                [30.0, 3.0],
            ]
        ),
    )

    np.testing.assert_array_equal(
        xk,
        np.array(
            [
                [20.0, 10.0],
                [30.0, 20.0],
                [40.0, 30.0],
            ]
        ),
    )

    np.testing.assert_array_equal(
        k_space,
        np.array(
            [
                [10.0],
                [20.0],
                [30.0],
            ]
        ),
    )

    assert n_points == 3


def test_make_spaces_embedding_two():
    """Document the exact spaces currently built for embedding=2."""
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    xky, ky, xk, k_space, n_points = helper.make_spaces(
        x,
        y,
        embedding=2,
    )

    np.testing.assert_array_equal(
        xky,
        np.array(
            [
                [30.0, 20.0, 10.0, 2.0],
                [40.0, 30.0, 20.0, 3.0],
                [50.0, 40.0, 30.0, 4.0],
            ]
        ),
    )

    np.testing.assert_array_equal(
        ky,
        np.array(
            [
                [20.0, 10.0, 2.0],
                [30.0, 20.0, 3.0],
                [40.0, 30.0, 4.0],
            ]
        ),
    )

    np.testing.assert_array_equal(
        xk,
        np.array(
            [
                [30.0, 20.0, 10.0],
                [40.0, 30.0, 20.0],
                [50.0, 40.0, 30.0],
            ]
        ),
    )

    np.testing.assert_array_equal(
        k_space,
        np.array(
            [
                [20.0, 10.0],
                [30.0, 20.0],
                [40.0, 30.0],
            ]
        ),
    )

    assert n_points == 3

@pytest.mark.xfail(
    strict=True,
    reason="CPU implementation currently receives one-element arrays instead of scalars",
)
def test_column_vectors_run_without_safety_check():
    """Document that N x 1 arrays already work when safetyCheck is off."""
    rng = np.random.RandomState(seed=41)
    x = rng.randn(200, 1)
    y = rng.randn(200, 1)

    result = te.te_compute(
        x,
        y,
        k=1,
        embedding=1,
        safetyCheck=False,
        GPU=False,
    )

    assert np.isfinite(result)


@pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason="safetyCheck cannot currently hash rows from N x 1 arrays",
)
def test_column_vectors_run_with_safety_check():
    """The documented N x 1 input shape should also support safetyCheck."""
    rng = np.random.RandomState(seed=42)
    x = rng.randn(200, 1)
    y = rng.randn(200, 1)

    result = te.te_compute(
        x,
        y,
        k=1,
        embedding=1,
        safetyCheck=True,
        GPU=False,
    )

    assert np.isfinite(result)


def test_duplicate_pairs_stop_estimation(capsys):
    """A repeated (X, Y) pair should trigger the current safety behavior."""
    x = np.array([0.0, 1.0, 1.0, 2.0])
    y = np.array([3.0, 4.0, 4.0, 5.0])

    result = te.te_compute(
        x,
        y,
        safetyCheck=True,
        GPU=False,
    )

    captured = capsys.readouterr()

    assert result is None
    assert "duplicates" in captured.out.lower()


def test_nan_input_is_rejected():
    """NaN data should not silently produce a transfer-entropy value."""
    x = np.arange(20, dtype=float)
    y = np.arange(20, dtype=float)
    y[5] = np.nan

    with pytest.raises(ValueError):
        te.te_compute(
            x,
            y,
            safetyCheck=False,
            GPU=False,
        )


def test_embedding_two_executes_on_cpu():
    """Embedding=2 should at least complete mechanically on valid data."""
    rng = np.random.RandomState(seed=43)
    x = rng.randn(500)
    y = rng.randn(500)

    result = te.te_compute(
        x,
        y,
        k=1,
        embedding=2,
        safetyCheck=False,
        GPU=False,
    )

    assert np.isfinite(result)