
import numpy as np

from PyIF import helper


def test_make_spaces_embedding_two():
    X = np.array([10., 11., 12., 13., 14.])
    Y = np.array([100., 101., 102., 103., 104.])

    xky, ky, xk, k, n_points = helper.make_spaces(
        X, Y, embedding=2
    )

    expected_xky = np.array([
        [12., 11., 10., 101.],
        [13., 12., 11., 102.],
        [14., 13., 12., 103.]
    ])

    expected_ky = np.array([
        [11., 10., 101.],
        [12., 11., 102.],
        [13., 12., 103.]
    ])

    expected_xk = np.array([
        [12., 11., 10.],
        [13., 12., 11.],
        [14., 13., 12.]
    ])

    expected_k = np.array([
        [11., 10.],
        [12., 11.],
        [13., 12.]
    ])

    np.testing.assert_array_equal(xky, expected_xky)
    np.testing.assert_array_equal(ky, expected_ky)
    np.testing.assert_array_equal(xk, expected_xk)
    np.testing.assert_array_equal(k, expected_k)

    assert n_points == 3
