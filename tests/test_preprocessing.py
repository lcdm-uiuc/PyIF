
import numpy as np
import pytest

from PyIF import helper
from PyIF.preprocessing import (
    prepare_target,
    prepare_source,
    combine_spaces,
)
from PyIF.preprocessing import count_tied_observations



@pytest.mark.parametrize("embedding", [1, 2, 3, 5])
def test_preprocessing_matches_legacy(embedding):
    rng = np.random.default_rng(123)

    target = rng.normal(size=50)
    source = rng.normal(size=50)

    legacy_spaces = helper.make_spaces(
        target, source, embedding=embedding
    )

    prepared_target = prepare_target(target, embedding)
    prepared_source = prepare_source(source, embedding)

    new_spaces = combine_spaces(
        prepared_target, prepared_source
    )

    for original, new in zip(legacy_spaces, new_spaces):
        np.testing.assert_array_equal(original, new)


def test_prepared_target_reusable_across_sources():
    rng = np.random.default_rng(456)

    target = rng.normal(size=100)
    source_a = rng.normal(size=100)
    source_b = rng.normal(size=100)

    prepared_target = prepare_target(target, embedding=2)

    spaces_a = combine_spaces(
        prepared_target,
        prepare_source(source_a, embedding=2)
    )

    spaces_b = combine_spaces(
        prepared_target,
        prepare_source(source_b, embedding=2)
    )

    np.testing.assert_array_equal(spaces_a[2], spaces_b[2])
    np.testing.assert_array_equal(spaces_a[3], spaces_b[3])

    assert spaces_a[2] is prepared_target.xk
    assert spaces_b[2] is prepared_target.xk


def test_count_tied_observations():
    assert count_tied_observations(
        np.array([1, 1, 2, 3, 3, 3, 4])
    ) == 5

    assert count_tied_observations(
        np.array([1, 2, 3, 4])
    ) == 0

    assert count_tied_observations(
        np.array([[1], [1], [2], [3]])
    ) == 2
