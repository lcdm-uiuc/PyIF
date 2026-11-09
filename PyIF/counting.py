
import numpy as np


def _legacy_radius_count(sorted_values, point, radius):
    """Count values using the exact legacy distance comparisons."""

    with np.errstate(over="ignore", invalid="ignore"):
        left = int(np.searchsorted(
            sorted_values, point - radius, side="left"
        ))
        right = int(np.searchsorted(
            sorted_values, point + radius, side="right"
        ))

    # Correct rounding differences at the interval boundaries.
    # Jump over groups of identical values to handle ties efficiently.
    while (
        left > 0
        and abs(point - sorted_values[left - 1]) <= radius
    ):
        left = int(np.searchsorted(
            sorted_values, sorted_values[left - 1], side="left"
        ))

    while (
        left < len(sorted_values)
        and abs(point - sorted_values[left]) > radius
    ):
        left = int(np.searchsorted(
            sorted_values, sorted_values[left], side="right"
        ))

    while (
        right < len(sorted_values)
        and abs(point - sorted_values[right]) <= radius
    ):
        right = int(np.searchsorted(
            sorted_values, sorted_values[right], side="right"
        ))

    while (
        right > left
        and abs(point - sorted_values[right - 1]) > radius
    ):
        right = int(np.searchsorted(
            sorted_values, sorted_values[right - 1], side="left"
        ))

    # The legacy implementation excludes exact-zero differences.
    zero_left = int(np.searchsorted(
        sorted_values, point, side="left"
    ))
    zero_right = int(np.searchsorted(
        sorted_values, point, side="right"
    ))

    zero_count = max(
        0,
        min(right, zero_right) - max(left, zero_left)
    )

    # The legacy implementation replaces zero counts with one.
    return max(1, right - left - zero_count)


def counts_sorted(
    out_xky,
    out_xk,
    radii_xky,
    radii_xk,
    xky_points,
    sorted_values,
):
    """Fill legacy target-count arrays using sorted observations."""

    for i in range(len(out_xky)):
        point = xky_points[i, 0]

        out_xky[i] = _legacy_radius_count(
            sorted_values, point, radii_xky[i]
        )
        out_xk[i] = _legacy_radius_count(
            sorted_values, point, radii_xk[i]
        )
