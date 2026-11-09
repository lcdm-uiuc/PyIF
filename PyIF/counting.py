
import numpy as np
from numba import njit


@njit(cache=True)
def _legacy_radius_count(sorted_values, point, radius):
    """Count sorted values using exact legacy distance comparisons."""

    left = np.searchsorted(
        sorted_values, point - radius, side="left"
    )
    right = np.searchsorted(
        sorted_values, point + radius, side="right"
    )

    # Correct floating-point rounding at the left boundary.
    while (
        left > 0
        and abs(point - sorted_values[left - 1]) <= radius
    ):
        left = np.searchsorted(
            sorted_values, sorted_values[left - 1], side="left"
        )

    while (
        left < len(sorted_values)
        and abs(point - sorted_values[left]) > radius
    ):
        left = np.searchsorted(
            sorted_values, sorted_values[left], side="right"
        )

    # Correct floating-point rounding at the right boundary.
    while (
        right < len(sorted_values)
        and abs(point - sorted_values[right]) <= radius
    ):
        right = np.searchsorted(
            sorted_values, sorted_values[right], side="right"
        )

    while (
        right > left
        and abs(point - sorted_values[right - 1]) > radius
    ):
        right = np.searchsorted(
            sorted_values, sorted_values[right - 1], side="left"
        )

    # Exclude all observations with exactly zero difference.
    zero_left = np.searchsorted(
        sorted_values, point, side="left"
    )
    zero_right = np.searchsorted(
        sorted_values, point, side="right"
    )

    zero_count = max(
        0,
        min(right, zero_right) - max(left, zero_left)
    )

    return max(1, right - left - zero_count)


@njit(cache=True)
def counts_sorted(
    out_xky,
    out_xk,
    radii_xky,
    radii_xk,
    xky_points,
    sorted_values,
):
    """Fill target-count arrays using cached sorted observations."""

    for i in range(len(out_xky)):
        point = xky_points[i, 0]

        out_xky[i] = _legacy_radius_count(
            sorted_values, point, radii_xky[i]
        )

        out_xk[i] = _legacy_radius_count(
            sorted_values, point, radii_xk[i]
        )
