"""What IRAF's graphics do to a window, where a panel has to do the same."""

from __future__ import annotations


def widened(low: float, high: float) -> tuple[float, float]:
    """A window with no width: an axis on which every point has one value, such as
    the airmass of one image. Found from IRAF's metacode, at 0, 1e-6, 0.05, 0.5, 1,
    1.5, 5 and on real data."""
    if low != high:
        return low, high
    half = 0.01 * max(abs(low), 1.0)
    return low - half, high + half
