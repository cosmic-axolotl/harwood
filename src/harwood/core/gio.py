"""What IRAF's graphics do to a window, where a panel has to do the same."""

from __future__ import annotations

import numpy as np


def widened(low: float, high: float) -> tuple[float, float]:
    """A window with no width: an axis on which every point has one value, such as
    the airmass of one image. Found from IRAF's metacode, at 0, 1e-6, 0.05, 0.5, 1,
    1.5, 5 and on real data."""
    if low != high:
        return low, high
    half = 0.01 * max(abs(low), 1.0)
    return low - half, high + half


def with_room(values) -> tuple[float, float]:
    """gascale and gt_swind on one axis: the data, and 3% of its range around it."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return -1.0, 1.0
    low, high = float(values.min()), float(values.max())
    room = 0.03 * (high - low)
    return widened(low - room, high + room)
