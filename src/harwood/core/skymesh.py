"""The sky as a surface: the mesh background of SExtractor (Bertin & Arnouts 1996, A&AS 117,
393), which IRAF has no task for. Its fitsky measures an annulus around each star.

The image is cut into boxes. In each, the values are clipped at 3 sigma about the median
until they stop changing, and the level is the mode estimate 2.5 median - 1.5 mean, or the
median when mean and median disagree by more than 0.3 sigma, which says that the box is
crowded. A box that a mask takes most of, or that is mostly stars, gets its value from its
neighbours. The grid is median filtered, and a bicubic spline through it gives every pixel a
sky and a sigma.

Written here rather than taken from a package, so that every box can be shown, and held by the
tests to photutils' Background2D with its SExtractorBackground, box for box, and to a surface
known beforehand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import advice


@dataclass
class SkyModel:
    sky: np.ndarray          #: at every pixel
    rms: np.ndarray          #: the noise of the sky, at every pixel
    grid: np.ndarray         #: the box values, filtered
    grid_rms: np.ndarray
    raw: np.ndarray          #: before filtering, NaN where a box was left out
    centers_x: np.ndarray
    centers_y: np.ndarray
    replaced: int            #: the boxes that took their value from their neighbours


def clipped(values: np.ndarray, sigma: float = 3.0, maxiter: int = 10) -> np.ndarray:
    """Clipped about the median until nothing more goes."""
    values = values[np.isfinite(values)]
    for _ in range(maxiter):
        if values.size == 0:
            break
        centre, spread = np.median(values), np.std(values)
        keep = (values >= centre - sigma * spread) & (values <= centre + sigma * spread)
        if keep.all():
            break
        values = values[keep]
    return values


def level(values: np.ndarray) -> float:
    """SExtractor's estimate of the sky of a clipped box."""
    mean, median, std = np.mean(values), np.median(values), np.std(values)
    if std == 0 or abs(mean - median) / std >= 0.3:
        return float(median)
    return float(2.5 * median - 1.5 * mean)


def _fill(grid: np.ndarray) -> np.ndarray:
    """A box left out: the mean of the nearest ring of boxes that have a value."""
    out = grid.copy()
    bad = np.argwhere(~np.isfinite(grid))
    good = np.argwhere(np.isfinite(grid))
    if good.size == 0:
        advice.fail("every box of the mesh was left out",
                    "use a larger --box, or look at the mask: it takes most of the image")
    for j, i in bad:
        d = np.max(np.abs(good - [j, i]), axis=1)
        out[j, i] = grid[tuple(good[d == d.min()].T)].mean()
    return out


def _median_filter(grid: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return grid
    half = size // 2
    ny, nx = grid.shape
    out = np.empty_like(grid)
    for j in range(ny):
        for i in range(nx):
            out[j, i] = np.median(grid[max(0, j - half):j + half + 1,
                                       max(0, i - half):i + half + 1])
    return out


def _surface(grid: np.ndarray, xc: np.ndarray, yc: np.ndarray, shape) -> np.ndarray:
    from scipy.interpolate import RectBivariateSpline

    ny, nx = shape
    kx, ky = min(3, grid.shape[1] - 1), min(3, grid.shape[0] - 1)
    if kx < 1 or ky < 1:
        return np.full(shape, float(np.mean(grid)))
    spline = RectBivariateSpline(yc, xc, grid, kx=ky, ky=kx)
    return spline(np.arange(ny), np.arange(nx))


def model(data: np.ndarray, box: int = 64, filter_size: int = 3, mask=None,
          exclude: float = 0.5, crowded: float | None = None) -> SkyModel:
    """The sky of an image, whose pixels `mask` leaves out where True. A box with less than
    `exclude` of its pixels left is taken from its neighbours."""
    data = np.asarray(data, dtype=np.float64)
    ny, nx = data.shape
    if box < 3:
        advice.fail(f"a box of {box} pixels",
                    "boxes need room for sky between stars: 32 to 256")
    boxes_y, boxes_x = max(1, ny // box), max(1, nx // box)
    edges_y = np.linspace(0, ny, boxes_y + 1).astype(int)
    edges_x = np.linspace(0, nx, boxes_x + 1).astype(int)
    raw = np.full((boxes_y, boxes_x), np.nan)
    raw_rms = np.full((boxes_y, boxes_x), np.nan)
    for j in range(boxes_y):
        for i in range(boxes_x):
            cut = (slice(edges_y[j], edges_y[j + 1]), slice(edges_x[i], edges_x[i + 1]))
            values = data[cut]
            if mask is not None:
                values = values[~mask[cut]]
            values = values.ravel()
            size = (edges_y[j + 1] - edges_y[j]) * (edges_x[i + 1] - edges_x[i])
            values = values[np.isfinite(values)]
            if values.size < exclude * size:
                continue
            left = clipped(values)
            if left.size < 2 or (crowded is not None and left.size < crowded * size):
                continue
            raw[j, i] = level(left)
            raw_rms[j, i] = np.std(left)
    replaced = int((~np.isfinite(raw)).sum())
    filled = _fill(raw)
    filled_rms = _fill(raw_rms)
    grid = _median_filter(filled, filter_size)
    grid_rms = _median_filter(filled_rms, filter_size)
    yc = (edges_y[:-1] + edges_y[1:] - 1) / 2.0
    xc = (edges_x[:-1] + edges_x[1:] - 1) / 2.0
    return SkyModel(_surface(grid, xc, yc, data.shape), _surface(grid_rms, xc, yc, data.shape),
                    grid, grid_rms, raw, xc + 1, yc + 1, replaced)
