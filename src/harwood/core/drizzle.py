"""Drizzle: Fruchter & Hook (2002, PASP 114, 144). Frames dithered by fractions of a pixel,
onto a finer grid, with what a pixel saw shrunk to a smaller "drop" before it lands: resolution
an undersampled camera loses to its pixels comes back, and the flux is kept. IRAF has imshift
and imcombine, or geotran: each interpolates, and smooths what it interpolates.

Each input pixel, shrunk to `pixfrac` of its side about its centre, is mapped to the output by
its four corners. An output pixel takes it by the area of their overlap, a weighted mean of
what lands there. Noise in the result is correlated between neighbours, since a drop can fall
on more than one, and what an output pixel shows understates it: `noise_correlation` is
Casertano et al. (2000)'s factor for that.
"""

from __future__ import annotations

import math

import numpy as np

from .jit import jit


@jit
def _clamped_integral(x1, y1, x2, y2):
    """The area under the segment, in [0, 1] in both x and y, with the sign of its direction in
    x: what an edge of a polygon adds to its area within the unit square."""
    dx = x2 - x1
    if dx == 0.0:
        return 0.0
    sign = 1.0
    if dx < 0.0:
        sign = -1.0
        x1, y1, x2, y2 = x2, y2, x1, y1
        dx = -dx
    if x1 >= 1.0 or x2 <= 0.0:
        return 0.0
    slope = (y2 - y1) / dx
    start = y1 - slope * x1
    low, high = max(x1, 0.0), min(x2, 1.0)
    points = [low, high]
    if slope != 0.0:
        for level in (0.0, 1.0):
            crossing = (level - start) / slope
            if low < crossing < high:
                points.append(crossing)
    points.sort()
    total = 0.0
    for at in range(len(points) - 1):
        a, b = points[at], points[at + 1]
        ya = min(max(slope * a + start, 0.0), 1.0)
        yb = min(max(slope * b + start, 0.0), 1.0)
        total += 0.5 * (ya + yb) * (b - a)
    return sign * total


@jit
def overlap(xs, ys, column, line):
    """The area a quadrilateral has within the output pixel centred at (column, line)."""
    total = 0.0
    for at in range(4):
        following = (at + 1) % 4
        total += _clamped_integral(xs[at] - column + 0.5, ys[at] - line + 0.5,
                                   xs[following] - column + 0.5, ys[following] - line + 0.5)
    return abs(total)


@jit
def _add(data, weights, corner_x, corner_y, out, sum_weight, scale):
    ny, nx = data.shape
    rows, columns = out.shape
    xs = np.zeros(4)
    ys = np.zeros(4)
    for j in range(ny):
        for i in range(nx):
            weight = weights[j, i]
            if weight <= 0.0 or data[j, i] != data[j, i]:
                continue
            for at in range(4):
                xs[at] = corner_x[at, j, i]
                ys[at] = corner_y[at, j, i]
            x1 = max(int(math.floor(xs.min() + 0.5)), 0)
            x2 = min(int(math.floor(xs.max() + 0.5)), columns - 1)
            y1 = max(int(math.floor(ys.min() + 0.5)), 0)
            y2 = min(int(math.floor(ys.max() + 0.5)), rows - 1)
            for line in range(y1, y2 + 1):
                for column in range(x1, x2 + 1):
                    area = overlap(xs, ys, column, line)
                    if area <= 0.0:
                        continue
                    amount = area * weight
                    out[line, column] += amount * data[j, i] * scale
                    sum_weight[line, column] += amount


def corners(transform, pixfrac: float, shape: tuple[int, int]):
    """Where the corners of each input pixel's drop land, in order round it. `transform` takes
    0-based x and y, arrays, to those of the output."""
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    half = pixfrac / 2.0
    out_x, out_y = np.zeros((4, ny, nx)), np.zeros((4, ny, nx))
    around = ((-half, -half), (half, -half), (half, half), (-half, half))
    for at, (dx, dy) in enumerate(around):
        out_x[at], out_y[at] = transform(x + dx, y + dy)
    return out_x, out_y


def shifted(dx: float, dy: float, scale: float):
    """A transform for a frame that is (dx, dy) in input pixels from the first: an output
    pixel being `scale` input pixels, its centres are where the first frame's corner is at
    -0.5."""
    def transform(x, y):
        return (x - dx + 0.5) / scale - 0.5, (y - dy + 0.5) / scale - 0.5

    return transform


class Grid:
    """What lands on the output, frame after frame."""

    def __init__(self, shape: tuple[int, int], pixfrac: float = 0.8,
                 scale: float = 0.5) -> None:
        self.pixfrac, self.scale = pixfrac, scale
        self.sum = np.zeros(shape, np.float64)
        self.weight = np.zeros(shape, np.float64)

    def add(self, data, transform, weights=None) -> None:
        """`data` in counts a pixel, and `weights` of the frame's pixels, zero for a bad one.
        A pixel of `scale` input pixels has scale^2 of the counts of an input one, which keeps
        the flux of a source."""
        data = np.asarray(data, dtype=np.float64)
        weights = np.ones(data.shape) if weights is None else np.asarray(weights, np.float64)
        corner_x, corner_y = corners(transform, self.pixfrac, data.shape)
        _add(data, weights, corner_x, corner_y, self.sum, self.weight, self.scale ** 2)

    @property
    def image(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.weight > 0, self.sum / self.weight, 0.0)


def noise_correlation(pixfrac: float, scale: float) -> float:
    """Casertano et al. (2000), their 1 / F_A: the true noise of a drizzled image over what its
    pixels show, for pixfrac p and scale s, an output pixel in input pixels. With r = s / p,
    F_A = 1 - 1 / (3 r) at r >= 1, and r (1 - r / 3) below it."""
    r = scale / pixfrac
    return 1.0 / (1.0 - 1.0 / (3.0 * r) if r >= 1.0 else r * (1.0 - r / 3.0))


def scaled_header(header, scale: float):
    """The header with its WCS on the output's pixels. Nothing to do where it has none."""
    out = header.copy()
    for axis in (1, 2):
        key = f"CRPIX{axis}"
        if key in out:
            out[key] = (out[key] - 0.5) / scale + 0.5
        if f"CDELT{axis}" in out:
            out[f"CDELT{axis}"] = out[f"CDELT{axis}"] * scale
        for other in (1, 2):
            key = f"CD{axis}_{other}"
            if key in out:
                out[key] = out[key] * scale
    return out


def wcs_transform(frame_header, output_header):
    """A transform by what two headers' WCS say: 0-based pixels to the sky and back."""
    from astropy.wcs import WCS

    frame = WCS(frame_header, naxis=2, relax=True)
    output = WCS(output_header, naxis=2, relax=True)

    def transform(x, y):
        ra, dec = frame.all_pix2world(x, y, 0)
        return output.all_world2pix(ra, dec, 0)

    return transform
