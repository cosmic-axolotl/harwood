"""boxcar without the task: images/imfilter/src/boxcar.x and aboxcar.x, as IRAF
2.18 runs them.

The mean over an xwindow by ywindow box, kept as running sums in single
precision: a column sum that adds each new line and subtracts the one leaving,
and along the line a sum that adds the pixel coming and subtracts the one
going. The rounding of those sums is part of IRAF's answer, and is kept.

Unlike median, the window is not made odd. An even side reaches one pixel
further on one side than the other -- xwindow/2 pixels to the left and
xwindow/2 - 1 to the right, ywindow/2 lines below and ywindow/2 - 1 above --
so an even box shifts the image by half a pixel. The pixels beyond the image
come from IMIO's boundary extension and, off an image section, from the image
itself, as median reads them (core/median.py, `extended`).
"""

from __future__ import annotations

import numpy as np

from . import median
from .jit import jit

f32 = np.float32


class BoxcarError(ValueError):
    """What IRAF's boxcar stops on, in its words."""


def smooth(image: np.ndarray, xwindow: int, ywindow: int, boundary: str = "nearest",
           constant: float = 0.0, section=None) -> np.ndarray:
    """cnv_boxcar of `image` (read as reals), or of a section of it, in single
    precision; `constant` already in the image's pixel type."""
    image = np.asarray(image, dtype=f32)
    if image.ndim > 2:
        raise BoxcarError("T_CONVOLVE: Image dimension > 2.")
    nxk = int(xwindow)
    nyk = 1 if image.ndim == 1 else int(ywindow)
    try:
        ext, ny, nx = median.extended(image, section, nxk if nxk % 2 else nxk + 1,
                                      nyk if nyk % 2 else nyk + 1, boundary, constant)
    except median.MedianError as error:
        raise BoxcarError(str(error)) from None
    out = _boxcar(ext, ny, nx, nxk, nyk)
    return out[0] if image.ndim == 1 else out


@jit
def _boxcar(ext, nlines, ncols, nxk, nyk):
    """The lines from 1 - nyk/2 and the columns from 1 - nxk/2 are ext's first,
    whichever way nxk and nyk are odd."""
    width = ncols + nxk - 1
    accum = np.zeros(width, dtype=np.float32)
    out = np.zeros((nlines, ncols), dtype=np.float32)
    norm = np.float32(nxk * nyk)
    for i in range(nyk - 1):
        for c in range(width):
            accum[c] = accum[c] + ext[i, c]
    for line in range(nlines):
        first = line
        new = first + nyk - 1
        for c in range(width):
            accum[c] = accum[c] + ext[new, c]
        total = np.float32(0.0)
        for c in range(nxk - 1):
            total = total + accum[c]
        for c in range(ncols):
            total = total + accum[c + nxk - 1]
            out[line, c] = total
            total = total - accum[c]
        for c in range(ncols):
            out[line, c] = out[line, c] / norm
        for c in range(width):
            accum[c] = accum[c] - ext[first, c]
    return out
