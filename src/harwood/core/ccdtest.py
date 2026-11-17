"""ccdtest, the test package of IRAF's ccdred: mkimage, which makes or edits an image of
simple values -- a sloped plane and Gaussian noise -- and what the CL scripts artobs and
subsection make with it: a CCD observation, with its bias, dark, flat and bad pixels, for
ccdred's demonstration (setinstrument demo).

From noao/imred/ccdred/ccdtest/t_mkimage.x (t_mkimage, mkline, mksigma), artobs.cl and
subsection.cl, in real as f2c compiles them. The noise is only nearly random, as IRAF's:
a row of 2 x ncols Gaussian numbers made at each call, and each line takes ncols of them
from a place urand picks. urand's seed is mkimage's static: it carries from call to call
(within one harwood, as within one of IRAF's processes), unless a seed is given.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT. mksigma's
Box-Muller is Copyright (c) 2017 Anastasia Galkin, under the same terms in IRAF.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import artdata

f32 = np.float32
PI = f32(3.1415926535897932385)
OPTIONS = ("make", "replace", "add", "multiply")
#: mkimage's pixel types, by the first letter of pixtype.
PIXTYPES = {"s": np.int16, "i": np.int32, "l": np.int32, "r": np.float32, "d": np.float64}

#: t_mkimage's `data seed/1/`: one generator for every call.
_STATE = artdata.new_state(1)[0]


def seed(value: int) -> None:
    """A seed above zero starts the sequence again there; zero leaves it."""
    if value > 0:
        _STATE[0] = value


def reset() -> None:
    """As a new IRAF process: the static seed back to 1."""
    _STATE[0] = 1


def _urand() -> np.float32:
    return artdata.urand(_STATE)


def mksigma(sigma: float, n: int) -> np.ndarray:
    """mksigma: n Gaussian numbers of the sigma, two from each pair of urand's."""
    out = np.zeros(n, f32)
    if sigma <= 0:
        return out
    sigma = f32(sigma)
    i = 0
    while i < n:
        u1 = f32(f32(1) - _urand())
        u2 = _urand()
        root = math.sqrt(-2.0 * math.log(float(u1)))
        angle = float(f32(f32(f32(2) * PI) * u2))
        out[i] = f32(f32(root * math.cos(angle)) * sigma)
        if i == n - 1:
            break
        out[i + 1] = f32(f32(root * math.sin(angle)) * sigma)
        i += 2
    return out


def mkline(value, slope, sigma, rannums: np.ndarray, line: int, ncols: int,
           nlines: int) -> np.ndarray:
    """mkline: one line of the plane -- value at the plane's middle, slope a pixel in
    each direction -- and, with a sigma, ncols of the stored noise from a random start."""
    value, slope = f32(value), f32(slope)
    if slope == 0:
        data = np.full(ncols, value, f32)
    else:
        middle = f32(f32(ncols + nlines) / f32(2))
        a = f32(value + f32(slope * f32(f32(f32(line) - middle) - f32(1))))
        data = np.array([f32(a + f32(slope * f32(i))) for i in range(1, ncols + 1)], f32)
    if sigma > 0:
        start = int(f32(f32(f32(ncols - 1) * _urand()) + f32(1)))
        data = (data + rannums[start - 1:start - 1 + ncols]).astype(f32)
    return data


def reduced(shape: tuple[int, ...]) -> tuple[int, ...]:
    """An image section's shape as IRAF opens it: the axes one pixel long dropped --
    a[15:15,*] is a line of 15 pixels, not 15 lines of one."""
    kept = tuple(n for n in shape if n != 1)
    return kept or (1,)


def plane(option: str, current: np.ndarray, value, slope, sigma,
          new: bool = False) -> np.ndarray:
    """What one call of mkimage makes of an image (or a section of one, its one-pixel
    axes dropped first), line by line in IRAF's order, the planes of a cube one after
    another: replace, add or multiply, in real. `current` is the pixels as real; `new`,
    an image being made.

    The line mkline is told is impnlr's vector, after the line, less one: of a 1-D image
    0 (the vector has no second axis to advance), and in a cube 0 at each plane's last
    line (the second axis has wrapped to 1). A new 1-D image's second length is 0,
    IMIO not having set it."""
    full = current.shape
    shape = reduced(full)
    ncols = shape[-1]
    ndim = len(shape)
    # IM_LEN(im,2) of a 1-D image: of a section, the section's second length before its
    # one-pixel axes went (a column of a frame keeps the frame's lines); of a new image,
    # nothing
    nlines = shape[-2] if ndim > 1 else (0 if new else (full[-2] if len(full) > 1
                                                        else 1))
    rannums = mksigma(sigma, 2 * ncols)
    lines = np.asarray(current, f32).reshape(-1, ncols)
    out = np.empty_like(lines)
    for k in range(lines.shape[0]):
        if ndim == 1:
            line = 0
        else:
            line = k % nlines + 1
            if ndim > 2 and line == nlines:
                line = 0
        made = mkline(value, slope, sigma, rannums, line, ncols, nlines)
        if option in ("make", "replace"):
            out[k] = made
        elif option == "add":
            out[k] = (lines[k] + made).astype(f32)
        else:
            out[k] = (lines[k] * made).astype(f32)
    return out.reshape(full)


def stored(values: np.ndarray, dtype) -> np.ndarray:
    """Real pixels as IMIO writes them into an image of `dtype`: integers truncated
    toward zero, as a Fortran assignment does."""
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return np.asarray(values, dtype)
    return np.trunc(np.asarray(values, np.float64)).astype(np.int64).astype(dtype)


def read_regions(path: Path | str) -> list[tuple[int, int, int, int]]:
    """artobs's reading of a bad pixel file: the lines with four numbers, c1 c2 l1 l2."""
    found = []
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        words = line.split()
        try:
            numbers = [int(w) for w in words[:4]]
        except ValueError:
            continue
        if len(numbers) == 4:
            found.append(tuple(numbers))
    return found
