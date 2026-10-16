"""IRAF's center1d (xtools), ported.

The centring every one-dimensional feature in IRAF goes through: apfind and
aptrace centre spectra across the slit with it, identify centres arc lines.

From iraf-community 2.18, pkg/xtools/center1d.x. In IRAF's order:

1. the data range around the guess must exceed `threshold`;
2. a window of the guess +/- (width/2 + radius) is taken, its minimum (or
   zero, if that is lower) plus the threshold subtracted, and clipped at 0
   (an absorption feature is negated first);
3. starting from the nearest local maximum, the centre is moved until the
   data, weighted by a triangle of the given width, balance on either side
   -- the integrals of an interpolating cubic spline through the data, so
   the answer is not tied to pixel boundaries;
4. INDEF (None here) if it runs off the data, does not converge in 100
   steps, or ends further than `radius` from the guess.

Positions are 1-based. Single precision, as IRAF.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import numpy as np

from .asi import Interpolant

f32 = np.float32

MIN_WIDTH = 3.0
EPSILON = 0.001
EPSILON1 = 0.005
ITERATIONS = 100
MAX_DXCHECK = 3


def center1d(x: float, data, width: float, kind: str = "emission",
             radius: float = 10.0, threshold: float = 0.0) -> float | None:
    """The centre of the feature near `x` in `data`, 1-based, or None."""
    data = np.asarray(data, dtype=f32)
    npts = data.size
    if x is None or not np.isfinite(x) or x < 1 or x > npts:
        return None
    x = f32(x)  # type: ignore[assignment]
    wid = f32(max(width, MIN_WIDTH))
    rad = f32(max(2.0, radius))

    x1 = int(max(f32(1), f32(x - wid / f32(2) - rad - wid)))
    x2 = int(min(f32(npts), f32(x + wid / f32(2) + rad + wid + f32(1))))
    window = data[x1 - 1:x2]
    a, b = f32(window.min()), f32(window.max())
    if b - a < f32(threshold):
        return None

    x1 = int(max(f32(1), f32(x - wid / f32(2) - rad)))
    x2 = int(min(f32(npts), f32(x + wid / f32(2) + rad + f32(1))))
    piece = data[x1 - 1:x2]
    if kind == "emission":
        a = min(f32(0), a)
        work = np.maximum(f32(piece - (a + f32(threshold))), f32(0)).astype(f32)
    elif kind == "absorption":
        work = np.maximum(f32(-piece - f32(f32(threshold) - b)), f32(0)).astype(f32)
    else:
        raise ValueError("Unknown feature type")

    xc = _center(f32(x - f32(x1) + f32(1)), work, width)
    if xc is None:
        return None
    xc = f32(xc + f32(x1) - f32(1))
    if abs(x - xc) > radius:
        return None
    return float(xc)


def _center(x, data, width):
    """c1d_center on the prepared window (1-based x)."""
    npts = data.size
    d = np.concatenate(([f32(0)], data))        # 1-based
    i = int(x + f32(0.5))
    while i < npts and d[i] <= d[i + 1]:
        i += 1
    while i > 1 and d[i] <= d[i - 1]:
        i -= 1
    j = int(x + f32(0.5))
    while j > 1 and d[j] <= d[j - 1]:
        j -= 1
    while j < npts and d[j] <= d[j + 1]:
        j += 1
    xc = f32(i) if abs(f32(i) - x) < abs(x - f32(j)) else f32(j)
    if width <= 1.0:
        return xc

    wid = f32(max(width, MIN_WIDTH))
    hwidth = f32(wid / f32(2))
    if xc - hwidth < 1 or xc + hwidth > npts:
        return None
    asi1 = Interpolant("spline3", data)
    asi2 = Interpolant("spline3", (data * np.arange(1, npts + 1, dtype=f32)).astype(f32))

    dxlast = f32(npts)
    dxcheck = 0
    for _ in range(ITERATIONS):
        a = f32(xc - hwidth)
        b = f32(xc - hwidth / f32(2))
        g1, g2 = f32(asi1.integral(a, b)), f32(asi2.integral(a, b))
        sum1 = f32((xc - hwidth) * g1 - g2)
        sum2 = f32(-g1)
        a, b = b, f32(xc + hwidth / f32(2))
        g1, g2 = f32(asi1.integral(a, b)), f32(asi2.integral(a, b))
        sum1 = f32(sum1 - xc * g1 + g2)
        sum2 = f32(sum2 + g1)
        a, b = b, f32(xc + hwidth)
        g1, g2 = f32(asi1.integral(a, b)), f32(asi2.integral(a, b))
        sum1 = f32(sum1 + (xc + hwidth) * g1 - g2)
        sum2 = f32(sum2 - g1)
        if sum2 == 0:
            return None
        dx = f32(sum1 / abs(sum2))
        dxabs = abs(dx)
        xc = f32(xc + max(f32(-1), min(f32(1), dx)))
        if xc - hwidth < 1 or xc + hwidth > npts:
            return None
        if dxabs < EPSILON:
            return xc
        if dxabs > dxlast + f32(EPSILON1):
            dxcheck += 1
            if dxcheck > MAX_DXCHECK:
                return None
        elif dxabs > dxlast - f32(EPSILON1):
            xc = f32(xc - max(f32(-1), min(f32(1), dx)) / f32(2))
            dxcheck = 0
        else:
            dxcheck = 0
            dxlast = dxabs
    return None
