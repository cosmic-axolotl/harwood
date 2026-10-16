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

from . import jit

f32 = np.float32

MIN_WIDTH = 3.0
EPSILON = 0.001
EPSILON1 = 0.005
ITERATIONS = 100
MAX_DXCHECK = 3


def center1d(x: float, data, width: float, kind: str = "emission",
             radius: float = 10.0, threshold: float = 0.0, interp: str = "spline3",
             epsilon: float = EPSILON) -> float | None:
    """The centre of the feature near `x` in `data`, 1-based, or None.

    `interp` and `epsilon` are c1d_params: IRAF's default is a cubic spline
    to 0.001 px; autoidentify searches with a linear interpolant to 0.02.
    """
    if kind not in ("emission", "absorption"):
        raise ValueError("Unknown feature type")
    if x is None or not np.isfinite(x):
        return None
    data = data if (isinstance(data, np.ndarray) and data.dtype == f32) \
        else np.asarray(data, dtype=f32)
    xc = _center1d_kernel(f32(x), data, f32(width), kind == "emission", f32(radius),
                          f32(threshold), interp == "linear", f32(epsilon))
    return None if np.isnan(xc) else float(xc)


@jit.jit
def _center1d_kernel(x, data, width, emission, radius, threshold, linear, epsilon):
    """center1d: window, continuum and threshold, then c1d_center. NaN for
    INDEF."""
    npts = data.shape[0]
    if x < 1 or x > npts:
        return np.float32(np.nan)
    wid = np.float32(max(width, np.float32(MIN_WIDTH)))
    rad = np.float32(max(np.float32(2.0), radius))
    two = np.float32(2.0)
    x1 = int(max(np.float32(1.0), np.float32(x - wid / two - rad - wid)))
    x2 = int(min(np.float32(npts), np.float32(x + wid / two + rad + wid + np.float32(1.0))))
    a = data[x1 - 1]
    b = data[x1 - 1]
    for k in range(x1 - 1, x2):
        if data[k] < a:
            a = data[k]
        if data[k] > b:
            b = data[k]
    if b - a < threshold:
        return np.float32(np.nan)
    x1 = int(max(np.float32(1.0), np.float32(x - wid / two - rad)))
    x2 = int(min(np.float32(npts), np.float32(x + wid / two + rad + np.float32(1.0))))
    nx = x2 - x1 + 1
    work = np.zeros(nx, dtype=np.float32)
    if emission:
        if a > 0:
            a = np.float32(0.0)
        for k in range(nx):
            v = np.float32(data[x1 - 1 + k] - np.float32(a + threshold))
            work[k] = v if v > 0 else np.float32(0.0)
    else:
        for k in range(nx):
            v = np.float32(-data[x1 - 1 + k] - np.float32(threshold - b))
            work[k] = v if v > 0 else np.float32(0.0)
    xc = _center_kernel(np.float32(x - np.float32(x1) + np.float32(1.0)), work, width,
                        linear, epsilon)
    if np.isnan(xc):
        return xc
    xc = np.float32(xc + np.float32(x1) - np.float32(1.0))
    if abs(x - xc) > radius:
        return np.float32(np.nan)
    return xc


def _center(x, data, width, interp="spline3", epsilon=EPSILON):
    """c1d_center on the prepared window (1-based x); None for INDEF."""
    xc = _center_kernel(np.float32(x), np.asarray(data, dtype=f32), np.float32(width),
                        interp == "linear", np.float32(epsilon))
    return None if np.isnan(xc) else f32(xc)


# The kernel and the interpolator below are IRAF's asifit/asigrl and
# c1d_center written as plain loops over arrays, so numba can compile them;
# core.asi holds the same interpolator as a class for everyone else.

@jit.jit
def _coefficients(data, linear):
    """asifit: COEFF[k] at index k; linear: data then one extension,
    spline3: natural-end B-spline coefficients."""
    n = data.shape[0]
    c = np.zeros(n + 4, dtype=np.float32)
    if linear:
        for k in range(n):
            c[k + 1] = data[k]
        c[n + 1] = np.float32(2.0) * data[n - 1] - data[n - 2]
        return c
    for k in range(n):
        c[k + 2] = data[k]
    # ii_spline on COEFF[1..n+2]; b[i] is COEFF[i] here (1-based).
    diag = np.zeros(n + 3, dtype=np.float32)
    diag[1] = np.float32(-2.0)
    c[1] = c[1] / np.float32(6.0)
    diag[2] = np.float32(0.0)
    c[2] = (c[2] - c[1]) / np.float32(6.0)
    for i in range(3, n + 2):
        diag[i] = np.float32(1.0) / (np.float32(4.0) - diag[i - 1])
        c[i] = diag[i] * (c[i] - c[i - 1])
    c[n + 2] = ((diag[n] + np.float32(2.0)) * c[n + 1] - c[n] + c[n + 2] / np.float32(6.0)) \
        / (np.float32(1.0) + diag[n + 1] * (diag[n] + np.float32(2.0)))
    for i in range(n + 1, 2, -1):
        c[i] = c[i] - diag[i] * c[i + 1]
    c[1] = c[1] + np.float32(2.0) * c[2] - c[3]
    return c


@jit.jit
def _pieces(c, index):
    a = c[index - 1]
    b = c[index]
    cc = c[index + 1]
    d = c[index + 2]
    p0 = np.float32(a + np.float32(4.0) * b + cc)
    p1 = np.float32(np.float32(3.0) * (cc - a))
    p2 = np.float32(np.float32(3.0) * (a - np.float32(2.0) * b + cc))
    p3 = np.float32(-a + np.float32(3.0) * b - np.float32(3.0) * cc + d)
    return p0, p1, p2, p3


@jit.jit
def _pow(x, i):
    out = np.float32(1.0)
    for _ in range(i):
        out = np.float32(out * x)
    return out


@jit.jit
def _integral(c, linear, a, b):
    """asigrl from a to b."""
    if a <= b:
        xa, xb = np.float32(a), np.float32(b)
    else:
        xa, xb = np.float32(b), np.float32(a)
    neara = int(xa)
    nearb = int(xb)
    accum = np.float32(0.0)
    da = np.float32(xa - np.float32(neara))
    half = np.float32(0.5)
    one = np.float32(1.0)
    if linear:
        if nearb == neara:
            db = np.float32(xb - np.float32(nearb))
            accum = np.float32(accum + (db - da) * c[neara]
                               + half * (c[neara + 1] - c[neara]) * (db * db - da * da))
        else:
            accum = np.float32(accum + (one - da) * c[neara]
                               + half * (c[neara + 1] - c[neara]) * (one - da * da))
            for j in range(neara + 1, nearb):
                accum = np.float32(accum + half * (c[j + 1] + c[j]))
            db = np.float32(xb - np.float32(nearb))
            accum = np.float32(accum + c[nearb] * db
                               + half * (c[nearb + 1] - c[nearb]) * db * db)
    else:
        if nearb == neara:
            db = np.float32(xb - np.float32(nearb))
            p = _pieces(c, 1 + neara)
            for i in range(1, 5):
                accum = np.float32(accum + np.float32(1.0 / i) * p[i - 1]
                                   * np.float32(_pow(db, i) - _pow(da, i)))
        else:
            p = _pieces(c, 1 + neara)
            for i in range(1, 5):
                accum = np.float32(accum + np.float32(1.0 / i) * p[i - 1]
                                   * np.float32(one - _pow(da, i)))
            for j in range(neara + 1, nearb):
                p = _pieces(c, 1 + j)
                for i in range(1, 5):
                    accum = np.float32(accum + np.float32(1.0 / i) * p[i - 1])
            db = np.float32(xb - np.float32(nearb))
            p = _pieces(c, 1 + nearb)
            for i in range(1, 5):
                accum = np.float32(accum + np.float32(1.0 / i) * p[i - 1] * _pow(db, i))
    return accum if a < b else np.float32(-accum)


@jit.jit
def _center_kernel(x, data, width, linear, epsilon):
    """c1d_center; NaN for INDEF."""
    npts = data.shape[0]
    i = int(x + np.float32(0.5))
    while i < npts and data[i - 1] <= data[i]:
        i += 1
    while i > 1 and data[i - 1] <= data[i - 2]:
        i -= 1
    j = int(x + np.float32(0.5))
    while j > 1 and data[j - 1] <= data[j - 2]:
        j -= 1
    while j < npts and data[j - 1] <= data[j]:
        j += 1
    xc = np.float32(i) if abs(np.float32(i) - x) < abs(x - np.float32(j)) \
        else np.float32(j)
    if width <= 1.0:
        return xc
    wid = np.float32(max(width, np.float32(MIN_WIDTH)))
    hwidth = np.float32(wid / np.float32(2.0))
    if xc - hwidth < 1 or xc + hwidth > npts:
        return np.float32(np.nan)
    c1 = _coefficients(data, linear)
    weighted = np.zeros(npts, dtype=np.float32)
    for k in range(npts):
        weighted[k] = np.float32(data[k] * np.float32(k + 1))
    c2 = _coefficients(weighted, linear)
    dxlast = np.float32(npts)
    dxcheck = 0
    two = np.float32(2.0)
    for _ in range(ITERATIONS):
        a = np.float32(xc - hwidth)
        b = np.float32(xc - hwidth / two)
        g1 = _integral(c1, linear, a, b)
        g2 = _integral(c2, linear, a, b)
        sum1 = np.float32((xc - hwidth) * g1 - g2)
        sum2 = np.float32(-g1)
        a = b
        b = np.float32(xc + hwidth / two)
        g1 = _integral(c1, linear, a, b)
        g2 = _integral(c2, linear, a, b)
        sum1 = np.float32(sum1 - xc * g1 + g2)
        sum2 = np.float32(sum2 + g1)
        a = b
        b = np.float32(xc + hwidth)
        g1 = _integral(c1, linear, a, b)
        g2 = _integral(c2, linear, a, b)
        sum1 = np.float32(sum1 + (xc + hwidth) * g1 - g2)
        sum2 = np.float32(sum2 - g1)
        if sum2 == 0:
            return np.float32(np.nan)
        dx = np.float32(sum1 / abs(sum2))
        dxabs = abs(dx)
        step = dx
        if step > 1:
            step = np.float32(1.0)
        elif step < -1:
            step = np.float32(-1.0)
        xc = np.float32(xc + step)
        if xc - hwidth < 1 or xc + hwidth > npts:
            return np.float32(np.nan)
        if dxabs < epsilon:
            return xc
        if dxabs > dxlast + np.float32(EPSILON1):
            dxcheck += 1
            if dxcheck > MAX_DXCHECK:
                return np.float32(np.nan)
        elif dxabs > dxlast - np.float32(EPSILON1):
            xc = np.float32(xc - step / two)
            dxcheck = 0
        else:
            dxcheck = 0
            dxlast = dxabs
    return np.float32(np.nan)
