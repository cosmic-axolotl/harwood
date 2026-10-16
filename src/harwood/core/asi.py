"""IRAF's one-dimensional image interpolator (math/iminterp, asi*), ported.

The piece that center1d integrates and apall weights its aperture edges
with. Two interpolants are needed: linear, and spline3 -- the interpolating
cubic B-spline with natural end conditions (second derivative zero), fitted
by ii_spline. Evaluation (asieval) and integration (asigrl) follow ii_eval.x
and asigrl.x, in single precision as IRAF computes them.

Coordinates are IRAF's: the first point is at x = 1.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import numpy as np

f32 = np.float32


class Interpolant:
    """asiinit + asifit: an interpolant of `data` (1-based), linear or spline3."""

    def __init__(self, kind: str, data):
        data = np.asarray(data, dtype=f32)
        n = data.size
        self.kind = kind
        self.n = n
        if kind == "linear":
            if n < 2:
                raise ValueError("ASIFIT: too few points for LINEAR")
            # coeff[1..n] the data, coeff[n+1] extended; stored 0-based with
            # a leading pad so that coeff[k] is IRAF's COEFF[k].
            c = np.zeros(n + 2, dtype=f32)
            c[1:n + 1] = data
            c[n + 1] = f32(2) * data[n - 1] - data[n - 2]
            self.coeff = c
        elif kind == "spline3":
            if n < 4:
                raise ValueError("ASIFIT: too few points for SPLINE3")
            # COEFF[1] and COEFF[n+2], COEFF[n+3] are the natural end
            # conditions; ii_spline turns COEFF[1..n+2] into B-spline
            # coefficients in place.
            c = np.zeros(n + 4, dtype=f32)
            c[2:n + 2] = data
            c[1:n + 3] = _spline(c[1:n + 3], n)
            self.coeff = c
        else:
            raise ValueError(f"interpolant '{kind}' is not ported")

    def __call__(self, x: float) -> float:
        """asieval at one point."""
        x = f32(x)  # type: ignore[assignment]
        nx = int(x)
        c = self.coeff
        if self.kind == "linear":
            return float(f32((x - f32(nx)) * c[nx + 1] + (f32(nx + 1) - x) * c[nx]))
        dx = f32(x - f32(nx))
        c0, c1, c2, c3 = _poly(c, nx + 1)
        return float(f32(c0 + dx * (c1 + dx * (c2 + dx * c3))))

    def integral(self, a: float, b: float) -> float:
        """asigrl from a to b."""
        xa, xb = (f32(a), f32(b)) if a <= b else (f32(b), f32(a))
        neara, nearb = int(xa), int(xb)
        c = self.coeff
        accum = f32(0)
        da = f32(xa - f32(neara))
        if self.kind == "linear":
            if nearb == neara:
                db = f32(xb - f32(nearb))
                accum = f32(accum + (db - da) * c[neara]
                            + f32(0.5) * (c[neara + 1] - c[neara]) * (db * db - da * da))
            else:
                accum = f32(accum + (f32(1) - da) * c[neara]
                            + f32(0.5) * (c[neara + 1] - c[neara]) * (f32(1) - da * da))
                for j in range(neara + 1, nearb):
                    accum = f32(accum + f32(0.5) * (c[j + 1] + c[j]))
                db = f32(xb - f32(nearb))
                accum = f32(accum + c[nearb] * db
                            + f32(0.5) * (c[nearb + 1] - c[nearb]) * db * db)
        else:
            if nearb == neara:
                db = f32(xb - f32(nearb))
                p = _poly(c, 1 + neara)
                for i in range(1, 5):
                    accum = f32(accum + f32(1.0 / i) * p[i - 1] * (db ** i - da ** i))
            else:
                p = _poly(c, 1 + neara)
                for i in range(1, 5):
                    accum = f32(accum + f32(1.0 / i) * p[i - 1] * (f32(1) - da ** i))
                for j in range(neara + 1, nearb):
                    p = _poly(c, 1 + j)
                    for i in range(1, 5):
                        accum = f32(accum + f32(1.0 / i) * p[i - 1])
                db = f32(xb - f32(nearb))
                p = _poly(c, 1 + nearb)
                for i in range(1, 5):
                    accum = f32(accum + f32(1.0 / i) * p[i - 1] * db ** i)
        return float(accum if a < b else -accum)


def _poly(c, index):
    """ii_getpcoeff for spline3: the cubic on [index-1, index) of COEFF."""
    a, b, cc, d = c[index - 1], c[index], c[index + 1], c[index + 2]
    return (f32(a + f32(4) * b + cc),
            f32(f32(3) * (cc - a)),
            f32(f32(3) * (a - f32(2) * b + cc)),
            f32(-a + f32(3) * b - f32(3) * cc + d))


def _spline(bcoeff, npts):
    """ii_spline: bcoeff[1] and bcoeff[npts+2] (1-based) are the end second
    derivatives, between them the data; returns the npts+2 B-spline
    coefficients."""
    b = np.asarray(bcoeff, dtype=f32).copy()
    diag = np.zeros(npts + 3, dtype=f32)
    # 1-based: b[k] is element k - 1.
    diag[1] = f32(-2)
    b[0] = b[0] / f32(6)
    diag[2] = f32(0)
    b[1] = (b[1] - b[0]) / f32(6)
    for i in range(3, npts + 2):
        diag[i] = f32(1) / (f32(4) - diag[i - 1])
        b[i - 1] = diag[i] * (b[i - 1] - b[i - 2])
    b[npts + 1] = ((diag[npts] + f32(2)) * b[npts] - b[npts - 1] + b[npts + 1] / f32(6)) \
        / (f32(1) + diag[npts + 1] * (diag[npts] + f32(2)))
    for i in range(npts + 1, 2, -1):
        b[i - 1] = b[i - 1] - diag[i] * b[i]
    b[0] = b[0] + f32(2) * b[1] - b[2]
    return b
