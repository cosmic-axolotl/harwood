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

from .nlfit import square

f32 = np.float32


#: ii_sinc's truncation and its least offset worth interpolating.
NSINC = 15
SINC_DX = f32(0.001)


def _sinc_taper(nsinc: int) -> np.ndarray:
    """ii_sinc's cosine-bell taper with the sinc's alternating sign."""
    halfpi = 1.5707963267948966
    sconst = square(f32(halfpi) / f32(nsinc))
    a2, a4 = f32(-0.49670), f32(0.03705)
    w1 = f32(1.0) if nsinc % 2 == 0 else f32(-1.0)
    taper = np.zeros(2 * nsinc + 1, dtype=f32)
    for j in range(-nsinc, nsinc + 1):
        dx2 = f32(f32(sconst * f32(j)) * f32(j))
        t = f32(f32(f32(1.0) + f32(a2 * dx2)) + f32(f32(a4 * dx2) * dx2))
        taper[j + nsinc] = f32(w1 * f32(t * t))
        w1 = f32(-w1)
    return taper


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
            self.offset = 0
        elif kind == "nearest":
            c = np.zeros(n + 1, dtype=f32)
            c[1:n + 1] = data
            self.coeff = c
            self.offset = 0
        elif kind in ("poly3", "poly5"):
            need = 4 if kind == "poly3" else 6
            if n < need:
                raise ValueError(f"ASIFIT: too few points for {kind.upper()}")
            off = 1 if kind == "poly3" else 2
            # COEFF[k] at index k; data[i] at COEFF[off + i].
            c = np.zeros(n + 2 * off + 2, dtype=f32)
            c[off + 1:off + n + 1] = data
            d = lambda i: c[off + i]  # noqa: E731  (1-based data)
            two = f32(2)
            if kind == "poly3":
                c[1] = two * d(1) - d(2)
                c[off + n + 1] = two * d(n) - d(n - 1)
                c[off + n + 2] = two * d(n) - d(n - 2)
            else:
                c[1] = two * d(1) - d(3)
                c[2] = two * d(1) - d(2)
                c[off + n + 1] = two * d(n) - d(n - 1)
                c[off + n + 2] = two * d(n) - d(n - 2)
                c[off + n + 3] = two * d(n) - d(n - 3)
            self.coeff = c
            self.offset = off
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
            self.offset = 1
        elif kind == "sinc":
            if n < 1:
                raise ValueError("ASIFIT: too few points for SINC")
            # COEFF[1..n] the data, stored with a leading pad as above.
            c = np.zeros(n + 1, dtype=f32)
            c[1:] = data
            self.coeff = c
            self.offset = 0
            self.nsinc = NSINC
            self._taper = _sinc_taper(NSINC)
        else:
            raise ValueError(f"interpolant '{kind}' is not ported")

    def _sinc(self, x) -> f32:
        """ii_sinc at one point: the tapered sinc convolution, normalised by
        the sum of its weights, the end pixels repeated past the ends."""
        x = f32(x)
        npix, nsinc = self.n, self.nsinc
        data = self.coeff
        xc = int(np.floor(x + f32(0.5)))
        if xc < 1 or xc > npix:
            return f32(0)
        dx = f32(x - f32(xc))
        if abs(dx) < SINC_DX:
            return data[xc]
        minj, maxj = max(1, xc - nsinc), min(npix, xc + nsinc)
        offj = -xc + nsinc
        total = weight = f32(0)
        dxn = f32(dx + f32(xc))
        for j in range(xc - nsinc, xc + nsinc + 1):
            w1 = f32(self._taper[j + offj] / f32(dxn - f32(j)))
            value = data[1] if j < minj else (data[npix] if j > maxj else data[j])
            total = f32(total + f32(w1 * value))
            weight = f32(weight + w1)
        return f32(total / weight)

    def __call__(self, x: float) -> float:
        """asieval at one point."""
        if self.kind == "sinc":
            return float(self._sinc(x))
        x = f32(x)  # type: ignore[assignment]
        nx = int(x)
        c = self.coeff
        if self.kind == "nearest":
            return float(c[int(x + f32(0.5))])
        if self.kind in ("poly3", "poly5"):
            return float(_poly_eval(c, self.offset, x, self.kind == "poly5"))
        if self.kind == "linear":
            return float(f32((x - f32(nx)) * c[nx + 1] + (f32(nx + 1) - x) * c[nx]))
        dx = f32(x - f32(nx))
        c0, c1, c2, c3 = _poly(c, nx + 1)
        return float(f32(c0 + dx * (c1 + dx * (c2 + dx * c3))))

    def derivative(self, x: float) -> float:
        """asider's first derivative at one point (linear and spline3)."""
        x = f32(x)  # type: ignore[assignment]
        nx = int(x)
        c = self.coeff
        if self.kind == "linear":
            return float(f32(c[nx + 1] - c[nx]))
        if self.kind != "spline3":
            raise ValueError(f"derivative of {self.kind} is not ported")
        dx = f32(x - f32(nx))
        _, c1, c2, c3 = _poly(c, nx + 1)
        return float(f32(c1 + dx * (f32(2) * c2 + f32(3) * c3 * dx)))

    def integral(self, a: float, b: float) -> float:
        """asigrl from a to b."""
        xa, xb = (f32(a), f32(b)) if a <= b else (f32(b), f32(a))
        if self.kind == "sinc":
            # ii_sincigrl: the value at the middle of each of n equal steps
            # of at most a pixel, times the step.
            n = max(1, int(np.floor(f32(xb - xa) + f32(0.5))))
            dx = f32(f32(xb - xa) / f32(n))
            total = f32(0)
            x = f32(xa + f32(dx / f32(2)))
            while x < xb:
                total = f32(total + f32(self._sinc(x) * dx))
                x = f32(x + dx)
            return float(total if a <= b else -total)
        neara, nearb = int(xa), int(xb)
        c = self.coeff
        accum = f32(0)
        da = f32(xa - f32(neara))
        if self.kind == "nearest":
            na, nb = int(xa + f32(0.5)), int(xb + f32(0.5))
            da = f32(xa - f32(na))
            if na == nb:
                accum = f32((f32(xb - f32(nb)) - da) * c[na])
            else:
                accum = f32((f32(0.5) - da) * c[na])
                for j in range(na + 1, nb):
                    accum = f32(accum + c[j])
                accum = f32(accum + (f32(xb - f32(nb)) + f32(0.5)) * c[nb])
            return float(accum if a < b else -accum)
        if self.kind in ("poly3", "poly5"):
            nterms = 4 if self.kind == "poly3" else 6

            def pc(near):
                return _newton(c, self.offset + near, nterms)

            if nearb == neara:
                db = f32(xb - f32(nearb))
                p = pc(neara)
                for i in range(1, nterms + 1):
                    accum = f32(accum + f32(1.0 / i) * p[i - 1] * f32(db ** i - da ** i))
            else:
                p = pc(neara)
                for i in range(1, nterms + 1):
                    accum = f32(accum + f32(1.0 / i) * p[i - 1] * f32(f32(1) - da ** i))
                for j in range(neara + 1, nearb):
                    p = pc(j)
                    for i in range(1, nterms + 1):
                        accum = f32(accum + f32(1.0 / i) * p[i - 1])
                db = f32(xb - f32(nearb))
                p = pc(nearb)
                for i in range(1, nterms + 1):
                    accum = f32(accum + f32(1.0 / i) * p[i - 1] * f32(db ** i))
            return float(accum if a < b else -accum)
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


def _poly_eval(c, off, x, five):
    """ii_poly3 / ii_poly5: Everett's formula on the data around x."""
    nx = int(x)
    dx = f32(x - f32(nx))
    dy = f32(f32(1) - dx)

    def d(i):
        return c[off + i]

    cd20 = f32(f32(1.0 / 6.0) * (d(nx + 1) - f32(2) * d(nx) + d(nx - 1)))
    cd21 = f32(f32(1.0 / 6.0) * (d(nx + 2) - f32(2) * d(nx + 1) + d(nx)))
    if not five:
        return f32(dx * (d(nx + 1) + (dx * dx - f32(1)) * cd21)
                   + dy * (d(nx) + (dy * dy - f32(1)) * cd20))
    cd40 = f32(f32(1.0 / 120.0) * (d(nx - 2) - f32(4) * d(nx - 1) + f32(6) * d(nx)
                                   - f32(4) * d(nx + 1) + d(nx + 2)))
    cd41 = f32(f32(1.0 / 120.0) * (d(nx - 1) - f32(4) * d(nx) + f32(6) * d(nx + 1)
                                   - f32(4) * d(nx + 2) + d(nx + 3)))
    return f32(dx * (d(nx + 1) + (dx * dx - f32(1)) * (cd21 + (dx * dx - f32(4)) * cd41))
               + dy * (d(nx) + (dy * dy - f32(1)) * (cd20 + (dy * dy - f32(4)) * cd40)))


def _newton(c, index, nterms):
    """ii_getpcoeff for poly3/poly5: the polynomial through the nterms points
    around COEFF[index], in powers of (x - index), lowest first."""
    diff = [f32(c[index - nterms // 2 + i]) for i in range(1, nterms + 1)]
    for k in range(1, nterms):
        for i in range(nterms - k):
            diff[i] = f32((diff[i + 1] - diff[i]) / f32(k))
    for k in range(nterms, 1, -1):
        for i in range(2, k + 1):
            diff[i - 1] = f32(diff[i - 1] + diff[i - 2] * f32(k - i - nterms // 2))
    return [diff[nterms - i] for i in range(1, nterms + 1)]


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
