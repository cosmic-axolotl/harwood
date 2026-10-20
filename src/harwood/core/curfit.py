"""IRAF's curfit, the single-precision version: cvinit, cvacpts, cvsolve,
cveval and cvvector.

The tasks that fit a curve through IRAF's math/curfit -- sensfunc among
them -- solve the normal equations in real arithmetic: the basis functions,
the accumulated matrix, its Cholesky factorization and the coefficients are
all single precision. A fit in double precision is closer to the exact
least-squares curve and differs from IRAF's by up to 1e-4 in a sensitivity
of 39 magnitudes; this one is IRAF's, operation for operation (VOPS
included: x**3 is x*(x*x), sums run in order in a real).

From iraf-community 2.18: math/curfit/*.gx, sys/vops. See
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import numpy as np

from . import advice

f32 = np.float32
LEGENDRE, CHEBYSHEV, SPLINE3, SPLINE1 = 1, 2, 3, 4
FUNCTIONS = {"legendre": LEGENDRE, "chebyshev": CHEBYSHEV, "spline3": SPLINE3,
             "spline1": SPLINE1}
OK, SINGULAR, NO_DEG_FREEDOM = 0, 1, 2
EPSILONR = f32(1.192092896e-07)


class Curve:
    """A curve descriptor: cvinit, then cvacpts as often as wanted, then
    cvsolve; cveval and cvvector evaluate it."""

    def __init__(self, kind, order: int, xmin, xmax) -> None:
        kind = FUNCTIONS.get(kind, kind) if isinstance(kind, str) else kind
        xmin, xmax = f32(xmin), f32(xmax)
        if order < 1:
            advice.fail("CVINIT: Illegal order.", "the order has to be 1 or more.")
        if xmax <= xmin:
            advice.fail("CVINIT: xmax <= xmin.",
                        "the points need some range in x to fit a curve to.")
        self.kind, self.xmin, self.xmax = kind, xmin, xmax
        self.npieces = 0
        self.range = self.maxmin = self.spacing = f32(0)
        if kind in (CHEBYSHEV, LEGENDRE):
            self.order = self.ncoeff = order
            self.range = f32(f32(2) / f32(xmax - xmin))
            self.maxmin = f32(-f32(xmax + xmin) / f32(2))
        elif kind in (SPLINE3, SPLINE1):
            self.order = 4 if kind == SPLINE3 else 2
            self.ncoeff = order + self.order - 1
            self.npieces = order - 1
            self.spacing = f32(f32(order) / f32(xmax - xmin))
        else:
            advice.fail("CVINIT: Unknown curve type.",
                        "use chebyshev, legendre, spline3 or spline1.")
        self.matrix = np.zeros((self.ncoeff, self.order), f32)   # [row, band]
        self.vector = np.zeros(self.ncoeff, f32)
        self.coeff = np.zeros(self.ncoeff, f32)
        self.npts = 0

    # ------------------------------------------------------------ basis

    def _basis(self, x: np.ndarray):
        """cv_bcheb, cv_bleg, cv_bspline3, cv_bspline1: the non-zero basis
        functions at each x, (order, npts), and for splines the first."""
        n = x.size
        b = np.zeros((self.order, n), f32)
        left = None
        if self.kind in (CHEBYSHEV, LEGENDRE):
            for k in range(1, self.order + 1):
                if k == 1:
                    b[0] = f32(1)
                elif k == 2:
                    b[1] = (x + self.maxmin) * self.range
                elif self.kind == CHEBYSHEV:
                    b[k - 1] = b[1] * b[k - 2] * f32(2) - b[k - 3]
                else:
                    ri = f32(k)
                    ri1 = f32(f32(f32(2) * ri - f32(3)) / f32(ri - f32(1)))
                    ri2 = f32(-f32(ri - f32(2)) / f32(ri - f32(1)))
                    p = b[1] * b[k - 2]
                    b[k - 1] = p * ri1 + b[k - 3] * ri2
            return b, left
        sx = (x + f32(-self.xmin)) * self.spacing
        left = np.minimum(sx.astype(np.int64), self.npieces)
        sx = np.maximum(f32(0), np.minimum(f32(1), sx - left.astype(f32)))
        tx = np.maximum(f32(0), np.minimum(f32(1), f32(1) - sx))
        if self.kind == SPLINE1:
            b[1], b[0] = sx, tx
        else:
            b[0] = tx * tx * tx
            b[1] = f32(1) + tx * (f32(3) + tx * (f32(3) - f32(3) * tx))
            b[2] = f32(1) + sx * (f32(3) + sx * (f32(3) - f32(3) * sx))
            b[3] = sx * sx * sx
        return b, left

    # ------------------------------------------------------------ fitting

    def accumulate(self, x, y, w) -> None:
        """cvacpts with user weights (WTS_USER)."""
        x = np.asarray(x, f32)
        y = np.asarray(y, f32)
        w = np.asarray(w, f32)
        n = x.size
        self.npts += n
        if n == 0:
            return
        basis, left = self._basis(x)
        order = self.order
        if left is None:
            for k in range(order):
                bw = w * basis[k]
                total = self.vector[k]
                for i in range(n):
                    total = f32(total + f32(bw[i] * y[i]))
                self.vector[k] = total
                for ii, j in enumerate(range(k, order)):
                    total = self.matrix[k, ii]
                    for i in range(n):
                        total = f32(total + f32(bw[i] * basis[j, i]))
                    self.matrix[k, ii] = total
            return
        for k in range(order):
            bw = w * basis[k]
            for i in range(n):
                row = int(left[i]) + k
                self.vector[row] = f32(self.vector[row] + f32(bw[i] * y[i]))
            for ii, j in enumerate(range(k, order)):
                for i in range(n):
                    row = int(left[i]) + k
                    self.matrix[row, ii] = f32(self.matrix[row, ii]
                                               + f32(bw[i] * basis[j, i]))

    def solve(self) -> int:
        """cvsolve: Cholesky factorization and substitution, as cvchofac and
        cvchoslv do them; OK, SINGULAR or NO_DEG_FREEDOM."""
        if self.npts - self.ncoeff < 0:
            return NO_DEG_FREEDOM
        ier = OK
        nbands, nrows = self.order, self.ncoeff
        a = self.matrix
        m = a.copy()
        if nrows == 1:
            if a[0, 0] > 0:
                m[0, 0] = f32(f32(1) / a[0, 0])
        else:
            for n in range(nrows):
                if f32(f32(m[n, 0] + a[n, 0]) - a[n, 0]) <= f32(10) * EPSILONR:
                    m[n, :] = f32(0)
                    ier = SINGULAR
                    continue
                m[n, 0] = f32(f32(1) / m[n, 0])
                imax = min(nbands - 1, nrows - 1 - n)
                if imax < 1:
                    continue
                jmax = imax
                for i in range(1, imax + 1):
                    ratio = f32(m[n, i] * m[n, 0])
                    for j in range(jmax):
                        m[n + i, j] = f32(m[n + i, j] - f32(m[n, j + i] * ratio))
                    jmax -= 1
                    m[n, i] = ratio
        c = self.vector.copy()
        if nrows == 1:
            c[0] = f32(c[0] * m[0, 0])
        else:
            for n in range(nrows):
                for j in range(1, min(nbands - 1, nrows - 1 - n) + 1):
                    c[j + n] = f32(c[j + n] - f32(m[n, j] * c[n]))
            for n in range(nrows - 1, -1, -1):
                c[n] = f32(c[n] * m[n, 0])
                for j in range(1, min(nbands - 1, nrows - 1 - n) + 1):
                    c[n] = f32(c[n] - f32(m[n, j] * c[j + n]))
        self.coeff = c
        return ier

    # ------------------------------------------------------------ evaluation

    def __call__(self, x) -> f32:
        """cveval: one value, the basis dotted with the coefficients."""
        x = f32(x)
        if self.kind in (CHEBYSHEV, LEGENDRE):
            basis = [f32(1)] * self.order
            if self.order > 1:
                xnorm = f32(f32(x + self.maxmin) * self.range)
                basis[1] = xnorm
                for i in range(3, self.order + 1):
                    if self.kind == CHEBYSHEV:
                        basis[i - 1] = f32(f32(f32(2) * xnorm) * basis[i - 2]) - basis[i - 3]
                    else:
                        ri = f32(i)
                        basis[i - 1] = f32(f32(f32(f32(f32(2) * ri - f32(3)) * xnorm)
                                               * basis[i - 2])
                                           - f32(f32(ri - f32(2)) * basis[i - 3])) / \
                            f32(ri - f32(1))
                    basis[i - 1] = f32(basis[i - 1])
            left = 0
        else:
            b, lefts = self._basis(np.array([x], f32))
            basis, left = list(b[:, 0]), int(lefts[0])
        total = f32(0)
        for k in range(self.order):
            total = f32(total + f32(basis[k] * self.coeff[left + k]))
        return total

    def vector_at(self, x) -> np.ndarray:
        """cvvector: the curve at many x, as cv_evcheb, cv_evleg,
        cv_evspline3 and cv_evspline1 evaluate it."""
        x = np.asarray(x, f32)
        c = self.coeff
        if self.kind in (CHEBYSHEV, LEGENDRE):
            if self.order == 1:
                return np.full(x.size, c[0], f32)
            c1 = f32(self.range * c[1])
            c2 = f32(f32(c1 * self.maxmin) + c[0])
            yfit = x * c1 + c2
            if self.order == 2:
                return yfit
            sx = (x + self.maxmin) * self.range
            pnm2 = np.ones(x.size, f32)
            pnm1 = sx.copy()
            if self.kind == CHEBYSHEV:
                sx = sx * f32(2)
            for i in range(3, self.order + 1):
                if self.kind == CHEBYSHEV:
                    pn = sx * pnm1 - pnm2
                else:
                    ri = f32(i)
                    ri1 = f32(f32(f32(2) * ri - f32(3)) / f32(ri - f32(1)))
                    ri2 = f32(-f32(ri - f32(2)) / f32(ri - f32(1)))
                    pn = sx * pnm1
                    pn = pn * ri1 + pnm2 * ri2
                if i < self.order:
                    pnm2, pnm1 = pnm1, pn.copy()
                yfit = yfit + pn * c[i - 1]
            return yfit
        sx = (x + f32(-self.xmin)) * self.spacing
        index = np.minimum(sx.astype(np.int64), self.npieces)
        sx = np.maximum(f32(0), np.minimum(f32(1), sx - index.astype(f32)))
        tx = np.maximum(f32(0), np.minimum(f32(1), f32(1) - sx))
        if self.kind == SPLINE1:
            return tx * c[index] + sx * c[index + 1]
        yfit = np.zeros(x.size, f32)
        for i in range(4):
            if i == 0:
                temp = tx * (tx * tx)
            elif i == 1:
                temp = f32(1) + tx * (f32(3) + tx * (f32(3) - f32(3) * tx))
            elif i == 2:
                temp = f32(1) + sx * (f32(3) + sx * (f32(3) - f32(3) * sx))
            else:
                temp = sx * (sx * sx)
            yfit = yfit + temp * c[index + i]
        return yfit
