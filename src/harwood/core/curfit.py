"""IRAF's curfit: cvinit, cvacpts, cvsolve, cvrject, cverrors, cvpower, cveval
and cvvector, in Curve the single-precision version and in DoubleCurve the other.

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

from . import advice, iraffmt

f32 = np.float32
LEGENDRE, CHEBYSHEV, SPLINE3, SPLINE1 = 1, 2, 3, 4
FUNCTIONS = {"legendre": LEGENDRE, "chebyshev": CHEBYSHEV, "spline3": SPLINE3,
             "spline1": SPLINE1}
OK, SINGULAR, NO_DEG_FREEDOM = 0, 1, 2
EPSILONR = f32(1.192e-7)   # hlib$mach.h
EPSILOND = 2.220e-16


class Curve:
    """A curve descriptor: cvinit, then cvacpts as often as wanted, then
    cvsolve; cveval and cvvector evaluate it."""

    real = f32
    epsilon = EPSILONR

    def __init__(self, kind, order: int, xmin, xmax) -> None:
        real = self.real
        kind = FUNCTIONS.get(kind, kind) if isinstance(kind, str) else kind
        xmin, xmax = real(xmin), real(xmax)
        if order < 1:
            advice.fail("CVINIT: Illegal order.", "the order has to be 1 or more.")
        if xmax <= xmin:
            advice.fail("CVINIT: xmax <= xmin.",
                        "the points need some range in x to fit a curve to.")
        self.kind, self.xmin, self.xmax = kind, xmin, xmax
        self.npieces = 0
        self.range = self.maxmin = self.spacing = real(0)
        if kind in (CHEBYSHEV, LEGENDRE):
            self.order = self.ncoeff = order
            self.range = real(real(2) / real(xmax - xmin))
            self.maxmin = real(-real(xmax + xmin) / real(2))
        elif kind in (SPLINE3, SPLINE1):
            self.order = 4 if kind == SPLINE3 else 2
            self.ncoeff = order + self.order - 1
            self.npieces = order - 1
            self.spacing = real(real(order) / real(xmax - xmin))
        else:
            advice.fail("CVINIT: Unknown curve type.",
                        "use chebyshev, legendre, spline3 or spline1.")
        self.matrix = np.zeros((self.ncoeff, self.order), real)   # [row, band]
        self.vector = np.zeros(self.ncoeff, real)
        self.coeff = np.zeros(self.ncoeff, real)
        self.npts = 0

    # ------------------------------------------------------------ basis

    def _basis(self, x: np.ndarray):
        """cv_bcheb, cv_bleg, cv_bspline3, cv_bspline1: the non-zero basis
        functions at each x, (order, npts), and for splines the first."""
        real = self.real
        n = x.size
        b = np.zeros((self.order, n), real)
        left = None
        if self.kind in (CHEBYSHEV, LEGENDRE):
            for k in range(1, self.order + 1):
                if k == 1:
                    b[0] = real(1)
                elif k == 2:
                    b[1] = (x + self.maxmin) * self.range
                elif self.kind == CHEBYSHEV:
                    b[k - 1] = b[1] * b[k - 2] * real(2) - b[k - 3]
                else:
                    ri = real(k)
                    ri1 = real(real(real(2) * ri - real(3)) / real(ri - real(1)))
                    ri2 = real(-real(ri - real(2)) / real(ri - real(1)))
                    p = b[1] * b[k - 2]
                    b[k - 1] = p * ri1 + b[k - 3] * ri2
            return b, left
        sx = (x + real(-self.xmin)) * self.spacing
        left = np.minimum(sx.astype(np.int64), self.npieces)
        sx = np.maximum(real(0), np.minimum(real(1), sx - left.astype(real)))
        tx = np.maximum(real(0), np.minimum(real(1), real(1) - sx))
        if self.kind == SPLINE1:
            b[1], b[0] = sx, tx
        else:
            b[0] = tx * tx * tx
            b[1] = real(1) + tx * (real(3) + tx * (real(3) - real(3) * tx))
            b[2] = real(1) + sx * (real(3) + sx * (real(3) - real(3) * sx))
            b[3] = sx * sx * sx
        return b, left

    # ------------------------------------------------------------ fitting

    def accumulate(self, x, y, w) -> None:
        """cvacpts with user weights (WTS_USER)."""
        real = self.real
        x = np.asarray(x, real)
        y = np.asarray(y, real)
        w = np.asarray(w, real)
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
                    total = real(total + real(bw[i] * y[i]))
                self.vector[k] = total
                for ii, j in enumerate(range(k, order)):
                    total = self.matrix[k, ii]
                    for i in range(n):
                        total = real(total + real(bw[i] * basis[j, i]))
                    self.matrix[k, ii] = total
            return
        for k in range(order):
            bw = w * basis[k]
            for i in range(n):
                row = int(left[i]) + k
                self.vector[row] = real(self.vector[row] + real(bw[i] * y[i]))
            for ii, j in enumerate(range(k, order)):
                for i in range(n):
                    row = int(left[i]) + k
                    self.matrix[row, ii] = real(self.matrix[row, ii]
                                               + real(bw[i] * basis[j, i]))

    def reject(self, x, y, w) -> None:
        """cvrject: take points out of the sums, as accumulate put them in."""
        real = self.real
        for point, value, weight in zip(np.atleast_1d(x), np.atleast_1d(y),
                                        np.atleast_1d(w), strict=True):
            self.npts -= 1
            basis, left = self._b1(real(point))
            for i in range(self.order):
                bw = real(basis[i] * real(weight))
                self.vector[left + i] = real(self.vector[left + i] - real(bw * real(value)))
                for ii, j in enumerate(range(i, self.order)):
                    self.matrix[left + i, ii] = real(self.matrix[left + i, ii]
                                                     - real(basis[j] * bw))

    def solve(self) -> int:
        """cvsolve: Cholesky factorization and substitution, as cvchofac and
        cvchoslv do them; OK, SINGULAR or NO_DEG_FREEDOM."""
        real = self.real
        if self.npts - self.ncoeff < 0:
            return NO_DEG_FREEDOM
        ier = OK
        nbands, nrows = self.order, self.ncoeff
        a = self.matrix
        m = a.copy()
        if nrows == 1:
            if a[0, 0] > 0:
                m[0, 0] = real(real(1) / a[0, 0])
        else:
            for n in range(nrows):
                if real(real(m[n, 0] + a[n, 0]) - a[n, 0]) <= real(10) * self.epsilon:
                    m[n, :] = real(0)
                    ier = SINGULAR
                    continue
                m[n, 0] = real(real(1) / m[n, 0])
                imax = min(nbands - 1, nrows - 1 - n)
                if imax < 1:
                    continue
                jmax = imax
                for i in range(1, imax + 1):
                    ratio = real(m[n, i] * m[n, 0])
                    for j in range(jmax):
                        m[n + i, j] = real(m[n + i, j] - real(m[n, j + i] * ratio))
                    jmax -= 1
                    m[n, i] = ratio
        self.chofac = m
        self.coeff = self._choslv(self.vector)
        return ier

    def _choslv(self, vector) -> np.ndarray:
        """cvchoslv: what the factor of cvsolve gives for a right hand side."""
        real = self.real
        nbands, nrows, m = self.order, self.ncoeff, self.chofac
        c = vector.copy()
        if nrows == 1:
            c[0] = real(c[0] * m[0, 0])
            return c
        for n in range(nrows):
            for j in range(1, min(nbands - 1, nrows - 1 - n) + 1):
                c[j + n] = real(c[j + n] - real(m[n, j] * c[n]))
        for n in range(nrows - 1, -1, -1):
            c[n] = real(c[n] * m[n, 0])
            for j in range(1, min(nbands - 1, nrows - 1 - n) + 1):
                c[n] = real(c[n] - real(m[n, j] * c[j + n]))
        return c

    def errors(self, y, w, yfit) -> tuple:
        """cverrors: the reduced chi square, and the errors of the coefficients."""
        real = self.real
        variance, chisq, n = real(0), real(0), 0
        for i in range(len(y)):
            if w[i] <= 0:
                continue
            hold = real(real(y[i] - yfit[i]) * real(y[i] - yfit[i]))
            variance = real(variance + hold)
            chisq = real(chisq + real(hold * w[i]))
            n += 1
        nfree = n - self.ncoeff
        chisqr = real(chisq / real(nfree)) if nfree > 0 else real(0)
        # DELTA is EPSILON, the real one, in both versions
        if abs(real(chisq - variance)) <= real(EPSILONR):
            variance = real(chisq / real(nfree)) if nfree > 0 else real(0)
        else:
            variance = real(1)
        errors = np.zeros(self.ncoeff, real)
        for i in range(self.ncoeff):
            unit = np.zeros(self.ncoeff, real)
            unit[i] = real(1)
            cov = self._choslv(unit)[i]
            errors[i] = np.sqrt(real(cov * variance)) if cov >= 0 else real(0)
        return chisqr, errors

    def power(self) -> np.ndarray | None:
        """cvpower: the coefficients of the same polynomial in powers of x, None
        for a spline. The work is in double in both versions."""
        if self.kind not in (LEGENDRE, CHEBYSHEV):
            return None
        real, ncoeff = self.real, self.ncoeff
        matrix = np.zeros((ncoeff, ncoeff))    # [s, power]
        for s in range(ncoeff):
            for m in range(s // 2 + 1):
                if self.kind == LEGENDRE:
                    sum1 = (-1) ** m * _factorial(2 * s - 2 * m)
                    divisor = 2**s * _factorial(m) * _factorial(s - m) * _factorial(s - 2 * m)
                    matrix[s, s - 2 * m] = sum1 / divisor
                elif s == 0:
                    matrix[0, 0] = 1.0
                else:
                    sum1 = (-1) ** m * _factorial(s - m - 1) * 2 ** (s - 2 * m)
                    divisor = _factorial(s - 2 * m) * _factorial(m)
                    matrix[s, s - 2 * m] = (s / 2.0) * (sum1 / divisor)
        coeff = np.zeros(ncoeff, real)
        for n in range(ncoeff):
            total = 0.0
            for i in range(ncoeff):
                total += matrix[i, n] * float(self.coeff[i])
            coeff[n] = total
        # cv_normalize
        k1 = float(self.range)
        k2 = k1 * float(self.maxmin)
        elm = np.zeros((ncoeff, ncoeff))
        for k in range(ncoeff):
            for i in range(k + 1):
                bc = 1.0 if i == 0 or i == k else (
                    _factorial(k) / (_factorial(k - i) * _factorial(i)))
                elm[i, k] = (bc * float(coeff[k]) * iraffmt.pow_di(k1, i)
                             * iraffmt.pow_di(k2, k - i))
        for n in range(ncoeff):
            total = 0.0
            for i in range(ncoeff):
                total += elm[n, i]
            coeff[n] = total
        return coeff

    # ------------------------------------------------------------ evaluation

    def _b1(self, x):
        """cv_b1cheb, cv_b1leg, cv_b1spline3, cv_b1spline1: the basis at one x."""
        real = self.real
        if self.kind in (CHEBYSHEV, LEGENDRE):
            basis = [real(1)] * self.order
            if self.order > 1:
                xnorm = real(real(x + self.maxmin) * self.range)
                basis[1] = xnorm
                for i in range(3, self.order + 1):
                    if self.kind == CHEBYSHEV:
                        basis[i - 1] = (real(real(real(2) * xnorm) * basis[i - 2])
                                        - basis[i - 3])
                    else:
                        ri = real(i)
                        basis[i - 1] = real(real(real(real(real(2) * ri - real(3)) * xnorm)
                                               * basis[i - 2])
                                           - real(real(ri - real(2)) * basis[i - 3])) / \
                            real(ri - real(1))
                    basis[i - 1] = real(basis[i - 1])
            left = 0
        else:
            b, lefts = self._basis(np.array([x], real))
            basis, left = list(b[:, 0]), int(lefts[0])
        return basis, left

    def __call__(self, x):
        """cveval: one value, the basis dotted with the coefficients."""
        real = self.real
        basis, left = self._b1(real(x))
        total = real(0)
        for k in range(self.order):
            total = real(total + real(basis[k] * self.coeff[left + k]))
        return total

    def vector_at(self, x) -> np.ndarray:
        """cvvector: the curve at many x, as cv_evcheb, cv_evleg,
        cv_evspline3 and cv_evspline1 evaluate it."""
        real = self.real
        x = np.asarray(x, real)
        c = self.coeff
        if self.kind in (CHEBYSHEV, LEGENDRE):
            if self.order == 1:
                return np.full(x.size, c[0], real)
            c1 = real(self.range * c[1])
            c2 = real(real(c1 * self.maxmin) + c[0])
            yfit = x * c1 + c2
            if self.order == 2:
                return yfit
            sx = (x + self.maxmin) * self.range
            pnm2 = np.ones(x.size, real)
            pnm1 = sx.copy()
            if self.kind == CHEBYSHEV:
                sx = sx * real(2)
            for i in range(3, self.order + 1):
                if self.kind == CHEBYSHEV:
                    pn = sx * pnm1 - pnm2
                else:
                    ri = real(i)
                    ri1 = real(real(real(2) * ri - real(3)) / real(ri - real(1)))
                    ri2 = real(-real(ri - real(2)) / real(ri - real(1)))
                    pn = sx * pnm1
                    pn = pn * ri1 + pnm2 * ri2
                if i < self.order:
                    pnm2, pnm1 = pnm1, pn.copy()
                yfit = yfit + pn * c[i - 1]
            return yfit
        sx = (x + real(-self.xmin)) * self.spacing
        index = np.minimum(sx.astype(np.int64), self.npieces)
        sx = np.maximum(real(0), np.minimum(real(1), sx - index.astype(real)))
        tx = np.maximum(real(0), np.minimum(real(1), real(1) - sx))
        if self.kind == SPLINE1:
            return tx * c[index] + sx * c[index + 1]
        yfit = np.zeros(x.size, real)
        for i in range(4):
            if i == 0:
                temp = tx * (tx * tx)
            elif i == 1:
                temp = real(1) + tx * (real(3) + tx * (real(3) - real(3) * tx))
            elif i == 2:
                temp = real(1) + sx * (real(3) + sx * (real(3) - real(3) * sx))
            else:
                temp = sx * (sx * sx)
            yfit = yfit + temp * c[index + i]
        return yfit


class DoubleCurve(Curve):
    """dcvinit and the rest: the same operations, in double."""

    real = np.float64
    epsilon = EPSILOND


def _factorial(n: int) -> float:
    """cv_factorial, in double."""
    fact = 1.0
    for i in range(n, 0, -1):
        fact *= float(i)
    return fact
