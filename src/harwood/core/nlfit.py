"""IRAF's nlfit: non-linear least squares, Levenberg-Marquardt as IRAF does it.

From iraf-community 2.18 math/nlfit (nlinit, nlfit, nliter, nlacpts,
nlresid, nlsolve, nlchomat, nlerrors), the single precision version, in
single precision: apphot's gauss centring and gauss sky fit call nlfitr, and
which iteration it stops at depends on the rounding.

Its quirks are kept, because they decide where a fit lands: a damping step
that fails adds its increment to the previous failed trial rather than to
the parameters; the singular test is against EPSILONR absolutely; and the
errors come from the curvature matrix of the last accepted iteration's
starting point, undamped.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

f32 = np.float32
EPSILONR = f32(1.192e-7)
LAMBDAMAX = f32(1000.0)
MINITER = 3

DONE, SINGULAR, NO_DEG_FREEDOM, NOT_DONE = 0, 1, 2, 3
OK = 0


def square(x) -> np.float32:
    """x ** 2 as IRAF's compiled code does it, x * x. numpy's ** on a float32
    scalar goes through powf, and about one time in a thousand is off in the
    last place."""
    x = f32(x)
    return f32(x * x)


def chfac(matrix: np.ndarray, nbands: int, nrows: int, matfac: np.ndarray,
          singular=EPSILONR) -> int:
    """nl_chfac: banded Cholesky factorisation (de Boor's bchfac), in place
    into matfac; matrix[j, n] is element (n + j, n), 0-based. A pivot at or
    below `singular` makes the row singular (gsurfit's gs_chofac uses
    1000 / MAX_REAL where nlfit uses EPSILONR)."""
    ier = OK
    if nrows == 1:
        if matrix[0, 0] > 0:
            matfac[0, 0] = f32(1.0) / matrix[0, 0]
        return ier
    matfac[:nbands, :nrows] = matrix[:nbands, :nrows]
    for n in range(nrows):
        if (f32(matfac[0, n] + matrix[0, n]) - matrix[0, n]) <= singular:
            matfac[:nbands, n] = 0
            ier = SINGULAR
            continue
        matfac[0, n] = f32(1.0) / matfac[0, n]
        imax = min(nbands - 1, nrows - 1 - n)
        if imax < 1:
            continue
        jmax = imax
        for i in range(1, imax + 1):
            ratio = f32(matfac[i, n] * matfac[0, n])
            for j in range(jmax):
                matfac[j, n + i] = f32(matfac[j, n + i] - f32(matfac[j + i, n] * ratio))
            jmax -= 1
            matfac[i, n] = ratio
    return ier


def chslv(matfac: np.ndarray, nbands: int, nrows: int, vector: np.ndarray) -> np.ndarray:
    """nl_chslv: solve with the factorisation from chfac."""
    if nrows == 1:
        return np.array([f32(vector[0] * matfac[0, 0])], dtype=f32)
    coeff = np.array(vector, dtype=f32)
    for n in range(nrows):
        jmax = min(nbands - 1, nrows - 1 - n)
        for j in range(1, jmax + 1):
            coeff[j + n] = f32(coeff[j + n] - f32(matfac[j, n] * coeff[n]))
    for n in range(nrows - 1, -1, -1):
        coeff[n] = f32(coeff[n] * matfac[0, n])
        jmax = min(nbands - 1, nrows - 1 - n)
        for j in range(1, jmax + 1):
            coeff[n] = f32(coeff[n] - f32(matfac[j, n] * coeff[j + n]))
    return coeff


class NLFit:
    """One nlinitr .. nlfreer: `func(x, p) -> z` and `dfunc(x, p) -> (z, der)`
    with p the full parameter vector (0-based) and `plist` the 0-based
    indices of the parameters fitted, in IRAF's order."""

    def __init__(self, func: Callable, dfunc: Callable, params: Sequence,
                 plist: Sequence[int], tol: float, itmax: int):
        self.func, self.dfunc = func, dfunc
        self.param = np.array(params, dtype=f32)
        self.nparams = self.param.size
        self.plist = list(plist)
        self.nfit = len(self.plist)
        self.tol = f32(tol)
        self.itmax = itmax
        nfit = self.nfit
        self.alpha = np.zeros((nfit, nfit), dtype=f32)
        self.chofac = np.zeros((nfit, nfit), dtype=f32)
        self.beta = np.zeros(nfit, dtype=f32)
        self.dparam = np.zeros(nfit, dtype=f32)
        self.trial = np.zeros(self.nparams, dtype=f32)
        self.lam = f32(0.001)
        self.oldsq = self.sumsq = self.refsq = f32(0.0)
        self.iter = 0
        self.npts = 0

    # nlacpts / nlresid
    def _accumulate(self, x, z, w) -> np.float32:
        nfit = self.nfit
        self.alpha[:] = 0
        self.beta[:] = 0
        self.npts = len(z)
        total = f32(0.0)
        for i in range(len(z)):
            z0, deriv = self.dfunc(x[i], self.param)
            dz = f32(z[i] - z0)
            for a in range(nfit):
                wt = f32(deriv[self.plist[a]] * w[i])
                for k, b in enumerate(range(a, nfit)):
                    self.alpha[k, a] = f32(self.alpha[k, a] + f32(wt * deriv[self.plist[b]]))
                self.beta[a] = f32(self.beta[a] + f32(dz * wt))
            total = f32(total + f32(f32(w[i] * dz) * dz))
        nfree = self.npts - nfit
        return f32(0.0) if nfree <= 0 else f32(total / f32(nfree))

    def _residuals(self, x, z, w) -> np.float32:
        total = f32(0.0)
        for i in range(len(z)):
            dz = f32(z[i] - self.func(x[i], self.trial))
            total = f32(total + f32(f32(dz * dz) * w[i]))
        nfree = len(z) - self.nfit
        return f32(0.0) if nfree <= 0 else f32(total / f32(nfree))

    def _solve(self) -> int:
        covar = self.alpha.copy()
        self.dparam = self.beta.copy()
        covar[0, :] = covar[0, :] * f32(f32(1.0) + self.lam)
        if self.npts - self.nfit < 0:
            return NO_DEG_FREEDOM
        ier = chfac(covar, self.nfit, self.nfit, self.chofac)
        self.dparam = chslv(self.chofac, self.nfit, self.nfit, self.dparam)
        return ier

    def _iterate(self, x, z, w) -> int:
        self.oldsq = self._accumulate(x, z, w)
        self.trial = self.param.copy()
        while True:
            ier = self._solve()
            if ier == NO_DEG_FREEDOM:
                return ier
            for a in range(self.nfit):
                p = self.plist[a]
                self.trial[p] = f32(self.trial[p] + self.dparam[a])
            self.sumsq = self._residuals(x, z, w)
            if self.oldsq >= self.sumsq:
                self.param = self.trial.copy()
                self.lam = f32(f32(0.10) * self.lam)
                break
            self.lam = min(LAMBDAMAX, f32(f32(10.0) * self.lam))
            if self.lam <= 0 or self.lam >= LAMBDAMAX:
                break
        return ier

    def fit(self, x, z, w=None, scatter: bool = False) -> int:
        """nlfitr: DONE, SINGULAR, NO_DEG_FREEDOM or NOT_DONE. Without `w` the
        weights are uniform (WTS_UNIFORM); with it they are the user's
        (WTS_USER), or with `scatter` the user's with a scatter term fitted
        and added to their variances as the fit goes (WTS_SCATTER). The
        weights used, so altered, are left in self.w and the scatter in
        self.scatter."""
        z = np.asarray(z, dtype=f32)
        self.w = (np.ones(len(z), dtype=f32) if w is None
                  else np.asarray(w, dtype=f32).copy())
        self.iter = 0
        self.lam = f32(0.001)
        self.refsq = f32(0.0)
        self.scatter = f32(0.0)
        miniter = MINITER + 1 if scatter else MINITER
        total = f32(0.0)
        stat = NOT_DONE
        while True:
            ier = self._iterate(x, z, self.w)
            self.iter += 1
            stat = ier
            if stat == NO_DEG_FREEDOM:
                break
            if self.iter < miniter:
                stat = NOT_DONE
            elif self.sumsq <= f32(10.0) * EPSILONR:
                stat = DONE
            elif self.refsq < self.sumsq:
                stat = NOT_DONE
            elif f32(f32(self.refsq - self.sumsq) / self.sumsq) < self.tol:
                stat = DONE
            else:
                stat = NOT_DONE
            if stat == DONE:
                if ier == SINGULAR:
                    stat = ier
                break
            if self.lam <= 0:
                break
            if self.iter >= miniter and self.iter >= self.itmax:
                break
            if scatter:
                dscatter = self._scatter(x, z)
                if self.iter >= MINITER and dscatter >= f32(10.0) * EPSILONR:
                    good = self.w > 0
                    self.w[~good] = 0
                    self.w[good] = f32(1.0) / (f32(1.0) / self.w[good] + dscatter)
                    total = f32(total + dscatter)
            self.refsq = min(self.oldsq, self.sumsq)
        self.scatter = f32(self.scatter + total)
        return stat

    def _scatter(self, x, z) -> np.float32:
        """nlscatter: half the variance in excess of the weights' account."""
        variance, chisqr, _ = self.statistics(z, self.vector(x), self.w)
        if chisqr <= 0 or variance <= 0:
            return f32(0.0)
        return f32(f32(f32(f32(0.5) * variance) * f32(chisqr - f32(1.0))) / chisqr)

    def vector(self, x) -> np.ndarray:
        return np.array([self.func(v, self.param) for v in x], dtype=f32)

    def errors(self, z, zfit) -> np.ndarray:
        """nlerrorsr: the parameters' errors (0 for those not fitted)."""
        return self.statistics(z, zfit, self.w)[2]

    def statistics(self, z, zfit, w) -> tuple[np.float32, np.float32, np.ndarray]:
        """nlerrorsr with the weights given: the variance and reduced
        chi-squared of the fit, and the parameters' errors."""
        z = np.asarray(z, dtype=f32)
        n = 0
        variance = chisqr = f32(0.0)
        for i in range(len(z)):
            if w[i] <= 0:
                continue
            factor = square(z[i] - zfit[i])
            variance = f32(variance + factor)
            chisqr = f32(chisqr + f32(factor * w[i]))
            n += 1
        nfree = n - self.nfit
        if nfree > 0:
            variance = f32(variance / f32(nfree))
            chisqr = f32(chisqr / f32(nfree))
        else:
            variance = chisqr = f32(0.0)
        if abs(f32(chisqr - variance)) <= EPSILONR:
            factor = chisqr if nfree > 0 else f32(0.0)
        else:
            factor = f32(1.0)
        errors = np.zeros(self.nparams, dtype=f32)
        chfac(self.alpha, self.nfit, self.nfit, self.chofac)
        for i in range(self.nfit):
            cov = np.zeros(self.nfit, dtype=f32)
            cov[i] = 1
            cov = chslv(self.chofac, self.nfit, self.nfit, cov)
            errors[self.plist[i]] = (np.sqrt(f32(cov[i] * factor)) if cov[i] >= 0
                                     else f32(0.0))
        return variance, chisqr, errors
