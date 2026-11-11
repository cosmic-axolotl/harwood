"""images.imfit's fit1d without the task: pkg/images/imfit/src/fit1d.x, as IRAF 2.18
runs it, and what imred.generic's background makes of it (a fit1d of type
difference).

Every line (axis 1) or column (axis 2) of the image is fitted with ICFIT, and the
output is the fit, the data minus the fit, or the data over it. The fits are
ic_fit's in real (core/icfitbits.py), called as fit1d calls it: the sample and the
curve are set up on the first line only, and after that ic_fit is told that only
the data are new. What follows from that, kept:

* a line after one that rejected no point is not fitted afresh: its data are put
  through the factorization of the last fresh fit (cvrefit) -- with a mask, the
  weights of that earlier line, not its own; a line after one that rejected
  points is fitted afresh;
* "Singular solution" is printed again for each line that only re-solves, since
  the code it comes from is kept from the fit before;
* the difference and the ratio take the fit point by point (cveval), the fit
  itself as a vector (cvvector); a ratio's divisor below 1e-20 is 1.
"""

from __future__ import annotations

import numpy as np

from . import icfitbits
from .curfit import CHEBYSHEV, LEGENDRE, NO_DEG_FREEDOM, SINGULAR

f32 = np.float32
TYPES = ("fit", "difference", "ratio")


class Fitter:
    """ic_fit's state between the calls of one fit1d: the structure, the curve, and
    the two locals f2c keeps from one call to the next (refit and ier)."""

    def __init__(self, spec: dict):
        self.ic = icfitbits.Ic(function=spec["function"], order=int(spec["order"]),
                               sample=spec["sample"], naverage=int(spec["naverage"]),
                               low_reject=float(spec["low_reject"]),
                               high_reject=float(spec["high_reject"]),
                               niterate=int(spec["niterate"]), grow=float(spec["grow"]))
        self.curve = None
        self.refit = False
        self.ier = 0
        self.messages: list[str] = []

    def fit(self, x: np.ndarray, y: np.ndarray, wts: np.ndarray):
        """ic_fit(newx = newwts = newfunction = the first call, newy = YES)."""
        ic = self.ic
        if self.curve is None:
            xmin_all, xmax_all = icfitbits._alim(x)
            self.curve = icfitbits._setup(ic, x, wts, xmin_all, xmax_all)
            self.refit = False
        curve = self.curve
        if ic.nfit == x.size:
            xuse, yuse, wuse = x, y, wts
        else:
            ic.yfit, ic.wtsfit = icfitbits._wtbin(ic.ranges, ic.naverage, y, wts)
            xuse, yuse, wuse = ic.xfit, ic.yfit, ic.wtsfit
        if not self.refit:
            self.ier = icfitbits._cvfit(curve, xuse, yuse, wuse)
        else:
            refit(curve, xuse, yuse, wuse)
        fiterror = False
        if self.ier == SINGULAR:
            self.messages.append("Singular solution")
        elif self.ier == NO_DEG_FREEDOM:
            self.messages.append("No degrees of freedom")
            fiterror = True
        if not fiterror:
            self.refit = True
            if ic.low_reject > 0 or ic.high_reject > 0:
                icfitbits._reject(curve, xuse, yuse, wuse, ic)
                if ic.nreject > 0:
                    self.refit = False
            else:
                ic.nreject = 0
        return curve


def refit(curve, x, y, w) -> None:
    """cvrefit: the right-hand side from the new data, with the basis of the points,
    solved with the factorization the curve already has."""
    real = curve.real
    x = np.asarray(x, real)
    wy = (np.asarray(w, real) * np.asarray(y, real)).astype(real)
    if getattr(curve, "_refit_basis", None) is None:
        n = curve.npts
        curve._refit_basis = curve._basis(x[:n])
    basis, left = curve._refit_basis
    n = basis.shape[1]
    vector = np.zeros(curve.ncoeff, real)
    for k in range(curve.order):
        for i in range(n):
            row = k if left is None else int(left[i]) + k
            vector[row] = real(vector[row] + real(wy[i] * basis[k, i]))
    curve.vector = vector
    curve.coeff = curve._choslv(vector)


def evaluate(curve, x: np.ndarray) -> np.ndarray:
    """cveval at each x: cv_b1cheb, cv_b1leg (or the spline's single-point basis)
    and the sum, as for one point, in real."""
    real = curve.real
    x = np.asarray(x, real)
    if curve.kind in (CHEBYSHEV, LEGENDRE):
        order = curve.order
        basis = [np.ones(x.size, real) for _ in range(order)]
        if order > 1:
            xnorm = ((x + curve.maxmin).astype(real) * curve.range).astype(real)
            basis[1] = xnorm
            for i in range(3, order + 1):
                if curve.kind == CHEBYSHEV:
                    basis[i - 1] = (((real(2) * xnorm).astype(real) * basis[i - 2])
                                    .astype(real) - basis[i - 3]).astype(real)
                else:
                    ri = real(i)
                    a = ((real(real(2) * ri - real(3)) * xnorm).astype(real)
                         * basis[i - 2]).astype(real)
                    b = (real(ri - real(2)) * basis[i - 3]).astype(real)
                    basis[i - 1] = ((a - b).astype(real) / real(ri - real(1))).astype(real)
        total = np.zeros(x.size, real)
        for k in range(order):
            total = (total + (basis[k] * curve.coeff[k]).astype(real)).astype(real)
        return total
    b, lefts = curve._basis(x)
    total = np.zeros(x.size, real)
    for k in range(curve.order):
        total = (total + (b[k] * curve.coeff[lefts + k]).astype(real)).astype(real)
    return total


def vectors(data: np.ndarray, axis: int):
    """The image's vectors along the axis, in fit1d's order: lines, or columns."""
    return data if axis == 1 else data.T


def fit1d(data: np.ndarray, axis: int, kind: str, spec: dict, mask=None):
    """f1d_fit1d over the whole image (or section): the output values and what
    ic_fit printed."""
    data = np.asarray(data, f32)
    lines = vectors(data, axis)
    weights = None if mask is None else vectors(np.asarray(mask), axis)
    nx = lines.shape[1]
    x = np.arange(1, nx + 1, dtype=f32)
    fitter = Fitter(spec)
    out = np.empty_like(lines)
    for j in range(lines.shape[0]):
        y = lines[j].copy()
        w = np.ones(nx, f32) if weights is None else np.where(weights[j] == 0, f32(1),
                                                              f32(0)).astype(f32)
        curve = fitter.fit(x, y, w)
        if kind == "fit":
            out[j] = np.asarray(curve.vector_at(x), f32)
        elif kind == "difference":
            out[j] = (lines[j] - evaluate(curve, x)).astype(f32)
        else:
            div = evaluate(curve, x)
            div = np.where(np.abs(div) < f32(1e-20), f32(1), div).astype(f32)
            out[j] = (lines[j] / div).astype(f32)
    result = out if axis == 1 else out.T
    return np.ascontiguousarray(result, f32), fitter.messages
