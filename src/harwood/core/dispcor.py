"""IRAF's dispcor (noao.onedspec), ported: the dispersion function, its
inverse and the resampling.

From iraf-community 2.18: sys/mwcs/wfmspec.x (the multispec dispersion
function -- a weighted sum of curfit curves, each plus a shift -- and its
inverse, by Newton iteration on a 0.0001 px grid, starting where the last
inversion ended), noao/onedspec/dispcor/dcio.x (two reference arcs combined
by their REFSPEC weights), t_dispcor.x (dc_defaults: the output grid) and
dispcor.x (dispcora: each output pixel the integral of an interpolant of
the input over the input pixels it covers -- counts when flux is conserved,
their mean when not).

Pixels are 1-based.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .asi import Interpolant

f32 = np.float32
NIT = 10
DX = 0.0001
CHEBYSHEV, LEGENDRE, SPLINE3, SPLINE1 = 1, 2, 3, 4
FUNCTIONS = {"chebyshev": CHEBYSHEV, "legendre": LEGENDRE, "spline3": SPLINE3,
             "spline1": SPLINE1}


def nint(value) -> int:
    v = float(value)
    return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(-v + 0.5))


@dataclass
class Curve:
    """A curfit curve as IRAF saves it: type, order, xmin, xmax and the
    coefficients."""

    type: int
    order: int
    xmin: float
    xmax: float
    coeffs: list[float]

    @classmethod
    def from_saved(cls, values) -> Curve:
        values = [float(v) for v in values]
        return cls(int(round(values[0])), int(round(values[1])), values[2], values[3],
                   values[4:])

    def __call__(self, x: float) -> float:
        """wf_msp_eval's term for this curve (no clamping here)."""
        c = self.coeffs
        order = self.order
        if self.type in (CHEBYSHEV, LEGENDRE):
            centre = (self.xmax + self.xmin) / 2
            scale = 2 / (self.xmax - self.xmin)
            xx = (x - centre) * scale
            y = c[0]
            if order > 1:
                y += c[1] * xx
            if order > 2:
                a, b = 1.0, xx
                for i in range(3, order + 1):
                    if self.type == CHEBYSHEV:
                        cc = 2 * xx * b - a
                    else:
                        cc = ((2 * i - 3) * xx * b - (i - 2) * a) / (i - 1)
                    y += c[i - 1] * cc
                    a, b = b, cc
            return y
        scale = order / (self.xmax - self.xmin)
        xx = (x - self.xmin) * scale
        i = max(0, min(int(xx), order - 1))
        b = xx - i
        a = 1 - b
        if self.type == SPLINE3:
            return (a * a * a * c[i] + (1 + 3 * a * (1 + a * b)) * c[i + 1]
                    + (1 + 3 * b * (1 + a * b)) * c[i + 2] + b * b * b * c[i + 3])
        if self.type == SPLINE1:
            return a * c[i] + b * c[i + 1]
        raise ValueError(f"curve type {self.type} is not supported")


class Dispersion:
    """A multispec dispersion function: sum of weight * (curve + shift) over
    the references, normalised by the weights, for a spectrum of npts."""

    def __init__(self, terms: list[tuple[float, float, Curve]], npts: int):
        total = sum(w for w, _, _ in terms)
        # dcio prints the normalised weights with %.8g and reads them back.
        self.terms = [(float(f"{w / total:.8g}"), s, c) for w, s, c in terms]
        self.npts = npts
        self.xmin, self.xmax = 0.5, npts + 0.5
        lo, hi = self.eval(self.xmin), self.eval(self.xmax)
        self.ymin, self.ymax = min(lo, hi), max(lo, hi)
        self.x = float(npts)
        self.dydx = self.eval(float(npts)) - self.eval(float(npts - 1))

    def eval(self, x: float) -> float:
        """wf_msp_eval."""
        xval = max(self.xmin, min(self.xmax, x))
        ysum = wsum = 0.0
        for weight, shift, curve in self.terms:
            ysum += weight * (shift + curve(xval))
            wsum += weight
        return ysum / wsum

    def __call__(self, x):
        return np.array([self.eval(float(v)) for v in np.atleast_1d(x)])

    def inverse(self, y: float) -> float:
        """wf_msp_evali, continuing from the last inversion."""
        yval = max(self.ymin, min(self.ymax, y))
        x, dydx = self.x, self.dydx
        dx = dy = 0.0
        converged = False
        for _ in range(NIT):
            y1 = self.eval(x)
            if dx > 1:
                dy = self.eval(x + 1) - y1 if x + 1 < self.xmax else y1 - self.eval(x - 1)
            elif dx < -1:
                dy = y1 - self.eval(x - 1) if x - 1 > self.xmin else self.eval(x + 1) - y1
            if dy != 0.0:
                dydx = dy
            dx = (yval - y1) / dydx
            x = max(self.xmin, min(self.xmax, x + dx))
            if abs(dx) < DX:
                converged = True
                break
        if not converged:
            xval = (self.xmin + self.xmax) / 2
            best = abs(self.eval(xval) - y)
            step = (self.xmax - self.xmin) / 18
            while step > DX:
                v = max(self.xmin, xval - 9 * step)
                top = min(self.xmax, xval + 9 * step)
                while v <= top:
                    d = abs(self.eval(v) - y)
                    if d < best:
                        xval, best = v, d
                    v += step
                step /= 10
            x = xval
            dy = (self.eval(x + 1) - self.eval(x) if x + 1 < self.xmax
                  else self.eval(x) - self.eval(x - 1))
            if dy != 0.0:
                dydx = dy
        self.x, self.dydx = x, dydx
        whole = float(int(x))
        return whole + nint((x - whole) / DX) * DX


def defaults(a: float, b: float, n: int, w1=None, w2=None, dw=None, nw=None):
    """dc_defaults: complete the output grid (w1, w2, dw, nw) from what is
    given and the solution's own ends a, b and length n."""
    nindef = sum(v is None for v in (w1, w2, dw, nw))
    if nindef == 0:
        nw = int((w2 - w1) / dw + 1.5)
        w2 = w1 + dw * (nw - 1)
    elif nindef == 1:
        if w1 is None:
            w1 = w2 - dw * (nw - 1)
        if w2 is None:
            w2 = w1 + dw * (nw - 1)
        if dw is None:
            dw = (w2 - w1) / (nw - 1)
        if nw is None:
            nw = int((w2 - w1) / dw + 1.5)
            w2 = w1 + dw * (nw - 1)
    elif nindef == 2:
        if dw is None:
            if w1 is None:
                w1 = a if abs(w2 - a) > abs(w2 - b) else b
                dw = (w2 - w1) / (nw - 1)
            elif w2 is None:
                w2 = a if abs(w1 - a) > abs(w1 - b) else b
                dw = (w2 - w1) / (nw - 1)
            else:
                dw = (b - a) / n
                nw = int(abs((w2 - w1) / dw) + 1.5)
                dw = (w2 - w1) / (nw - 1)
        elif nw is None:
            if w1 is None:
                w1 = min(a, b) if dw > 0 else max(a, b)
                nw = int((w2 - w1) / dw + 1.5)
                w1 = w2 - dw * (nw - 1)
            else:
                w2 = max(a, b) if dw > 0 else min(a, b)
                nw = int((w2 - w1) / dw + 1.5)
                w2 = w1 + dw * (nw - 1)
        else:
            w1 = min(a, b) if dw > 0 else max(a, b)
            w2 = w1 + dw * (nw - 1)
    elif nindef == 3:
        if w1 is not None:
            w2 = a if abs(w1 - a) > abs(w1 - b) else b
            dw = (b - a) / n
            nw = int(abs((w2 - w1) / dw) + 1.5)
            dw = (w2 - w1) / (nw - 1)
        elif w2 is not None:
            w1 = a if abs(w2 - a) > abs(w2 - b) else b
            dw = (b - a) / n
            nw = int(abs((w2 - w1) / dw) + 1.5)
            dw = (w2 - w1) / (nw - 1)
        elif nw is not None:
            w1, w2 = min(a, b), max(a, b)
            dw = (w2 - w1) / (nw - 1)
        elif dw < 0:
            w1, w2 = max(a, b), min(a, b)
            nw = int((w2 - w1) / dw + 1.5)
            w2 = w1 + dw * (nw - 1)
        else:
            w1, w2 = min(a, b), max(a, b)
            nw = int((w2 - w1) / dw + 1.5)
            w2 = w1 + dw * (nw - 1)
    else:
        w1, w2 = min(a, b), max(a, b)
        dw = (b - a) / (n - 1)
        nw = int(abs((w2 - w1) / dw) + 1.5)
        dw = (w2 - w1) / (nw - 1)
    return w1, w2, dw, nw


def resample(flux, dispersion: Dispersion, w1: float, dw: float, nw: int,
             conserve: bool = True, blank: float | None = 0.0, interp: str = "poly5",
             log: bool = False) -> np.ndarray:
    """dispcora: the spectrum on the grid w1 + (i - 1) dw, i = 1..nw (in
    log10 if `log`)."""
    flux = np.asarray(flux, dtype=f32)
    npts = flux.size
    extended = np.concatenate(([flux[0]], flux, [flux[-1]])).astype(f32)
    asi = Interpolant(interp, extended)
    xmin, xmax = 0.5, npts + 0.5

    def world(x):
        w = w1 + (x - 1) * dw
        return 10 ** w if log else w

    out = np.zeros(nw, dtype=f32)
    x = dispersion.inverse(world(0.5))
    ofb_b = x < xmin or x > xmax
    b = f32(max(xmin, min(xmax, x)) + 1)
    for i in range(1, nw + 1):
        ofb_a, a = ofb_b, b
        x = dispersion.inverse(world(i + 0.5))
        ofb_b = x < xmin or x > xmax
        b = f32(max(xmin, min(xmax, x)) + 1)
        if (ofb_a and ofb_b) or a == b:
            if blank is not None:
                out[i - 1] = f32(blank)
            continue
        lo, hi = (a, b) if a < b else (b, a)
        ia, ib = nint(lo + f32(0.5)), nint(hi - f32(0.5))
        if abs(lo + f32(0.5) - ia) < 0.00001 and abs(hi - f32(0.5) - ib) < 0.00001:
            total = f32(0)
            for j in range(ia, ib + 1):
                total = f32(total + f32(asi(float(j))))
            value = total
        else:
            value = f32(asi.integral(float(lo), float(hi)))
        if not conserve:
            value = f32(value / max(f32(hi - lo), f32(1e-4)))
        out[i - 1] = value
    return out
