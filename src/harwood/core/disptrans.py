"""disptrans: a spectrum's dispersion in other units -- nanometers, microns, hertz,
inverse centimeters, km/s from a line -- and from air to vacuum or back.

From noao/onedspec/dispcor/t_disptrans.x. Each pixel's coordinate is converted
(core/units.py) and, for air and vacuum, scaled by the index of refraction of
Edlen's formula as IRAF writes it, for a temperature, a pressure and a water vapour
pressure. Then a function is fitted to the new coordinates against the pixels: a
straight line, Chebyshev of order 2, if it is within the error allowed of every
pixel (in pixels, the coordinate's step about it), else a cubic spline of one piece
more until one is; the coordinates are the function's, its coefficients through
%g, as the world coordinates hold them. linearize resamples the pixels onto an even
grid between the function's ends, not conserving flux.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import numpy as np

from . import curfit, dispcor, iraffmt, units

AIRVAC = ("none", "air2vac", "vac2air")


def airvac(values, un: units.Units, air: str, t: float, p: float, f: float) -> np.ndarray:
    """dt_airvac: the coordinates in microns, times or over the index of refraction."""
    values = np.asarray(values, dtype=float)
    if air == "none":
        return values
    microns = units.decode("microns")
    x = units.convert(values, un, microns)
    out = []
    for v in x:
        x2 = 1 / v ** 2
        a = 64.328 + 29498.1 / (146 - x2) + 255.4 / (41 - x2)
        a = a * p * (1 + (1.049 - 0.0157 * t) * 1e-6 * p) / (720.883 * (1 + 0.003661 * t))
        a = a - (0.0624 - 0.000680 * x2) / (1 + 0.003661 * t) * f
        a = 1 + a / 1e6
        out.append(a * v if air == "air2vac" else v / a)
    return units.convert(out, microns, un)


def _fit(kind: int, order: int, x, y):
    """dt_cvfit: the curve and its worst error, in pixels."""
    curve = curfit.DoubleCurve(kind, order, x[0], x[-1])
    curve.accumulate(x, y, np.ones(x.size))
    curve.solve()
    fit = np.array([float(curve(v)) for v in x])
    n = x.size
    w = np.empty(n)
    w[1:-1] = np.abs((y[1:-1] - fit[1:-1]) / ((y[2:] - y[:-2]) / 2))
    w[0] = abs((y[0] - fit[0]) / (y[1] - y[0]))
    w[-1] = abs((y[-1] - fit[-1]) / (y[-1] - y[-2]))
    return curve, float(w.max())


def fit(y, maxerr: float):
    """The dispersion function of the new coordinates: whether it is linear, and the
    curve, as dispcor's Curve with its coefficients through %g."""
    n = len(y)
    x = np.arange(1.0, n + 1.0)
    y = np.asarray(y, dtype=float)
    curve, err = _fit(curfit.CHEBYSHEV, 2, x, y)
    linear = True
    if err > maxerr:
        linear = False
        for order in range(1, n - 3):
            curve, err = _fit(curfit.SPLINE3, order, x, y)
            if err <= maxerr:
                break
    order = curve.order if curve.kind == curfit.CHEBYSHEV else curve.npieces + 1
    # The world coordinates' codes are IRAF's curfit's, chebyshev 1 and spline3 3.
    kind = dispcor.CHEBYSHEV if curve.kind == curfit.CHEBYSHEV else dispcor.SPLINE3
    text = lambda v: iraffmt.ctod(iraffmt.g(float(v), double=True))  # noqa: E731
    saved = [kind, order, text(curve.xmin), text(curve.xmax),
             *[text(c) for c in curve.coeff]]
    return linear, dispcor.Curve.from_saved(saved)


def transform(wave, flux, from_units: str, to_units: str, air: str, t: float, p: float,
              f: float, maxerr: float, linearize: bool, interp: str = "poly5"):
    """One aperture: its new coordinates and pixels, and whether the function is linear."""
    a, b = units.decode(from_units or "angstroms"), units.decode(to_units)
    y = airvac(units.convert(wave, a, b), b, air, t, p, f)
    n = len(y)
    linear, curve = fit(y, maxerr)
    w1 = curve(1.0)
    dw = (curve(float(n)) - w1) / (n - 1)
    x = np.arange(1.0, n + 1.0)
    if linear:
        return w1 + dw * (x - 1), np.asarray(flux, dtype=float), True, b
    if not linearize:
        return np.array([curve(v) for v in x]), np.asarray(flux, dtype=float), False, b
    dispersion = dispcor.Dispersion([(1.0, 0.0, curve)], n)
    out = dispcor.resample(flux, dispersion, w1, dw, n, conserve=False, blank=0.0,
                           interp=interp)
    return w1 + dw * (x - 1), out.astype(float), False, b
