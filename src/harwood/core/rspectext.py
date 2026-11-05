"""rspectext: a spectrum from a text file of wavelengths and fluxes, the other way
round from wspectext.

From noao/onedspec/rspectext.cl and t_rstext.x. The file is lines of x and y, the
lines that are not two numbers skipped, after a FITS header if it has one (cards up
to END, as wspectext with its header writes). Both columns go through %g on their
way, as IRAF's script hands them to rtextimage and dispcor as text: seven figures,
the fluxes then in real and the wavelengths in double.

With a header the spectrum is what the header says. Without one the title, whether
it is flux calibrated, and the dispersion are given: none, linear or log from crval1
and cdelt1 (the x column is not read), nonlinear (the x column is the wavelength of
each pixel, a lookup table) or interp (the same, resampled by dispcor onto an even
grid from the first to the last).

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import numpy as np

from . import iraffmt

f32 = np.float32
DTYPES = ("none", "linear", "log", "nonlinear", "interp")


def _gargr(word: str) -> float | None:
    value = iraffmt.ctod(word)
    if value is None or value > 1e300:
        return None
    return float(value)


def read(text: str) -> tuple[list[str] | None, np.ndarray, np.ndarray]:
    """t_rstext: the header's cards (None without one), and the x and y of the lines
    that are two numbers, each as %g of a real."""
    lines = text.splitlines()
    header = None
    body = lines
    for at, line in enumerate(lines):
        if line.startswith("END"):
            header, body = lines[:at + 1], lines[at + 1:]
            break
    xs, ys = [], []
    for line in body:
        words = line.split()
        if len(words) < 2:
            continue
        x, y = _gargr(words[0]), _gargr(words[1])
        if x is None or y is None:
            continue
        # Read in real and printed with %g: the wavelengths go on to dispcor as
        # text, read back in double; the fluxes to rtextimage, stored in real.
        xs.append(float(iraffmt.g(float(f32(x)))))
        ys.append(float(f32(float(iraffmt.g(float(f32(y)))))))
    return header, np.array(xs), np.array(ys)


def wavelengths(dtype: str, x: np.ndarray, crval1: float, cdelt1: float):
    """The wavelength of each pixel, before any resampling (None for none)."""
    n = x.size
    pixel = np.arange(n, dtype=float)
    if dtype == "none":
        return None
    if dtype == "linear":
        return crval1 + cdelt1 * pixel
    if dtype == "log":
        return 10 ** (crval1 + cdelt1 * pixel)
    return x.copy()                                   # nonlinear, and interp's table


def interpolated(x: np.ndarray, y: np.ndarray, interp: str = "poly5"):
    """dispcor on the lookup table: the fluxes on an even grid from its first value
    to its last, as many pixels, not flux conserving. Gives the wavelengths and
    the fluxes."""
    from . import dispcor

    n = x.size
    curve = dispcor.Curve(dispcor.PIXEL, n, 0.0, 0.0, list(x))
    make = lambda: dispcor.Dispersion([(1.0, 0.0, curve)], n)  # noqa: E731
    first = make()
    a, b = first.eval(1.0), first.eval(float(n))
    w1, _, dw, nw = dispcor.grid_for(dispcor.Grid(), a, b, n)
    flux = dispcor.resample(y, make(), w1, dw, nw, conserve=False, blank=0.0,
                            interp=interp)
    return w1 + np.arange(nw) * dw, flux.astype(float)


def header_dispersion(cards, n: int):
    """The wavelengths a header gives a one-dimensional spectrum: CRVAL1, CDELT1 (or
    CD1_1) and CRPIX1, in log10 when DC-FLAG is 1; None without a dispersion."""
    get = {str(c.keyword): c.value for c in cards}
    flag = int(get.get("DC-FLAG", -1))
    if flag < 0 or "CRVAL1" not in get:
        return None
    step = float(get.get("CDELT1", get.get("CD1_1", 1.0)))
    pixel = np.arange(1.0, n + 1.0) - float(get.get("CRPIX1", 1.0))
    world = float(get["CRVAL1"]) + step * pixel
    return 10 ** world if flag == 1 else world
