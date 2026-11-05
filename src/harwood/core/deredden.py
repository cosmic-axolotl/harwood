"""deredden: a spectrum corrected for interstellar extinction, by Cardelli, Clayton
& Mathis's law (1989, ApJ 345, 245) with A(V)/E(B-V) = R.

From noao/onedspec/t_deredden.x. The extinction is given as A(V), E(B-V) or c, the
logarithmic extinction at H-beta, A(V) = R c (0.61 + 0.024 c); each pixel is
multiplied by 10^(0.4 A(V) A(lambda)/A(V)). The header's DEREDDEN card records it,
with the date, and a spectrum that has one is not corrected again unless the
correction is overridden -- and then, unless told otherwise, the old one is taken
out first, its A(V) and R read back from that card.

All of it is in real, as IRAF's is: the wavelengths, the law's polynomials and the
power of ten.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import re

import numpy as np

from . import advice, iraffmt

f32 = np.float32
TYPES = ("A(V)", "E(B-V)", "c")


def ccm(wavelength, rv) -> np.ndarray:
    """A(lambda)/A(V) at wavelengths in Angstroms, in real."""
    r = f32
    x = r(10000.0) / np.asarray(wavelength, dtype=r)
    rv = r(rv)
    if (x < r(0.3)).any() or (x > r(10.0)).any():
        advice.fail("Wavelength out of range of extinction function",
                    "Cardelli, Clayton & Mathis's law is defined from 1000 to 33333 A; "
                    "trim the spectrum to that range first.")
    a = np.empty_like(x)
    b = np.empty_like(x)
    ir = x < r(1.1)
    y = x[ir] ** r(1.61)
    a[ir], b[ir] = r(0.574) * y, r(-0.527) * y
    opt = (x >= r(1.1)) & (x < r(3.3))
    y = x[opt] - r(1.82)
    a[opt] = r(1) + y * (r(0.17699) + y * (r(-0.50447) + y * (r(-0.02427) + y * (
        r(0.72085) + y * (r(0.01979) + y * (r(-0.77530) + y * r(0.32999)))))))
    b[opt] = y * (r(1.41338) + y * (r(2.28305) + y * (r(1.07233) + y * (r(-5.38434) + y * (
        r(-0.62251) + y * (r(5.30260) + y * r(-2.09002)))))))
    uv = (x >= r(3.3)) & (x < r(8.0))
    xu = x[uv]
    y = (xu - r(4.67)) ** 2
    au = r(1.752) - r(0.316) * xu - r(0.104) / (y + r(0.341))
    y = (xu - r(4.62)) ** 2
    bu = r(-3.090) + r(1.825) * xu + r(1.206) / (y + r(0.263))
    far = xu >= r(5.9)
    y = xu[far] - r(5.9)
    au[far] = au[far] - r(0.04473) * y ** 2 - r(0.009779) * y ** 3
    bu[far] = bu[far] + r(0.2130) * y ** 2 + r(0.1207) * y ** 3
    a[uv], b[uv] = au, bu
    fuv = x >= r(8.0)
    y = x[fuv] - r(8)
    a[fuv] = r(-1.072) - r(0.628) * y + r(0.137) * y ** 2 - r(0.070) * y ** 3
    b[fuv] = r(13.670) + r(4.257) * y - r(0.420) * y ** 2 + r(0.374) * y ** 3
    return a + b / rv


def correct(wavelength, flux, av, rv, avold=0.0, rvold=None) -> np.ndarray:
    """deredden: the fluxes times 10^(0.4 (A(V) law(R) - A(V)old law(Rold)))."""
    r = f32
    av, rv = r(av), r(rv)
    avold = r(avold)
    rvold = rv if rvold is None else r(rvold)
    if avold != 0:
        if rv != rvold:
            power = r(0.4) * (av * ccm(wavelength, rv) - avold * ccm(wavelength, rvold))
        else:
            power = r(0.4) * (av - avold) * ccm(wavelength, rv)
    else:
        power = r(0.4) * av * ccm(wavelength, rv)
    cor = r(10.0) ** power.astype(r)
    return np.asarray(flux, dtype=r) * cor


def to_av(value: float, rv: float, kind: str) -> tuple[float, str]:
    """The A(V) of the extinction parameter, and the words the log card says it in."""
    g = lambda v: iraffmt.g(float(f32(v)))  # noqa: E731
    value, rv = f32(value), f32(rv)
    if kind == "A(V)":
        return float(value), f"A(V)={g(value)} R={g(rv)}"
    if kind == "E(B-V)":
        av = f32(rv * value)
        return float(av), f"E(B-V)={g(value)} A(V)={g(av)} R={g(rv)}"
    av = f32(rv * value * (f32(0.61) + f32(0.024) * value))
    return float(av), f"c={g(value)} A(V)={g(av)} R={g(rv)}"


def cnvdate(when: float | None = None) -> str:
    """IRAF's cnvdate (iraffmt's)."""
    return iraffmt.cnvdate(when)

def previous(card: str, rv: float) -> tuple[float, float]:
    """The A(V) and R an old DEREDDEN card says, as t_deredden reads its words back:
    R is the one given when the card has none."""
    avold, rvold = 0.0, float(f32(rv))
    for word in str(card).split():
        found = re.match(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", word[5:])
        if word.startswith("A(V)=") and found:
            avold = float(found.group(0))
        found = re.match(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", word[2:])
        if word.startswith("R=") and found:
            rvold = float(found.group(0))
    return avold, rvold
