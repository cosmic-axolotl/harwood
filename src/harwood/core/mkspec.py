"""mkspec: an artificial spectrum -- flat, a ramp, or a blackbody -- of so many
pixels, repeated on so many lines.

From noao/onedspec/t_mkspec.x, in real as IRAF computes it -- the powers by
squaring as f2c's pow_ri, the exponential in double -- so that its pixels are IRAF's
to the bit: the blackbody's
wavelength is stepped along pixel by pixel, the error of each step carried on, and
the curve divided by its peak, 1.2865e-4 T^5 erg/cm2/s/cm (Wien's), so that a
temperature whose peak is in the range reaches 1 there. The spectrum has no
wavelengths: dispcor or rspectext give it some.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math

import numpy as np

from . import advice

f32 = np.float32


def _pow(a, n: int) -> np.float32:
    """A real to a whole power as f2c's pow_ri goes: by squaring, in real."""
    p, x = f32(1.0), f32(a)
    while True:
        if n & 1:
            p = f32(p * x)
        n >>= 1
        if not n:
            return p
        x = f32(x * x)


def row(ncols: int, function: int, constant: float = 0.0, start_level: float = 0.0,
        end_level: float = 0.0, start_wave: float = 0.0, end_wave: float = 0.0,
        temperature: float = 0.0) -> np.ndarray:
    """One line of the spectrum, in real: function 1 flat, 2 a ramp, 3 a blackbody."""
    r = f32
    out = np.empty(ncols, dtype=r)
    if function == 1:
        out[:] = r(constant)
    elif function == 2:
        c1, c2 = r(start_level), r(end_level)
        dconst = r((c2 - c1) / r(ncols - 1))
        for i in range(ncols):
            out[i] = r(c1 + r(i) * dconst)
    elif function == 3:
        k1, k2 = r(3.7415e-5), r(1.4388)
        wstart, wend, temp = r(start_wave), r(end_wave), r(temperature)
        dw = r((wend - wstart) / r(ncols - 1))
        w = r(wstart * r(1.0e-8))
        fmax = r(r(1.2865e-4) * _pow(temp, 5))
        with np.errstate(over="ignore"):
            for i in range(ncols):
                # exp of the real argument in double, as the C library's is.
                x = r(math.exp(float(r(r(k2 / w) / temp))))
                out[i] = r(r(r(k1 / _pow(w, 5)) / r(x - r(1.0))) / fmax)
                w = r(w + r(dw * r(1.0e-8)))
    else:
        advice.fail("Unknown Function type", "function is 1 (flat), 2 (a ramp) or 3 (a "
                    "blackbody).")
    return out
