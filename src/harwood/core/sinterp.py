"""sinterp: a table of x and y points interpolated -- at the x's a file asks for, along
a range, or into a spectrum.

From noao/onedspec/t_sinterp.x. The interpolants are xtools' lintrp (a straight
line) and intrp (curves: two parabolas blended), or a curfit fit through every
point of the table, legendre, chebyshev, spline3 or spline1 of an order, made once
and evaluated; all of it in real, as IRAF's is. A range is x1 and two of x2, dx and
npts, and then npts again from them, nint((x2 - x1)/dx + 1).

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math

import numpy as np

from . import advice, curfit, flux, iraffmt

f32 = np.float32
MODES = ("linear", "curves", "legendre", "chebyshev", "spline3", "spline1")


def read_table(text: str, maxpts: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """The lines that are two numbers (gargr's), in real."""
    xs, ys = [], []
    for line in text.splitlines():
        words = line.split()
        values = [iraffmt.ctod(w) for w in words[:2]]
        if len(values) < 2 or any(v is None or v > 1e300 for v in values):
            continue
        if len(xs) >= maxpts:
            advice.fail("Maximum table size exceeded.", "raise --tbl-size.")
        xs.append(values[0])
        ys.append(values[1])
    if not xs:
        advice.fail("Table has no entries.", "the table is lines of x and y.")
    return np.array(xs, dtype=f32), np.array(ys, dtype=f32)


def mode_of(text: str) -> str:
    """clgwrd: the mode a word is, an abbreviation as good as the whole."""
    found = [m for m in MODES if m.startswith(text)] if text not in MODES else [text]
    if len(found) != 1:
        advice.fail(f"Ambiguous or unknown interp_mode '{text}'",
                    "one of " + ", ".join(MODES) + ".")
    return found[0]


class Interpolant:
    """gen_pixel: the table's y at an x."""

    def __init__(self, xtab, ytab, mode: str, order: int = 5):
        self.xtab, self.ytab, self.mode = xtab, ytab, mode
        self.curve = None
        if mode not in ("linear", "curves"):
            self.curve = curfit.Curve(curfit.FUNCTIONS[mode], order, xtab[0], xtab[-1])
            self.curve.accumulate(xtab, ytab, np.ones(xtab.size, f32))
            self.curve.solve()

    def __call__(self, x) -> float:
        x = f32(x)
        if self.mode == "linear":
            return flux.lintrp(self.xtab, self.ytab, x)
        if self.mode == "curves":
            return flux.intrp(self.xtab, self.ytab, x)
        return float(self.curve(x))


def grid(x1: float, x2: float, dx: float, npts: int) -> tuple[float, float, int]:
    """The range's x2, dx and npts, as t_sinterp completes them."""
    x1, x2, dx = f32(x1), f32(x2), f32(dx)
    if x2 == 0.0:
        x2 = f32(x1 + f32(npts - 1) * dx)
    elif dx == 0.0:
        dx = f32((x2 - x1) / f32(npts - 1))
    if dx == 0.0 or dx * (x2 - x1) < 0.0:
        advice.fail("Interval paramater dx implies infinite loop.",
                    "give x2 and npts, or dx and npts, that run the same way from x1.")
    n = int(math.floor(float(f32((x2 - x1) / dx) + f32(1)) + 0.5))
    return float(x2), float(dx), n


def line(x, y) -> str:
    """"%12.5g  %12.5g" of two reals."""
    g = lambda v: iraffmt.printf(float(f32(v)), "g", 5, width=12).rjust(12)  # noqa: E731
    return f"{g(x)}  {g(y)}"
