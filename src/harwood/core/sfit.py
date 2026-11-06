"""sfit and continuum: a curve fitted to each spectrum (icfit), and the fit, the data,
their difference or their ratio written in its place.

From noao/onedspec/t_sfit.x (sft_init, sft_immap, sft_icfit, sft_getline, sft_power,
sft_update), pkg/xtools/ranges (rg_ranges, rg_encode, rg_union, rg_intersect,
rg_inverse, rg_window, rg_unpack, rg_pack), pkg/xtools/icfit/icclean.gx and
pkg/xtools/xtlogfiles.x. The fit is IRAF's ic_fit in real, core/icfitbits.py.

IRAF's ways kept: the lines and bands to fit are ranges over the image's lines (its
apertures, in order) and bands; what an image's SFIT and SFITB say was fitted already,
in the input or in an existing output, is not fitted again unless --override; the
output is a copy of the input with the fitted lines replaced; the log gives the power
series coefficients of a legendre or chebyshev fit; a ratio's division by zero and the
log of what is zero or negative are 0.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import advice, icfitbits

f32 = np.float32
#: libf2c's r_lg10: log10e * log(x), in double, of a real.
LOG10E = 0.43429448190325182765
TYPES = ("data", "fit", "difference", "ratio")


# ------------------------------------------------------------------ ranges


def rg_ranges(text: str, rmin: int, rmax: int) -> set[int]:
    """rg_ranges: the integers a range string names, from rmin to rmax. '*' is all of
    them, a:b (or a-b, which the task turns into a:b) from a to b, @file a list."""
    out: set[int] = set()
    for word in text.replace(",", " ").split():
        if word.startswith("@"):
            lines = Path(word[1:]).read_text(encoding="utf-8").splitlines()
            for line in lines:
                out |= _rg_add(line, rmin, rmax)
        else:
            out |= _rg_add(word, rmin, rmax)
    return out


def _rg_add(text: str, rmin: int, rmax: int) -> set[int]:
    out: set[int] = set()
    for word in text.replace(",", " ").split():
        if word.startswith("@"):
            advice.fail("Cannot nest @files", "list the ranges in the file itself.")
        if word.startswith("*"):
            a, b = rmin, rmax
        else:
            parts = [p for p in word.split(":") if p]
            try:
                a = _ctoi(parts[0])
                b = _ctoi(parts[1]) if len(parts) > 1 else a
            except (IndexError, ValueError):
                advice.fail("Range syntax error", "write ranges as 1,3:5 or *.")
        a, b = min(a, b), max(a, b)
        if b >= rmin and a <= rmax:
            out |= set(range(max(rmin, a), min(rmax, b) + 1))
    return out


def _ctoi(text: str) -> int:
    digits = ""
    for i, char in enumerate(text):
        if char.isdigit() or (i == 0 and char in "+-"):
            digits += char
        else:
            break
    return int(digits)


def rg_encode(values: set[int]) -> str:
    """rg_encode of the merged ranges: a:b for a run, a for one."""
    runs, ordered = [], sorted(values)
    for value in ordered:
        if runs and value == runs[-1][1] + 1:
            runs[-1][1] = value
        else:
            runs.append([value, value])
    return ",".join(f"{a}:{b}" if a != b else f"{a}" for a, b in runs)


# ------------------------------------------------------------------ the fit


def alog(values) -> np.ndarray:
    """alogr with sft_efncr: log10 in real, and 0 for what is zero or negative."""
    out = np.zeros(len(values), dtype=f32)
    for i, v in enumerate(np.asarray(values, f32)):
        if v > 0:
            out[i] = f32(LOG10E * math.log(float(v)))
    return out


def divide(a, b) -> np.ndarray:
    """advzr with sft_efncr: a/b in real, and 0 where b is 0."""
    a, b = np.asarray(a, f32), np.asarray(b, f32)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a / np.where(b == 0, f32(1), b)).astype(f32)
    return np.where(b == 0, f32(0), out).astype(f32)


def clean(ic: icfitbits.Ic, curve, x, y, w) -> None:
    """ic_clean: the rejected points of y replaced by the fit, in place."""
    if ic.low_reject == 0 and ic.high_reject == 0:
        return
    npts = x.size
    if npts == ic.nfit:
        if ic.nreject > 0:
            for i in range(npts):
                if ic.rejpts[i] == 1:
                    y[i] = curve(x[i])
        return
    packed = np.concatenate([np.arange(a, b + 1) for a, b in ic.ranges])
    if abs(ic.naverage) < 2:
        yfit = np.array(ic.yfit, dtype=f32)
        if ic.nreject > 0:
            for i in range(yfit.size):
                if ic.rejpts[i] == 1:
                    yfit[i] = curve(ic.xfit[i])
        y[packed] = yfit
        return
    if packed.size == npts:
        ic.rejpts = np.zeros(npts, np.int32)
        ic.nreject, _ = icfitbits._deviant(curve, x, y, w, ic, ic.nreject, refit=False)
        if ic.nreject > 0:
            for i in range(npts):
                if ic.rejpts[i] == 1:
                    y[i] = curve(x[i])
        return
    xc, yc, wc = x[packed].copy(), y[packed].copy(), w[packed].copy()
    ic.rejpts = np.zeros(npts, np.int32)
    keep = ic.rejpts
    ic.rejpts = np.zeros(packed.size, np.int32)
    ic.nreject, _ = icfitbits._deviant(curve, xc, yc, wc, ic, ic.nreject, refit=False)
    keep[:packed.size] = ic.rejpts
    ic.rejpts = keep
    if ic.nreject > 0:
        for i in range(packed.size):
            if ic.rejpts[i] == 1:
                yc[i] = curve(xc[i])
    y[packed] = yc


def fit(x, y, spec: dict, outtype: str, replace: bool, weights=None):
    """sft_icfit on one spectrum, not interactive: (the output, the curve, ic, and what
    ic_fit printed). x and y are reals; y is cleaned in place with --replace."""
    ic = icfitbits.Ic(function=spec["function"], order=spec["order"],
                      sample=spec["sample"], naverage=spec["naverage"],
                      low_reject=spec["low_reject"], high_reject=spec["high_reject"],
                      niterate=spec["niterate"], grow=spec["grow"])
    w = np.ones(x.size, f32) if weights is None else np.asarray(weights, f32)
    curve = icfitbits.fit(ic, x, y, w)
    if outtype == "data":
        if replace:
            clean(ic, curve, x, y, w)
        out = y.copy()
    elif outtype == "fit":
        out = np.asarray(curve.vector_at(x), f32)
    elif outtype == "difference":
        values = np.asarray(curve.vector_at(x), f32)
        if replace:
            clean(ic, curve, x, y, w)
        out = (y - values).astype(f32)
    else:
        values = np.asarray(curve.vector_at(x), f32)
        if replace:
            clean(ic, curve, x, y, w)
        out = divide(y, values)
    return out, curve, ic


def power(curve, function: str, line: int, image: str) -> list[str]:
    """sft_power: the power series coefficients of a legendre or chebyshev fit."""
    if function not in ("legendre", "chebyshev"):
        return []
    coeff = curve.power()
    out = [f"Line {line} of {image}:", "", "      coeff           value"]
    out += [f"\t{j}\t" + icfitbits.number(value, "e", 5, 12)
            for j, value in enumerate(coeff, 1)]
    return out + [""]
