"""IRAF's ic_fit as it runs, in real or in double: the sample ranges, the binning,
the fit and the rejection, operation for operation.

core/icfit.py is the fit the panels and the tasks of a reduction share; it has IRAF's
meaning but works in double on whatever curve is at hand. This is for what prints IRAF's
numbers (curfit), where the precision and the order of the sums are in the digits.

From iraf-community 2.18: pkg/xtools/icfit/{icfit,icdosetup,icreject,icdeviant,icfvshow}.gx
and pkg/xtools/ranges. See LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import advice, iraffmt
from .curfit import (
    CHEBYSHEV,
    FUNCTIONS,
    LEGENDRE,
    NO_DEG_FREEDOM,
    SINGULAR,
    SPLINE1,
    SPLINE3,
    Curve,
    DoubleCurve,
)

MAX_REAL = 3.4028234663852886e38  # hlib$mach.h, in a real


class FitError(Exception):
    """ic_deviant's error: the refit after a rejection has no solution."""


@dataclass
class Ic:
    """What ic_open and ic_pstr leave in the structure, and what ic_fit sets."""

    function: str = "legendre"
    order: int = 1
    sample: str = "*"
    naverage: int = 1
    low_reject: float = 0.0
    high_reject: float = 0.0
    niterate: int = 0
    grow: float = 0.0
    nfit: int = 0
    nreject: int = 0
    xfit: np.ndarray | None = None
    yfit: np.ndarray | None = None
    wtsfit: np.ndarray | None = None
    rejpts: np.ndarray | None = None
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _alim(values) -> tuple:
    return values.min(), values.max()


def _xranges(sample: str, x, warnings: list[str]) -> list[list[int]]:
    """rg_xranges, rg_order and rg_merge: index ranges, from 0 and inclusive."""
    real = type(x[0])
    npts = x.size
    ranges: list[list[int]] = []
    words = sample.replace(",", " ").split()
    expanded = []
    for word in words:
        if word.startswith("@"):
            try:
                with open(word[1:], encoding="utf-8") as file:
                    for line in file:
                        expanded.extend(line.replace(",", " ").split())
            except OSError:
                warnings.append(f"Cannot open file ({word[1:]})")
        else:
            expanded.append(word)
    for word in expanded:
        fields = word.replace(":", " ").split()
        if word.startswith("*"):
            r1, r2 = x[0], x[npts - 1]
        else:
            first = iraffmt.ctod(fields[0]) if fields else None
            if first is None:
                warnings.append("Range syntax error")
                continue
            r1 = r2 = real(first)
            if len(fields) > 1:
                second = iraffmt.ctod(fields[1])
                if second is not None:
                    r2 = real(second)
        a1, b1 = min(r1, r2), max(r1, r2)
        a2, b2 = min(x[0], x[npts - 1]), max(x[0], x[npts - 1])
        if b1 >= a2 and a1 <= b2:
            a1 = max(a2, min(b2, a1))
            b1 = max(a2, min(b2, b1))
            k = 0
            while k < npts and x[k] < a1:
                k += 1
            j = k
            while j < npts and x[j] <= b1:
                j += 1
            if k <= j - 1:
                ranges.append([k, j - 1])
    ranges.sort(key=lambda r: r[0])
    merged: list[list[int]] = []
    for first, last in ranges:
        if merged and first <= merged[-1][1] + 1:
            merged[-1][1] = max(last, merged[-1][1])
        else:
            merged.append([first, last])
    return merged


def _median(values) -> float:
    """amed, whose value is the (n+1)/2-th smallest for more than two points."""
    if values.size <= 2:
        return values[0]
    return np.sort(values)[(values.size + 1) // 2 - 1]


def _wtbin(ranges, nbin: int, data, wts) -> tuple:
    """rg_wtbin: out and wtout. Bins of abs(nbin), weighted means for a positive one
    and medians for a negative, and a short last bin is left out."""
    real = type(data[0])
    if abs(nbin) < 2:
        keep = np.concatenate([np.arange(a, b + 1) for a, b in ranges])
        return data[keep], wts[keep]
    npts = abs(nbin)
    out, wtout = [], []
    for first, last in ranges:
        nsample, j = 0, first
        while j <= last:
            n = max(0, min(npts, last - j + 1))
            if nsample > 0 and n < max(min(npts, 3), (npts + 1) // 2):
                break
            nsample += 1
            total = real(0)
            for value in wts[j : j + n]:
                total = real(total + value)
            wtout.append(total)
            if nbin > 0:
                value = real(0)
                if total != 0:
                    for k in range(j, j + n):
                        value = real(value + real(data[k] * wts[k]))
                    value = real(value / total)
                else:
                    for k in range(j, j + n):
                        value = real(value + data[k])
                    value = real(value / real(n))
            else:
                value = _median(data[j : j + n])
            out.append(value)
            j += n
    return np.array(out, real), np.array(wtout, real)


def _setup(ic: Ic, x, wts, xmin_all, xmax_all):
    """ic_dosetup: the sample, and the curve over the right range."""
    npts = x.size
    if npts == 0:
        advice.fail("No data points for fit", "the input has no points to fit.")
    real = type(x[0])
    ic.rejpts = np.zeros(npts, np.int32)
    ic.nreject = 0
    ranges = _xranges(ic.sample, x, ic.warnings)
    if ranges:
        ic.xfit, ic.wtsfit = _wtbin(ranges, max(1, abs(ic.naverage)), x, wts)
        ic.nfit = ic.xfit.size
    else:
        ic.nfit = 0
    if ic.nfit == 0:
        advice.fail("No sample points for fit", "the sample ranges hold no points.")
    ic.ranges = ranges
    if ic.nfit == npts:
        ic.xfit = ic.yfit = ic.wtsfit = None
        xmin, xmax = _alim(x)
    elif ic.nfit == 1:
        xmin, xmax = _alim(x)
    else:
        xmin, xmax = _alim(ic.xfit)
    # IC_XMIN and IC_XMAX are reals, also in a fit in double
    low = min(np.float32(xmin_all), np.float32(xmin))
    high = max(np.float32(xmax_all), np.float32(xmax))
    kind = FUNCTIONS.get(ic.function)
    make = Curve if real is np.float32 else DoubleCurve
    if kind in (LEGENDRE, CHEBYSHEV):
        return make(kind, min(ic.order, ic.nfit), xmin, xmax)
    pieces = min(ic.order, ic.nfit - (1 if kind == SPLINE1 else 3))
    assert kind in (SPLINE1, SPLINE3)
    if pieces > 0:
        return make(kind, pieces, real(low), real(high))
    return make(LEGENDRE, ic.nfit, real(low), real(high))


def _cvfit(curve, x, y, w) -> int:
    curve.matrix[...] = 0
    curve.vector[...] = 0
    curve.npts = 0
    curve.accumulate(x, y, w)
    return curve.solve()


def _deviant(curve, x, y, w, ic: Ic, nreject: int, refit: bool = True) -> tuple[int, int]:
    """ic_deviant: reject and refit once; nreject and how many are new. Without
    `refit`, as ic_clean calls it, the points are only marked."""
    real, rej = type(x[0]), ic.rejpts
    npts = x.size
    low, high, grow = (np.float32(v) for v in (ic.low_reject, ic.high_reject, ic.grow))
    residuals = y - curve.vector_at(x)
    j, nreject, sigma = 0, 0, real(0)
    for i in range(npts):
        if w[i] != 0 and rej[i] == 0:
            sigma = real(sigma + real(residuals[i] * residuals[i]))
            j += 1
        elif rej[i] == 1:
            nreject += 1
    if j < 5:
        return nreject, 0
    sigma = real(np.sqrt(real(sigma / real(j))))
    low_cut = real(real(-low) * sigma) if low > 0 else real(-MAX_REAL)
    high_cut = real(real(high) * sigma) if high > 0 else real(MAX_REAL)
    pixgrow = 0
    if grow > 0:
        for i in range(npts - 1):
            dx = abs(real(x[i + 1] - x[i]))
            if dx < real(np.float32(0.0001)):
                continue
            if i == 0:
                pixgrow = int(real(real(grow) / dx))
            else:
                pixgrow = int(max(real(real(grow) / dx), real(pixgrow)))
    newreject = 0
    for i in range(npts):
        if w[i] == 0 or rej[i] == 1:
            continue
        if low_cut < residuals[i] < high_cut:
            continue
        for k in range(max(0, i - pixgrow), min(npts - 1, i + pixgrow) + 1):
            if abs(real(x[i] - x[k])) <= real(grow) and w[k] != 0 and rej[k] == 0:
                if refit:
                    curve.reject(x[k], y[k], w[k])
                rej[k] = 2
                newreject += 1
    rej[rej != 0] = 1
    nreject += newreject
    if refit and newreject > 0:
        status = curve.solve()
        if status == SINGULAR:
            raise FitError("ic_reject: Singular solution")
        if status == NO_DEG_FREEDOM:
            raise FitError("ic_reject: No degrees of freedom")
    return nreject, newreject


def _reject(curve, x, y, w, ic: Ic) -> None:
    """ic_reject: an error goes back to the fit, and as many passes as went well."""
    ic.nreject = 0
    ic.rejpts[...] = 0
    if ic.niterate <= 0:
        return
    done = 0
    try:
        for _ in range(ic.niterate):
            ic.nreject, new = _deviant(curve, x, y, w, ic, ic.nreject)
            done += 1
            if new == 0:
                break
    except FitError:
        _cvfit(curve, x, y, w)
        ic.nreject = 0
        ic.rejpts[...] = 0
        for _ in range(done):
            try:
                ic.nreject, _new = _deviant(curve, x, y, w, ic, ic.nreject)
            except FitError as error:
                advice.fail(str(error), "reject less, or fit a lower order.")


def fit(ic: Ic, x, y, wts):
    """cf_fit up to the output: ic_putr xmin, xmax, and then ic_fit. The curve."""
    xmin_all, xmax_all = _alim(x) if x.size else (0.0, 0.0)
    curve = _setup(ic, x, wts, xmin_all, xmax_all)
    if ic.nfit == x.size:
        xuse, yuse, wuse = x, y, wts
    else:
        ic.yfit, ic.wtsfit = _wtbin(ic.ranges, ic.naverage, y, wts)
        xuse, yuse, wuse = ic.xfit, ic.yfit, ic.wtsfit
    status = _cvfit(curve, xuse, yuse, wuse)
    if status == SINGULAR:
        ic.messages.append("Singular solution")
    elif status == NO_DEG_FREEDOM:
        ic.messages.append("No degrees of freedom")
    elif ic.low_reject > 0 or ic.high_reject > 0:
        _reject(curve, xuse, yuse, wuse, ic)
    return curve


def show(ic: Ic, curve, x, y, wts, title: str, when: str) -> list[str]:
    """ic_fshow and ic_fvshow: the head of the output, with the coefficients."""
    real = type(x[0])
    g = iraffmt.g
    lines = [
        "",
        f"# {when}",
        f"# {title}",
        f"# function = {ic.function}",
        f"# grow = {g(ic.grow)}",
        f"# naverage = {ic.naverage}",
        f"# order = {ic.order}",
        f"# low_reject = {g(ic.low_reject)}",
        f"# high_reject = {g(ic.high_reject)}",
        f"# niterate = {ic.niterate}",
        f"# sample = {ic.sample}",
    ]
    npts = x.size
    sampled = npts != ic.nfit
    xuse, yuse = (ic.xfit, ic.yfit) if sampled else (x, y)
    wts1 = (ic.wtsfit if sampled else wts).copy()
    n = ic.nfit
    if ic.nreject > 0:
        wts1[ic.rejpts[:n] == 1] = real(0)
    # the weights the points came with, the first n also when they are not the ones fit
    deleted = int((wts[:n] == 0).sum())
    fitted = curve.vector_at(xuse)
    chisqr, errors = curve.errors(yuse, wts1, fitted)
    rms, used = real(0), 0
    for i in range(n):
        if wts1[i] == 0:
            continue
        resid = real(yuse[i] - fitted[i])
        rms = real(rms + real(resid * resid))
        used += 1
    if used > 0:
        rms = real(np.sqrt(real(rms / real(used))))
    lines += [
        f"# total points = {npts}",
        f"# sample points = {n}",
        f"# nrejected = {ic.nreject}",
        f"# deleted = {deleted}",
        f"# RMS = {number(rms, 'g', 7, 10)}",
        f"# square root of reduced chi square = {number(np.sqrt(chisqr), 'g', 7, 10)}",
        "# \t  coefficent\t  error",
    ]
    for i in range(curve.ncoeff):
        lines.append(f"# \t{number(curve.coeff[i], 'e', 7, 14)}"
                     f"\t{number(errors[i], 'e', 7, 14)}")
    return lines


def number(value, fmt: str, decpl: int, width: int) -> str:
    """%width.decpl fmt, padded on the left as fmtio pads."""
    text = iraffmt.printf(float(value), fmt, decpl, width)
    return text.rjust(width)
