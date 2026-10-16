# ruff: noqa: E741, SIM102, SIM108, SIM109, SIM114
# A line-by-line port keeps IRAF's names (l, j, nl, nh) and its nesting, so
# that it can be read beside the .gx it came from.
"""IRAF's imcombine: the combining and the rejection, as IRAF does them.

A port of `images.immatch.imcombine` (iraf-community 2.18,
pkg/images/immatch/src/imcombine/src: icomb.gx, icscale.x, icstat.gx,
icaverage.gx, icmedian.gx, icsigma.gx, icmm.gx, icsclip.gx, icaclip.gx,
iccclip.gx, icpclip.gx, icgrow.gx), checked against IRAF's own output by
tests/test_imcombine_iraf.py.

Why a line-by-line port and not "the same idea in numpy": the result of a
rejection depends on details no description mentions -- the order the values
sit in when one is swapped out, residuals kept to add values back when
`nkeep` is not met, which median is used on the last pass, a sigma pooled over
a whole image line for avsigclip. So each pixel here carries what IRAF's
buffers carry: its values (`d`), which image each came from (`ids`) and how
many are in use (`n`), and every algorithm moves them the way IRAF moves
them. The combine and the sigma image then read the first `n` values, as
IRAF's do.

Two places where IRAF's code does something its authors cannot have meant,
reproduced because they change the numbers IRAF prints, and marked where they
happen:

* asigclip with sigma scaling (`doscale1`): after a rejected value is swapped
  out, the value swapped in is re-examined with the scaling weight of the
  *next* position;
* aavsigclip with `nkeep`: adding values back reuses the variable that holds
  the line's pooled sigma, so later pixels on that line are clipped against a
  residual instead.

Everything runs one image line at a time, as IRAF does, which avsigclip
needs. The same Python runs as is, or compiled by numba when it is installed
(`harwood[fast]`); HARWOOD_BACKEND=reference forces the plain version.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .jit import backend  # noqa: F401  (re-exported: tasks report it)
from .jit import jit as _jit

MAX_REAL = 3.4028235e38
TOL = 0.001

# combine / reject codes, as in icombine.h
AVERAGE, MEDIAN, SUM = 1, 2, 4
NONE, CCDCLIP, CRREJECT, MINMAX, PCLIP, SIGCLIP, AVSIGCLIP = 1, 2, 3, 4, 5, 6, 7
COMBINES = {"average": AVERAGE, "median": MEDIAN, "lmedian": MEDIAN, "sum": SUM}
REJECTS = {"none": NONE, "ccdclip": CCDCLIP, "crreject": CRREJECT, "minmax": MINMAX,
           "pclip": PCLIP, "sigclip": SIGCLIP, "avsigclip": AVSIGCLIP}




# ------------------------------------------------------------- scaling


@dataclass
class Scaling:
    """What ic_scale works out: per-image factors, and what they switch on."""

    scales: np.ndarray
    zeros: np.ndarray
    weights: np.ndarray
    doscale: bool = False
    dozero: bool = False
    doscale1: bool = False
    dowts: bool = False
    statistics: dict = field(default_factory=dict)


def ic_mode(sorted_values: np.ndarray, zrange: float = 0.7) -> float:
    """ic_mode: the densest bin of 10% of the central `zrange` of the values
    (70% in imcombine, 80% in ccdred's combine)."""
    a = sorted_values
    n = a.size
    if n < 10:
        return float(a[n // 2 - 1]) if n >= 2 else float(a[0])
    i = int(1 + n * (1.0 - zrange) / 2.0)
    j = int(1 + n * (1.0 + zrange) / 2.0)
    z1, z2 = float(a[i - 1]), float(a[j - 1])
    if z1 == z2:
        return z1
    zstep = 0.01 * (z2 - z1)
    zbin = 0.1 * (z2 - z1)
    z1 = z1 - zstep
    k = i
    nmax = 0
    mode = z1
    while True:
        z1 = z1 + zstep
        z2 = z1 + zbin
        while i < j and a[i - 1] < z1:
            i += 1
        while k < j and a[k - 1] < z2:
            k += 1
        if k - i > nmax:
            nmax = k - i
            mode = float(a[(i + k) // 2 - 1])
        if k >= j:
            break
    return mode


def ic_stat(frame: np.ndarray, section=None, lthresh=None, hthresh=None,
            nmax: int = 100000, zrange: float = 0.7):
    """ic_stat: the mode, median and mean of a sample of the image.

    IRAF steps through the section with the smallest stride, 1 to 10, that
    keeps the sample under 100,000 pixels, and ignores pixels outside the
    thresholds.
    """
    region = frame[section] if section is not None else frame
    for step in range(1, 11):
        n = 1
        for length in region.shape:
            n *= max(1, (length - 1) // step + 1)
        if n < nmax:
            break
    sample = region[tuple(slice(None, None, step) for _ in region.shape)].ravel()
    if lthresh is not None or hthresh is not None:
        low = -MAX_REAL if lthresh is None else lthresh
        high = MAX_REAL if hthresh is None else hthresh
        sample = sample[(sample >= low) & (sample <= high)]
    if sample.size < 1:
        raise ValueError("the statistics section contains no pixels")
    ordered = np.sort(sample.astype(np.float32)).astype(np.float64)
    return {"mode": ic_mode(ordered, zrange),
            "median": float(ordered[max(ordered.size // 2 - 1, 0)]),
            "mean": float(ordered.sum() / ordered.size)}


def ic_scale(stack, headers, scale="none", zero="none", weight="none",
             exposure_key="", section=None, lthresh=None, hthresh=None,
             sigscale=0.1, flavor="imcombine") -> Scaling:
    """ic_scale: scales, zeros and weights.

    Two IRAFs. imcombine (images.immatch) refers everything to the first
    image. ccdred's combine -- behind zerocombine, darkcombine and
    flatcombine -- refers scales and zeros to their means instead, so a
    combined dark comes out at the mean exposure rather than the first
    frame's, and its sigma scaling also switches on for a zero offset.
    """
    nimages = stack.shape[0]
    ncombine = np.array([int(h.get("NCOMBINE", 1)) for h in headers], dtype=float)
    exptime = np.array([float(h.get(exposure_key, 0.0)) if exposure_key else 0.0
                        for h in headers])

    stats: dict[int, dict] = {}

    def values(kind):
        if kind == "none":
            return np.full(nimages, np.nan)
        if kind == "exposure":
            return np.maximum(0.001, exptime)
        for index in range(nimages):
            if index not in stats:
                stats[index] = ic_stat(stack[index], section, lthresh, hthresh,
                                       *((10000, 0.8) if flavor == "ccdred"
                                         else (100000, 0.7)))
        return np.array([stats[index][kind] for index in range(nimages)])

    scales, zeros, wts = values(scale), values(zero), values(weight)
    scales = np.where(np.isnan(scales), 1.0, scales)
    zeros = np.where(np.isnan(zeros), 0.0, zeros)
    wts = np.where(np.isnan(wts), 1.0, wts)
    if np.any(scales <= 0):
        scales[:] = 1.0

    ccdred = flavor == "ccdred"
    scales = scales / (scales.mean() if ccdred else scales[0])
    zeros = zeros / scales
    reference = zeros.mean() if ccdred else zeros[0]
    if weight != "none":
        if np.any(wts <= 0 if ccdred else wts < 0):
            wts = ncombine.copy()
        elif zero == "none":
            wts = ncombine * wts
        elif np.any(zeros <= 0):
            wts = ncombine * wts
        else:
            wts = ncombine * wts * reference / zeros
    zeros = zeros - reference
    if np.all(zeros[1:] == zeros[0]):
        zeros[:] = 0.0
    total = wts.sum()
    wts = wts / total if total > 0 else np.ones(nimages)

    doscale = bool(np.any(scales[1:] != scales[0]))
    dozero = bool(np.any(zeros[1:] != zeros[0]))
    dowts = bool(np.any(wts[1:] != wts[0]))
    doscale1 = bool(doscale and sigscale != 0 and np.any(np.abs(scales - 1) > sigscale))
    if ccdred and doscale and sigscale != 0 and not doscale1 and reference > 0:
        doscale1 = bool(np.any(np.abs(zeros / reference) > sigscale))
    return Scaling(scales, zeros, wts, doscale, dozero, doscale1, dowts,
                   {k: v for k, v in stats.items()})


# ------------------------------------------------------- the algorithms
#
# Every function below works on one pixel's buffers: d[0:N] values, ids[0:N]
# image indices (0-based), n of them in use. Indices in comments are IRAF's
# 1-based ones; the code subtracts one where it touches an array.


@_jit
def _sort(d, ids, n):
    """Ascending, carrying the ids along (ic_2sort)."""
    for i in range(1, n):
        value, ident = d[i], ids[i]
        j = i - 1
        while j >= 0 and d[j] > value:
            d[j + 1] = d[j]
            ids[j + 1] = ids[j]
            j -= 1
        d[j + 1] = value
        ids[j + 1] = ident


@_jit
def _maxkeep(nkeep, n):
    if nkeep < 0:
        return max(0, n + nkeep)
    return min(n, nkeep)


@_jit
def _addback_s(n1, nl, nh, maxkeep, n_orig, n2, resid, s):
    """The `while (n1 < maxkeep)` of the median-based algorithms, returning
    also what it leaves in the variable `s` -- which ccdred's mavsigclip
    shares with the line's sigma."""
    while n1 < maxkeep:
        if nl == 1:
            nh = nh + 1
        elif nh == n_orig:
            nl = nl - 1
        else:
            r = resid[nl - 1]
            s = resid[nh + 1]
            if r < s:
                nl = nl - 1
                r = r + TOL
                if s <= r:
                    nh = nh + 1
                if nl > 1:
                    if resid[nl - 1] <= r:
                        nl = nl - 1
            else:
                nh = nh + 1
                s = s + TOL
                if r <= s:
                    nl = nl - 1
                if nh < n2:
                    if resid[nh + 1] <= s:
                        nh = nh + 1
        n1 = nh - nl + 1
    return n1, nl, nh, s


@_jit
def _addback_sorted(n1, nl, nh, maxkeep, n_orig, n2, resid):
    n1, nl, nh, _ = _addback_s(n1, nl, nh, maxkeep, n_orig, n2, resid, 0.0)
    return n1, nl, nh


@_jit
def _compact(d, ids, n1, nl, grow):
    """Move the kept range [nl, nh] to the front, as the m* algorithms do."""
    j = max(nl, n1 + 1)
    for l in range(1, min(n1, nl - 1) + 1):
        d[l - 1] = d[j - 1]
        if grow >= 1.0:
            ident = ids[l - 1]
            ids[l - 1] = ids[j - 1]
            ids[j - 1] = ident
        else:
            ids[l - 1] = ids[j - 1]
        j += 1


@_jit
def _median_of(d, nl, n1):
    n3 = nl + n1 // 2
    if n1 % 2 == 0:
        return (d[n3 - 2] + d[n3 - 1]) / 2.0
    return d[n3 - 1]


@_jit
def ic_msigclip(d, ids, n, nkeep, lsigma, hsigma, doscale1, scales, zeros,
                is_median, grow, blank, resid, w, ccdred):
    n1 = max(0, n)
    maxkeep = _maxkeep(nkeep, n1)
    nl, nh = 1, n1
    med = blank
    n2 = n1
    while True:
        n2 = n1
        if n1 == 0:
            med = blank
        else:
            med = _median_of(d, nl, n1)
        if n1 >= max(3, maxkeep + 1):
            s = 0.0
            if doscale1:
                for j in range(nl, nh + 1):
                    l = ids[j - 1]
                    r = np.sqrt(max(1.0, (med + zeros[l]) / scales[l]))
                    s += ((d[j - 1] - med) / r) ** 2
                    w[j - 1] = r
            else:
                for j in range(nl, nh + 1):
                    s += (d[j - 1] - med) ** 2
            s = np.sqrt(s / (n1 - 1))
            if s > 0.0:
                # ccdred's copy bounds this loop by n2 -- a count, not a position.
                while nl <= (n2 if ccdred else nh):
                    scale = s * w[nl - 1] if doscale1 else s
                    r = (med - d[nl - 1]) / scale
                    if r <= lsigma:
                        break
                    resid[nl] = r
                    n1 -= 1
                    nl += 1
                while nh >= nl:
                    scale = s * w[nh - 1] if doscale1 else s
                    r = (d[nh - 1] - med) / scale
                    if r <= hsigma:
                        break
                    resid[nh] = r
                    n1 -= 1
                    nh -= 1
        if n1 == n2 or n1 < max(3, maxkeep + 1):
            break
    n1, nl, nh = _addback_sorted(n1, nl, nh, maxkeep, max(0, n), n2, resid)
    if n1 > 0 and nl > 1 and ((not is_median) or grow >= 1.0):
        _compact(d, ids, n1, nl, grow)
    return n1, med


@_jit
def _trimmed(d, n1):
    """The average with the lowest and highest left out, and the full sum."""
    low, high = d[0], d[1]
    if low > high:
        low, high = high, low
    total = 0.0
    for j in range(3, n1 + 1):
        d1 = d[j - 1]
        if d1 < low:
            total += low
            low = d1
        elif d1 > high:
            total += high
            high = d1
        else:
            total += d1
    return total / (n1 - 2), total + low + high


@_jit
def _swap_out(d, ids, j, n1):
    """Put the rejected value at position j at the end, the last in its place."""
    d1 = d[j - 1]
    d[j - 1] = d[n1 - 1]
    d[n1 - 1] = d1
    ident = ids[j - 1]
    ids[j - 1] = ids[n1 - 1]
    ids[n1 - 1] = ident


@_jit
def _addback_unsorted(d, ids, n1, maxkeep, n2, resid, total):
    """The add-back of the average-based algorithms: smallest residual first,
    ties (within TOL) all coming back together."""
    nk = maxkeep
    j = n1 + 1
    while j <= nk:
        r = resid[j]
        jj = 0
        for l in range(j + 1, n2 + 1):
            s = resid[l]
            if s < r + TOL:
                if s > r - TOL:
                    jj += 1
                else:
                    jj = 0
                    resid[l] = r
                    r = s
                    d1 = d[j - 1]
                    d[j - 1] = d[l - 1]
                    d[l - 1] = d1
                    ident = ids[j - 1]
                    ids[j - 1] = ids[l - 1]
                    ids[l - 1] = ident
        total += d[j - 1]
        n1 += 1
        nk = max(nk, j + jj)
        j += 1
    return n1, total, r


@_jit
def ic_asigclip(d, ids, n, nkeep, lsigma, hsigma, doscale1, scales, zeros,
                blank, resid, w):
    n1 = max(0, n)
    maxkeep = _maxkeep(nkeep, n1)
    if n1 < max(3, maxkeep):
        return n1
    a, total = _trimmed(d, n1)
    n2 = n1
    while True:
        n2 = n1
        s = 0.0
        if doscale1:
            for j in range(1, n1 + 1):
                l = ids[j - 1]
                r = np.sqrt(max(1.0, (a + zeros[l]) / scales[l]))
                s += ((d[j - 1] - a) / r) ** 2
                w[j - 1] = r
        else:
            for j in range(1, n1 + 1):
                s += (d[j - 1] - a) ** 2
        s = np.sqrt(s / (n1 - 1))
        if s > 0.0:
            j = 1
            wp = 0            # IRAF's weight pointer, which runs on its own
            while j <= n1:
                d1 = d[j - 1]
                weight = w[wp] if doscale1 else 1.0
                r = (d1 - a) / (s * weight)
                if r < -lsigma or r > hsigma:
                    resid[n1] = abs(r)
                    if j < n1:
                        if doscale1:
                            w[wp] = w[n1 - 1]
                        _swap_out(d, ids, j, n1)
                        j -= 1
                        # IRAF: wp is not stepped back with j, so the value
                        # swapped in is judged with the next position's weight.
                    total -= d1
                    n1 -= 1
                j += 1
                wp += 1
        if n1 > 1:
            a = total / n1
        if n1 == n2 or n1 <= max(2, maxkeep):
            break
    if n1 < maxkeep:
        n1, total, _ = _addback_unsorted(d, ids, n1, maxkeep, n2, resid, total)
    return n1


@_jit
def _noise(nm1, nm2, nm3, level):
    return np.sqrt(nm1 + level / nm2 + (level * nm3) ** 2)


@_jit
def ic_accdclip(d, ids, n, nkeep, lsigma, hsigma, doscale1, scales, zeros,
                nm1, nm2, nm3, blank, resid):
    n1 = max(0, n)
    maxkeep = _maxkeep(nkeep, n1)
    if n1 <= max(1, maxkeep):
        return n1
    n2 = n1
    total = 0.0
    while True:
        if n1 == 2:
            total = d[0] + d[1]
            a = total / 2.0
        else:
            a, total = _trimmed(d, n1)
        n2 = n1
        j = 1
        while j <= n1:
            l = ids[j - 1]
            if doscale1:
                level = max(0.0, scales[l] * (a + zeros[l]))
                s = _noise(nm1[l], nm2[l], nm3[l], level) / scales[l]
            else:
                s = _noise(nm1[l], nm2[l], nm3[l], max(0.0, a))
            d1 = d[j - 1]
            r = (d1 - a) / s
            if r < -lsigma or r > hsigma:
                resid[n1] = abs(r)
                if j < n1:
                    _swap_out(d, ids, j, n1)
                    j -= 1
                total -= d1
                n1 -= 1
            j += 1
        if n1 == n2 or n1 < max(2, maxkeep + 1):
            break
    if n1 < maxkeep:
        n1, total, _ = _addback_unsorted(d, ids, n1, maxkeep, n2, resid, total)
    return n1


@_jit
def ic_mccdclip(d, ids, n, nkeep, lsigma, hsigma, doscale1, scales, zeros,
                nm1, nm2, nm3, is_median, grow, blank, resid, ccdred):
    n1 = max(0, n)
    maxkeep = _maxkeep(nkeep, n1)
    nl, nh = 1, n1
    med = blank
    n2 = n1
    while True:
        n2 = n1
        if n1 == 0:
            med = blank
        else:
            med = _median_of(d, nl, n1)
        if n1 >= max(2, maxkeep + 1):
            while nl <= (n2 if ccdred else nh):
                l = ids[nl - 1]
                if doscale1:
                    level = max(0.0, scales[l] * (med + zeros[l]))
                    s = _noise(nm1[l], nm2[l], nm3[l], level) / scales[l]
                else:
                    s = _noise(nm1[l], nm2[l], nm3[l], max(0.0, med))
                r = (med - d[nl - 1]) / s
                if r <= lsigma:
                    break
                resid[nl] = r
                n1 -= 1
                nl += 1
            while nh >= nl:
                l = ids[nh - 1]
                if doscale1:
                    level = max(0.0, scales[l] * (med + zeros[l]))
                    s = _noise(nm1[l], nm2[l], nm3[l], level) / scales[l]
                else:
                    s = _noise(nm1[l], nm2[l], nm3[l], max(0.0, med))
                r = (d[nh - 1] - med) / s
                if r <= hsigma:
                    break
                resid[nh] = r
                n1 -= 1
                nh -= 1
        if n1 == n2 or n1 < max(2, maxkeep + 1):
            break
    n1, nl, nh = _addback_sorted(n1, nl, nh, maxkeep, max(0, n), n2, resid)
    if n1 > 0 and nl > 1 and ((not is_median) or grow >= 1.0):
        _compact(d, ids, n1, nl, grow)
    return n1, med


@_jit
def ic_pclip(d, ids, n, nkeep, lsigma, hsigma, pclip, is_median, grow, blank, resid):
    n1 = max(0, n)
    maxkeep = _maxkeep(nkeep, n1)
    if n1 == 0:
        return 0, blank
    n2 = 1 + n1 // 2
    even = n1 % 2 == 0
    t = -1.0 if pclip < 0 else 1.0
    if pclip < 0:
        if even:
            n3 = max(1, int(np.floor(n2 - 1 + pclip + 0.5)) if n2 - 1 + pclip >= 0
                     else int(np.ceil(n2 - 1 + pclip - 0.5)))
        else:
            n3 = max(1, int(np.floor(n2 + pclip + 0.5)) if n2 + pclip >= 0
                     else int(np.ceil(n2 + pclip - 0.5)))
    else:
        n3 = min(n1, int(np.floor(n2 + pclip + 0.5)))
    if even:
        med = (d[n2 - 2] + d[n2 - 1]) / 2.0
    else:
        med = d[n2 - 1]
    if n1 < max(3, maxkeep + 1):
        return n1, med
    sigma = t * (d[n3 - 1] - med)
    if abs(sigma) <= 1.0e-7 * max(abs(d[n3 - 1]), abs(med), 1e-30):
        return n1, med
    nl = 1
    while nl <= n1:
        r = (med - d[nl - 1]) / sigma
        if r < lsigma:
            break
        resid[nl] = r
        nl += 1
    nh = n1
    while nh >= 1:
        r = (d[nh - 1] - med) / sigma
        if r < hsigma:
            break
        resid[nh] = r
        nh -= 1
    n4 = nh - nl + 1
    n4, nl, nh = _addback_sorted(n4, nl, nh, maxkeep, max(0, n), n2, resid)
    count = n1
    if nl > 1 or nh < n1:
        med = _median_of(d, nl, n4)
        count = n4
    if nl > 1 and ((not is_median) or grow >= 1.0):
        _compact(d, ids, n4, nl, grow)
    return count, med


@_jit
def ic_mm(d, ids, n, flow, fhigh):
    """minmax: drop the nlow lowest and nhigh highest, in pairs first."""
    n1 = max(0, n)
    nlow = int(flow * n1 + 0.001)
    nhigh = int(fhigh * n1 + 0.001)
    npairs = min(nlow, nhigh)
    nlow -= npairs
    nhigh -= npairs
    for _ in range(npairs):
        d1 = d[0]
        dmax, dmin, jmax, jmin = d1, d1, 1, 1
        d2 = d1
        for j in range(2, n1 + 1):
            d2 = d1
            d1 = d[j - 1]
            if d1 > dmax:
                dmax, jmax = d1, j
            elif d1 < dmin:
                dmin, jmin = d1, j
        j = n1 - 1
        if jmax < j:
            if jmin != j:
                d[jmax - 1] = d2
                ident = ids[jmax - 1]
                ids[jmax - 1] = ids[j - 1]
                ids[j - 1] = ident
            else:
                d[jmax - 1] = d1
                ident = ids[jmax - 1]
                ids[jmax - 1] = ids[n1 - 1]
                ids[n1 - 1] = ident
        if jmin < j:
            if jmax != n1:
                d[jmin - 1] = d1
                ident = ids[jmin - 1]
                ids[jmin - 1] = ids[n1 - 1]
                ids[n1 - 1] = ident
            else:
                d[jmin - 1] = d2
                ident = ids[jmin - 1]
                ids[jmin - 1] = ids[j - 1]
                ids[j - 1] = ident
        n1 -= 2
    for _ in range(nlow):
        d1 = d[0]
        dmin, jmin = d1, 1
        for j in range(2, n1 + 1):
            d1 = d[j - 1]
            if d1 < dmin:
                dmin, jmin = d1, j
        if jmin < n1:
            d[jmin - 1] = d1
            ident = ids[jmin - 1]
            ids[jmin - 1] = ids[n1 - 1]
            ids[n1 - 1] = ident
        n1 -= 1
    for _ in range(nhigh):
        d1 = d[0]
        dmax, jmax = d1, 1
        for j in range(2, n1 + 1):
            d1 = d[j - 1]
            if d1 > dmax:
                dmax, jmax = d1, j
        if jmax < n1:
            d[jmax - 1] = d1
            ident = ids[jmax - 1]
            ids[jmax - 1] = ids[n1 - 1]
            ids[n1 - 1] = ident
        n1 -= 1
    return n1


@_jit
def _select(values, k):
    """The k-th smallest (1-based), the value IRAF's quickselect lands on."""
    ordered = np.sort(values)
    return ordered[k - 1]


@_jit
def ic_median(d, n1, mclip, medavg, blank):
    if n1 <= 0:
        return blank
    if mclip:
        j1 = n1 // 2 + 1
        if n1 % 2 == 0 and (medavg or n1 > 2):
            return (d[j1 - 1] + d[n1 // 2 - 1]) / 2.0
        return d[j1 - 1]
    if n1 > 3:
        k = max(1, min(n1, (n1 + 1) // 2))
        median = _select(d[:n1], k)
        if n1 % 2 == 0 and (medavg or n1 > 2):
            median = (median + _select(d[:n1], k + 1)) / 2.0
        return median
    if n1 == 3:
        v1, v2, v3 = d[0], d[1], d[2]
        if v1 < v2:
            if v2 < v3:
                return v2
            if v1 < v3:
                return v3
            return v1
        if v2 > v3:
            return v2
        if v1 > v3:
            return v3
        return v1
    if n1 == 2:
        return (d[0] + d[1]) / 2.0 if medavg else min(d[0], d[1])
    return d[0]


@_jit
def ic_average(d, ids, n1, wts, dowts, average, blank):
    if n1 <= 0:
        return blank
    if dowts and average:
        total = 0.0
        sumwt = 0.0
        for j in range(n1):
            total += d[j] * wts[ids[j]]
            sumwt += wts[ids[j]]
        if sumwt > 0:
            return total / sumwt
        total = 0.0
        for j in range(n1):
            total += d[j]
        return total / n1
    total = 0.0
    for j in range(n1):
        total += d[j]
    return total / n1 if average else total


@_jit
def ic_sigma(d, ids, n1, wts, dowts, centre, blank):
    if n1 <= 0:
        return blank
    if dowts:
        sigcor = n1 / (n1 - 1.0) if n1 > 1 else 1.0
        total = 0.0
        sumwt = 0.0
        for j in range(n1):
            wt = wts[ids[j]]
            total += (d[j] - centre) ** 2 * wt
            sumwt += wt
        if sumwt > 0:
            return np.sqrt(total / sumwt * sigcor)
        total = 0.0
        for j in range(n1):
            total += (d[j] - centre) ** 2
        return np.sqrt(total / n1 * sigcor)
    sigcor = 1.0 / (n1 - 1.0) if n1 > 1 else 1.0
    total = 0.0
    for j in range(n1):
        total += (d[j] - centre) ** 2
    return np.sqrt(total * sigcor)


# ------------------------------------------------------------- one line


@_jit
def _line(data, valid, out, sigma, used, rejected, combine, medavg, reject, mclip,
          flow, fhigh, nkeep, lsigma, hsigma, pclip, grow, blank, scales, zeros,
          wts, dowts, doscale1, nm1, nm2, nm3, ccdred):
    """One image line: gather, reject, combine, sigma -- icomb.gx's loop body.

    data[nimages, npts] is the line already scaled; valid marks the pixels
    the thresholds and masks leave in. Writes out, sigma, how many values
    were used, and which (image, pixel) the rejection took.
    """
    nimages, npts = data.shape
    d = np.empty(nimages)
    ids = np.empty(nimages, dtype=np.int64)
    resid = np.zeros(nimages + 2)
    w = np.ones(nimages)
    is_median = combine == MEDIAN
    maxkeep_all = _maxkeep(nkeep, nimages)

    # avsigclip pools sigma over the line, so it runs as a line.
    if reject == AVSIGCLIP and nimages >= max(3, maxkeep_all + 1):
        _avsigclip_line(data, valid, out, sigma, used, rejected, combine, medavg,
                        mclip, nkeep, lsigma, hsigma, grow, blank, scales, zeros,
                        wts, dowts, doscale1, ccdred)
        return

    for i in range(npts):
        n = 0
        for k in range(nimages):
            if valid[k, i]:
                d[n] = data[k, i]
                ids[n] = k
                n += 1
        if mclip:
            _sort(d, ids, n)
        n_in = n
        med = blank
        docombine = True
        if reject == MINMAX:
            n = ic_mm(d, ids, n, flow, fhigh)
        elif reject == SIGCLIP and nimages >= max(3, maxkeep_all + 1):
            if mclip:
                n, med = ic_msigclip(d, ids, n, nkeep, lsigma, hsigma, doscale1,
                                     scales, zeros, is_median, grow, blank, resid, w,
                                     ccdred)
                docombine = not is_median
            else:
                n = ic_asigclip(d, ids, n, nkeep, lsigma, hsigma, doscale1, scales,
                                zeros, blank, resid, w)
        elif (reject == CCDCLIP or reject == CRREJECT) and \
                nimages >= max(2, maxkeep_all + 1):
            low = MAX_REAL if reject == CRREJECT else lsigma
            if mclip:
                n, med = ic_mccdclip(d, ids, n, nkeep, low, hsigma, doscale1, scales,
                                     zeros, nm1, nm2, nm3, is_median, grow, blank,
                                     resid, ccdred)
                docombine = not is_median
            else:
                n = ic_accdclip(d, ids, n, nkeep, low, hsigma, doscale1, scales,
                                zeros, nm1, nm2, nm3, blank, resid)
        elif reject == PCLIP and nimages >= max(3, maxkeep_all + 1):
            n, med = ic_pclip(d, ids, n, nkeep, lsigma, hsigma, pclip, is_median,
                              grow, blank, resid)
            docombine = not is_median

        if docombine:
            if is_median:
                value = ic_median(d, n, mclip, medavg, blank)
            else:
                value = ic_average(d, ids, n, wts, dowts and not is_median,
                                   combine == AVERAGE, blank)
        else:
            value = med
        out[i] = value
        sigma[i] = ic_sigma(d, ids, n, wts, dowts and not is_median, value, blank)
        used[i] = max(0, n)
        for j in range(max(0, n), n_in):
            rejected[ids[j], i] = True


@_jit
def _avsigclip_line(data, valid, out, sigma, used, rejected, combine, medavg, mclip,
                    nkeep, lsigma, hsigma, grow, blank, scales, zeros, wts, dowts,
                    doscale1, ccdred):
    nimages, npts = data.shape
    D = np.empty((npts, nimages))
    IDS = np.empty((npts, nimages), dtype=np.int64)
    N = np.zeros(npts, dtype=np.int64)
    NIN = np.zeros(npts, dtype=np.int64)
    for i in range(npts):
        n = 0
        for k in range(nimages):
            if valid[k, i]:
                D[i, n] = data[k, i]
                IDS[i, n] = k
                n += 1
        if mclip:
            _sort(D[i], IDS[i], n)
        N[i] = n
        NIN[i] = n
    resid = np.zeros(nimages + 2)
    is_median = combine == MEDIAN
    centre = np.full(npts, blank)
    docombine = True

    if mclip:
        # ic_mavsigclip
        s = 0.0
        n2 = 0
        for i in range(npts):
            n1 = N[i]
            d = D[i]
            if n1 < 3:
                if n1 == 0:
                    centre[i] = blank
                elif n1 == 1:
                    centre[i] = d[0]
                else:
                    centre[i] = (d[0] + d[1]) / 2.0
                continue
            med = _median_of(d, 1, n1)
            for j in range(n1):
                if doscale1:
                    l = IDS[i, j]
                    s1 = max(1.0, (med + zeros[l]) / scales[l])
                else:
                    s1 = max(1.0, med)
                s += (d[j] - med) ** 2 / s1
            n2 += n1
            centre[i] = med
        if n2 > 1 or ccdred:
            # ccdred's copy keeps the line's sigma in `s`, which the add-back
            # below reuses -- the same slip as aavsigclip, here too.
            sig = np.sqrt(s / (n2 - 1)) if n2 > 1 else s
            for i in range(npts):
                n1 = N[i]
                d = D[i]
                ids = IDS[i]
                maxkeep = _maxkeep(nkeep, n1)
                if n1 < max(3, maxkeep + 1):
                    continue
                nl, nh = 1, n1
                med = centre[i]
                n2 = n1
                while True:
                    n2 = n1
                    if n1 >= max(3, maxkeep + 1) and sig > 0.0:
                        while nl <= (n2 if ccdred else nh):
                            if doscale1:
                                l = ids[nl - 1]
                                s1 = sig * np.sqrt(max(1.0, (med + zeros[l]) / scales[l]))
                            else:
                                s1 = sig * np.sqrt(max(1.0, med))
                            r = (med - d[nl - 1]) / s1
                            if r <= lsigma:
                                break
                            resid[nl] = r
                            n1 -= 1
                            nl += 1
                        while nh >= nl:
                            if doscale1:
                                l = ids[nh - 1]
                                s1 = sig * np.sqrt(max(1.0, (med + zeros[l]) / scales[l]))
                            else:
                                s1 = sig * np.sqrt(max(1.0, med))
                            r = (d[nh - 1] - med) / s1
                            if r <= hsigma:
                                break
                            resid[nh] = r
                            n1 -= 1
                            nh -= 1
                        if n1 < n2:
                            med = _median_of(d, nl, n1) if n1 > 0 else blank
                    if n1 == n2 or n1 < max(3, maxkeep + 1):
                        break
                while n1 < maxkeep:
                    before = n1
                    n1, nl, nh, shared = _addback_s(n1, nl, nh, n1 + 1, NIN[i], n2,
                                                    resid, sig)
                    if ccdred:
                        sig = shared
                    if n1 < n2:
                        med = _median_of(d, nl, n1) if n1 > 0 else blank
                    if n1 == before:
                        break
                N[i] = n1
                if n1 > 0 and nl > 1 and ((not is_median) or grow >= 1.0):
                    _compact(d, ids, n1, nl, grow)
                centre[i] = med
        docombine = not is_median
    else:
        # ic_aavsigclip
        sums = np.zeros(npts)
        s = 0.0
        n2 = 0
        for i in range(npts):
            n1 = N[i]
            d = D[i]
            if n1 < 3:
                continue
            a, total = _trimmed(d, n1)
            for j in range(n1):
                if doscale1:
                    l = IDS[i, j]
                    s1 = max(1.0, (a + zeros[l]) / scales[l])
                else:
                    s1 = max(1.0, a)
                s += (d[j] - a) ** 2 / s1
            n2 += n1
            centre[i] = a
            sums[i] = total
        if n2 > 1:
            s = np.sqrt(s / (n2 - 1))
        for i in range(npts):
            n1 = N[i]
            d = D[i]
            ids = IDS[i]
            maxkeep = _maxkeep(nkeep, n1)
            if n1 <= max(2, maxkeep):
                if n1 > 0:
                    total = 0.0
                    for j in range(n1):
                        total += d[j]
                    centre[i] = total / n1
                continue
            a = centre[i]
            total = sums[i]
            n2 = n1
            while True:
                n2 = n1
                if s > 0.0:
                    j = 1
                    while j <= n1:
                        l = ids[j - 1]
                        if doscale1:
                            s1 = s * np.sqrt(max(1.0, (a + zeros[l]) / scales[l]))
                        else:
                            s1 = s * np.sqrt(max(1.0, a))
                        d1 = d[j - 1]
                        r = (d1 - a) / s1
                        if r < -lsigma or r > hsigma:
                            resid[n1] = abs(r)
                            if j < n1:
                                _swap_out(d, ids, j, n1)
                                j -= 1
                            total -= d1
                            n1 -= 1
                        j += 1
                if n1 > 1:
                    a = total / n1
                if n1 == n2 or n1 <= max(2, maxkeep):
                    break
            if n1 < maxkeep:
                n1, total, last = _addback_unsorted(d, ids, n1, maxkeep, n2, resid,
                                                    total)
                # IRAF reuses `s` in this loop, so the line's pooled sigma is
                # now the last residual compared, for every later pixel.
                s = last
                if n1 > 1:
                    a = total / n1
            N[i] = n1
            centre[i] = a

    for i in range(npts):
        n = N[i]
        d = D[i]
        ids = IDS[i]
        if docombine:
            if is_median:
                value = ic_median(d, n, mclip, medavg, blank)
            else:
                value = ic_average(d, ids, n, wts, dowts and not is_median,
                                   combine == AVERAGE, blank)
        else:
            value = centre[i]
        out[i] = value
        sigma[i] = ic_sigma(d, ids, n, wts, dowts and not is_median, value, blank)
        used[i] = max(0, n)
        for j in range(max(0, n), NIN[i]):
            rejected[ids[j], i] = True


# ------------------------------------------------------------- the image


def combine(stack: np.ndarray, valid: np.ndarray, scaling: Scaling, *,
            combine: str = "average", reject: str = "none", mclip: bool = True,
            nlow: float = 1, nhigh: float = 1, nkeep: int = 1, lsigma: float = 3.0,
            hsigma: float = 3.0, pclip: float = -0.5, grow: float = 0.0,
            blank: float = 0.0, rdnoise=0.0, gain=1.0, snoise=0.0,
            flavor: str = "imcombine"):
    """imcombine over a stack[nimages, ny, nx]: out, sigma, used, rejected.

    `valid` is False where a threshold or a mask removes a pixel. rdnoise,
    gain and snoise may be one number or one per image.
    """
    nimages = stack.shape[0]
    code = COMBINES[combine]
    medavg = combine != "lmedian"
    rcode = REJECTS[reject]
    if rcode == MINMAX:
        mclip = False
    elif rcode == PCLIP:
        mclip = True
    elif rcode == NONE:
        mclip = False

    flow = float(nlow) / nimages if nlow >= 1 else float(nlow)
    fhigh = float(nhigh) / nimages if nhigh >= 1 else float(nhigh)
    if rcode == MINMAX:
        if int(flow * nimages) + int(fhigh * nimages) == 0:
            rcode = NONE
        elif int(flow * nimages) + int(fhigh * nimages) >= nimages:
            from . import advice

            advice.fail(f"minmax would reject {nlow} low and {nhigh} high of "
                        f"{nimages} images, leaving none",
                        "lower --nlow and --nhigh, or combine more images")
    if rcode == PCLIP:
        half = int(nimages / 2.0)
        p = pclip * half if abs(pclip) < 1.0 else pclip
        pclip = float(min(-1, max(-half, int(p))) if p < 0 else max(1, min(half, int(p))))

    nm1 = np.empty(nimages)
    nm2 = np.empty(nimages)
    nm3 = np.empty(nimages)
    rd = np.broadcast_to(np.asarray(rdnoise, dtype=float), (nimages,))
    gn = np.broadcast_to(np.asarray(gain, dtype=float), (nimages,))
    sn = np.broadcast_to(np.asarray(snoise, dtype=float), (nimages,))
    for k in range(nimages):
        nm2[k] = gn[k]
        nm1[k] = max((rd[k] / gn[k]) ** 2, 1e4 / MAX_REAL)
        nm3[k] = sn[k]

    scaled = (stack / scaling.scales[:, None, None]
              - scaling.zeros[:, None, None]).astype(np.float64)
    dowts = scaling.dowts and code in (AVERAGE, SUM)
    lsig = MAX_REAL if lsigma is None else float(lsigma)
    hsig = MAX_REAL if hsigma is None else float(hsigma)

    def run(valid_now, rcode_now, first):
        ny, nx = stack.shape[1:]
        out = np.empty((ny, nx))
        sig = np.empty((ny, nx))
        used = np.empty((ny, nx), dtype=np.int64)
        rej = np.zeros((nimages, ny, nx), dtype=bool)
        line_out = np.empty(nx)
        line_sig = np.empty(nx)
        line_used = np.empty(nx, dtype=np.int64)
        for y in range(ny):
            line_rej = np.zeros((nimages, nx), dtype=bool)
            _line(np.ascontiguousarray(scaled[:, y, :]),
                  np.ascontiguousarray(valid_now[:, y, :]), line_out, line_sig,
                  line_used, line_rej, code, medavg, rcode_now, mclip, flow, fhigh,
                  int(nkeep), lsig, hsig, float(pclip), float(grow), float(blank),
                  scaling.scales.astype(np.float64), scaling.zeros.astype(np.float64),
                  scaling.weights.astype(np.float64), dowts, scaling.doscale1,
                  nm1, nm2, nm3, flavor == "ccdred")
            out[y], sig[y], used[y] = line_out, line_sig, line_used
            rej[:, y, :] = line_rej
        return out, sig, used, rej

    out, sig, used, rejected = run(valid, rcode, True)
    if grow >= 1.0 and rejected.any():
        grown = _grow(rejected, grow)
        out2, sig2, used2 = _second_pass(scaled, valid, grown, code, medavg, mclip,
                                         scaling, dowts, blank)
        keep_first = (used2 == 0) & (nkeep > 0)
        out = np.where(keep_first, out, out2)
        sig, used = sig2, used2
        rejected = rejected | (grown & valid)
    return out, sig, used, rejected


def _grow(rejected: np.ndarray, grow: float) -> np.ndarray:
    """ic_grow: every rejected pixel takes its neighbours within `grow`, in
    the same image, along the line and across lines."""
    igrow = int(grow)
    grown = rejected.copy()
    nimages, ny, nx = rejected.shape
    for dy in range(-igrow, igrow + 1):
        for dx in range(-igrow, igrow + 1):
            if dx * dx > grow * grow - dy * dy:
                continue
            shifted = np.zeros_like(rejected)
            ys = slice(max(0, dy), ny + min(0, dy))
            yd = slice(max(0, -dy), ny + min(0, -dy))
            xs = slice(max(0, dx), nx + min(0, dx))
            xd = slice(max(0, -dx), nx + min(0, -dx))
            shifted[:, yd, xd] = rejected[:, ys, xs]
            grown |= shifted
    return grown


@_jit
def _second_line(data, valid, grown, out, sigma, used, code, medavg, mclip, wts,
                 dowts, blank):
    """icomb.gx's second loop, for one line: re-read, remove the grown pixels
    the way ic_grow removes them (the last value into the gap, image by
    image), and combine without rejecting again."""
    nimages, npts = data.shape
    d = np.empty(nimages)
    ids = np.empty(nimages, dtype=np.int64)
    for i in range(npts):
        n = 0
        for k in range(nimages):
            if valid[k, i]:
                d[n] = data[k, i]
                ids[n] = k
                n += 1
        if mclip:
            _sort(d, ids, n)
        for k in range(nimages):
            if not grown[k, i]:
                continue
            for j in range(n):
                if ids[j] == k:
                    if j < n - 1:
                        d[j] = d[n - 1]
                        ids[j] = ids[n - 1]
                    n -= 1
                    break
        if code == MEDIAN:
            value = ic_median(d, n, mclip, medavg, blank)
        else:
            value = ic_average(d, ids, n, wts, dowts, code == AVERAGE, blank)
        out[i] = value
        sigma[i] = ic_sigma(d, ids, n, wts, dowts, value, blank)
        used[i] = n


def _second_pass(scaled, valid, grown, code, medavg, mclip, scaling, dowts, blank):
    nimages, ny, nx = scaled.shape
    out = np.empty((ny, nx))
    sig = np.empty((ny, nx))
    used = np.empty((ny, nx), dtype=np.int64)
    wts = scaling.weights.astype(np.float64)
    line_out, line_sig = np.empty(nx), np.empty(nx)
    line_used = np.empty(nx, dtype=np.int64)
    for y in range(ny):
        _second_line(np.ascontiguousarray(scaled[:, y, :]),
                     np.ascontiguousarray(valid[:, y, :]),
                     np.ascontiguousarray(grown[:, y, :]), line_out, line_sig,
                     line_used, code, medavg, mclip, wts, dowts, float(blank))
        out[y], sig[y], used[y] = line_out, line_sig, line_used
    return out, sig, used
