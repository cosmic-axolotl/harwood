"""IRAF's sky fitting (apphot fitskypars), every algorithm, ported.

From iraf-community 2.18 noao/digiphot/apphot/fitsky (apfitsky, apskybuf,
apmean, apmedian, apmode, apcentroid, apgauss, aplgsky, apcrosscor,
aprgrow, apgrowhist, aphgmsub) and aputil (apclip, apmeds, apmoments,
apvectors, apbsmooth, apmapr, aptopt).

The arithmetic is the arithmetic of IRAF as it is built (SPP -> Fortran ->
f2c -R -> C): REAL variables are single precision, the moment sums are
double, and an expression is promoted to double from a call to exp, sqrt
or pow on, rounded once when it is stored. The histogram algorithms move a
bin edge with the last bit of hmin or dh, so this is what makes the sky the
sky IRAF prints.

Arrays here are 1-based like IRAF's: element 0 is padding. `index` lists
the pixels in the order an algorithm walks them -- raster order, or sorted
by value for median, mode and the percentile clips -- and `coords` holds
each pixel's place in the annulus' bounding box, i + nx (j - 1), which is
what region growing (rgrow) uses to find a rejected pixel's neighbours.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math

import numpy as np

from . import apphot, iraffmt, nlfit
from .apphot import (
    SKY_BADPARAMS,
    SKY_FLAT_HIST,
    SKY_NOCONVERGE,
    SKY_NOHISTOGRAM,
    SKY_NOSKYAREA,
    SKY_NSKY_TOO_SMALL,
    SKY_OK,
    SKY_SINGULAR,
    DataPars,
    Sky,
    SkyPars,
    f32,
)

INDEFR = f32(1.6e38)
MAX_REAL = f32(9.9e36)


def _nint(value) -> int:
    """f2c's i_nint: round half away from zero, in double."""
    v = float(value)
    return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(0.5 - v))


def _ssum(a: np.ndarray) -> np.float32:
    """asumr: in order, in single precision."""
    a = np.asarray(a, dtype=f32)
    return np.cumsum(a, dtype=f32)[-1] if a.size else f32(0.0)


def _dsum(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    return float(np.cumsum(a)[-1]) if a.size else 0.0


def _undef(v) -> float:
    return float("nan") if v == INDEFR else float(v)


class _Sums:
    """The double-precision moment sums about sky_zero."""

    def __init__(self, values: np.ndarray, zero):
        d = (values - zero).astype(f32).astype(np.float64)
        self.s = [_dsum(d), _dsum(d * d), _dsum(d * d * d)]
        self.zero = zero

    def remove(self, value) -> None:
        d = float(f32(value - self.zero))
        self.s[0] -= d
        self.s[1] -= d * d
        self.s[2] -= d * (d * d)

    def moments(self, n: int):
        """apmoments: mean, sigma and skew (single precision)."""
        s1, s2, s3 = self.s
        dmean = s1 / n
        dsigma = s2 / n - dmean * dmean
        if dsigma <= 0.0:
            sigma = skew = f32(0.0)
        else:
            dskew = s3 / n - dmean * 3.0 * dsigma - dmean * (dmean * dmean)
            sigma = f32(math.sqrt(dsigma))
            if dskew < 0:
                skew = f32(-(abs(dskew) ** (1.0 / 3.0)))
            elif dskew > 0:
                skew = f32(dskew ** (1.0 / 3.0))
            else:
                skew = f32(0.0)
        return f32(float(self.zero) + dmean), sigma, skew


class _Annulus:
    """The sky pixels as apskybuf leaves them."""

    def __init__(self, values, coords, snx, sny):
        self.pix = np.concatenate(([f32(0)], np.asarray(values, dtype=f32)))
        self.coords = np.concatenate(([0], np.asarray(coords, dtype=np.int64)))
        self.wgt = np.ones(self.pix.size, dtype=f32)
        self.n = self.pix.size - 1
        self.snx, self.sny = snx, sny


def sky_buffer(data: np.ndarray, x: float, y: float, annulus: float, dannulus: float,
               dpars: DataPars) -> tuple[_Annulus | None, int, int]:
    """apskybuf: the annulus pixels in raster order with their coordinates,
    the number of bad ones, and an error code."""
    rin = f32(f32(annulus) * f32(dpars.scale))
    dann = f32(f32(dannulus) * f32(dpars.scale))
    if dann <= 0:
        return None, 0, SKY_NOSKYAREA
    rout = f32(rin + dann)
    if rout <= rin:
        return None, 0, apphot.SKY_OUTOFBOUNDS
    ny, nx = data.shape
    wx, wy = f32(x), f32(y)
    if (f32(wx + rout) < 1.0 or f32(wx - rout) > nx or f32(wy + rout) < 1.0
            or f32(wy - rout) > ny):
        return None, 0, apphot.SKY_OUTOFBOUNDS
    c1 = int(max(f32(1.0), min(f32(nx), f32(wx - rout))) + f32(0.5))
    c2 = int(min(f32(nx), max(f32(1.0), f32(wx + rout))) + f32(0.5))
    l1 = int(max(f32(1.0), min(f32(ny), f32(wy - rout))) + f32(0.5))
    l2 = int(min(f32(ny), max(f32(1.0), f32(wy + rout))) + f32(0.5))
    snx, sny = c2 - c1 + 1, l2 - l1 + 1
    rin2, rout2 = f32(rin * rin), f32(rout * rout)
    j, i = np.mgrid[l1:l2 + 1, c1:c2 + 1]
    rj2 = (f32(wy) - j.astype(f32)) ** 2
    r2 = ((f32(wx) - i.astype(f32)) ** 2 + rj2).astype(f32)
    ring = (r2 > rin2) & (r2 <= rout2)
    values = data[l1 - 1:l2, c1 - 1:c2].astype(f32)[ring]
    coords = ((i - c1 + 1) + snx * (j - l1))[ring]
    bad = ~np.isfinite(values)                # blank pixels: bad, as datamin's are
    if dpars.datamin is not None or dpars.datamax is not None:
        low = -MAX_REAL if dpars.datamin is None else f32(dpars.datamin)
        high = MAX_REAL if dpars.datamax is None else f32(dpars.datamax)
        bad |= (values < low) | (values > high)
    nbad = int(bad.sum())
    values, coords = values[~bad], coords[~bad]
    if values.size == 0:
        return None, nbad, (SKY_NSKY_TOO_SMALL if nbad else apphot.SKY_OUTOFBOUNDS)
    annulus = _Annulus(values, coords, snx, sny)
    # where the star is in the subraster, for radplot's radii
    annulus.sxc, annulus.syc = float(wx) - c1 + 1, float(wy) - l1 + 1
    return annulus, nbad, SKY_OK


# ------------------------------------------------------------ helpers


def _ialim(a: _Annulus, index, n):
    v = a.pix[index[1:n + 1]]
    return v.min(), v.max()


def _zero(a: _Annulus, index, n):
    """ap_asumr / n: the zero point of the moment sums."""
    total = f32(_dsum(a.pix[index[1:n + 1]].astype(np.float64)))
    return f32(total / f32(n))


def _medr(a: _Annulus, index, n):
    """apmedr -- which, for three pixels or fewer, looks at the first ones
    of the annulus rather than the ones it was given."""
    p = a.pix
    if n in (1, 2):
        return p[1]
    if n == 3:
        if p[1] < p[2]:
            if p[2] < p[3]:
                return p[2]
            return p[3] if p[1] < p[3] else p[1]
        if p[2] > p[3]:
            return p[2]
        return p[1] if p[1] < p[3] else p[3]
    return np.sort(p[index[1:n + 1]])[(n + 1) // 2 - 1]


def _smed(a: _Annulus, index, n, medcut):
    med = (n + 1) // 2
    lo = max(1, med - medcut)
    hi = min(n, med + medcut + (0 if med % 2 == 1 else 1))
    return f32(_ssum(a.pix[index[lo:hi + 1]]) / f32(hi - lo + 1))


def _imed(a: _Annulus, index, lo, hi, nmed):
    npts, med = 0, lo
    while med <= hi and npts < nmed:
        if a.wgt[index[med]] > 0:
            npts += 1
        med += 1
    return 0 if npts == 0 else med


def _wsmed(a: _Annulus, index, npix, med, medcut):
    total = a.pix[index[med]]
    maxmed = 2 * medcut + (1 if med % 2 == 1 else 2)
    nmed = 1
    j = med - 1
    while j >= 1 and nmed < medcut + 1:
        if a.wgt[index[j]] > 0:
            total = f32(total + a.pix[index[j]])
            nmed += 1
        j -= 1
    j = med + 1
    while j <= npix and nmed < maxmed:
        if a.wgt[index[j]] > 0:
            total = f32(total + a.pix[index[j]])
            nmed += 1
        j += 1
    return f32(total / f32(nmed))


def _grow_window(a: _Annulus, index: int, rgrow):
    ixc = int(a.coords[index] % a.snx)
    if ixc == 0:
        ixc = a.snx
    iyc = (int(a.coords[index]) - ixc) // a.snx + 1
    ymin = max(1, int(f32(iyc - rgrow)))
    ymax = min(a.sny, int(f32(iyc + rgrow)))
    xmin = max(1, int(f32(ixc - rgrow)))
    xmax = min(a.snx, int(f32(ixc + rgrow)))
    return ixc, iyc, xmin, xmax, ymin, ymax


def _grow(a: _Annulus, nskypix: int, index: int, rgrow, sums: _Sums,
          hist=None, histogram=False) -> int:
    """ap_grow_regions (histogram=False) and ap_grow_hist2: reject every
    pixel within rgrow of pixel `index`, not rejected already."""
    ixc, iyc, xmin, xmax, ymin, ymax = _grow_window(a, index, rgrow)
    if ymin <= iyc:
        cstart = min(nskypix, max(1, index - int(rgrow) + a.snx * (ymin - iyc)))
    else:
        cstart = index
    rgrow2 = f32(rgrow * rgrow)
    nreject = 0
    for j in range(ymin, ymax + 1):
        d = f32(rgrow2 - f32((j - iyc) * (j - iyc)))
        if histogram:
            d = f32(math.sqrt(d)) if d >= 0 else f32("nan")
        else:
            d = f32(0.0) if d <= 0 else f32(math.sqrt(d))
        lo = max(xmin, int(f32(ixc - d))) if np.isfinite(d) else xmin
        hi = min(xmax, int(f32(ixc + d))) if np.isfinite(d) else xmin - 1
        for k in range(lo, hi + 1):
            c = k + (j - 1) * a.snx
            while a.coords[cstart] < c and cstart < nskypix:
                cstart += 1
            r2 = f32((k - ixc) ** 2 + (j - iyc) ** 2)
            if r2 <= rgrow2 and c == a.coords[cstart] and a.wgt[cstart] > 0:
                nreject += 1
                a.wgt[cstart] = 0
                sums.remove(a.pix[cstart])
                if histogram:
                    hgm, z1, z2 = hist
                    v = a.pix[cstart]
                    if z1 <= v <= z2:
                        dh = f32(f32(hgm.size - 1) / f32(z2 - z1))
                        hgm[int(f32(f32(v - z1) * dh))] -= 1
    return nreject


def _clamp(value, dmin, dmax):
    return max(dmin, min(value, dmax))


# ------------------------------------------- mean, median and mode


def _mean(a, index, n, lo, hi, rgrow, maxiter):
    """ap_mean."""
    if n <= 0:
        return Sky(ier=SKY_NOSKYAREA)
    zero = _zero(a, index, n)
    sums = _Sums(a.pix[index[1:n + 1]], zero)
    mean, sigma, skew = sums.moments(n)
    dmin, dmax = _ialim(a, index, n)
    mean = _clamp(mean, dmin, dmax)
    nsky, nreject_total = n, 0
    if maxiter < 1 or (lo is None and hi is None) or sigma <= 0:
        return Sky(float(mean), float(sigma), float(skew), nsky, 0, SKY_OK)
    for i in range(1, maxiter + 1):
        if lo is None:
            locut = max(-MAX_REAL, dmin)
        elif i == 1:
            locut = f32(mean - min(f32(mean - dmin), f32(dmax - mean), f32(lo * sigma)))
        else:
            locut = f32(mean - f32(lo * sigma))
        if hi is None:
            hicut = min(MAX_REAL, dmax)
        elif i == 1:
            hicut = f32(mean + min(f32(dmax - mean), f32(mean - dmin), f32(hi * sigma)))
        else:
            hicut = f32(mean + f32(hi * sigma))
        nreject = 0
        for j in range(1, n + 1):
            p = index[j]
            if a.wgt[p] <= 0:
                continue
            if a.pix[p] < locut or a.pix[p] > hicut:
                if rgrow > 0:
                    # IRAF assigns rather than adds: only the last region
                    # grown is counted.
                    nreject = _grow(a, n, p, rgrow, sums)
                else:
                    sums.remove(a.pix[p])
                    a.wgt[p] = 0
                    nreject += 1
        if nreject <= 0:
            break
        nreject_total += nreject
        nsky = n - nreject_total
        if nsky <= 0:
            break
        mean, sigma, skew = sums.moments(nsky)
        if sigma <= 0:
            break
        mean = _clamp(mean, dmin, dmax)
    if nsky == 0 or nreject_total == n:
        return Sky(nsky=0, nreject=n, ier=SKY_NSKY_TOO_SMALL)
    return Sky(float(mean), float(sigma), float(skew), nsky, nreject_total, SKY_OK)


def _median_mode(a, index, n, lo, hi, rgrow, maxiter, use_mode: bool):
    """ap_median and ap_mode."""
    if n <= 0:
        return Sky(ier=SKY_NOSKYAREA)
    dmin, dmax = _ialim(a, index, n)
    zero = _zero(a, index, n)
    medcut = _nint(f32(f32(0.025) * f32(n)))
    med_value = _clamp(_smed(a, index, n, medcut), dmin, dmax)
    sums = _Sums(a.pix[index[1:n + 1]], zero)
    mean, sigma, skew = sums.moments(n)
    mean = _clamp(mean, dmin, dmax)

    def mode_of(med_value, mean):
        value = mean if mean < med_value else f32(f32(f32(3.0) * med_value)
                                                  - f32(f32(2.0) * mean))
        return _clamp(value, dmin, dmax)

    value = mode_of(med_value, mean) if use_mode else med_value
    nsky, nreject = n, 0
    if maxiter < 1 or (lo is None and hi is None) or sigma <= 0:
        return Sky(float(value), float(sigma), float(skew), nsky, 0, SKY_OK)
    ilo, ihi = 1, n
    for i in range(1, maxiter + 1):
        centre = med_value if (i == 1 or not use_mode) else value
        if lo is None:
            locut = max(-MAX_REAL, dmin)
        elif i == 1:
            locut = f32(centre - min(f32(centre - dmin), f32(dmax - centre),
                                     f32(lo * sigma)))
        else:
            locut = f32(centre - f32(lo * sigma))
        if hi is None:
            hicut = min(MAX_REAL, dmax)
        elif i == 1:
            spread = ((f32(centre - dmin), f32(dmax - centre)) if use_mode
                      else (f32(dmax - centre), f32(centre - dmin)))
            hicut = f32(centre + min(*spread, f32(hi * sigma)))
        else:
            hicut = f32(centre + f32(hi * sigma))
        il = ilo
        while il <= n and a.pix[index[il]] < locut:
            il += 1
        ih = ihi
        while ih >= 1 and a.pix[index[ih]] > hicut:
            ih -= 1
        if il == ilo and ih == ihi:
            break
        if rgrow > 0:
            for j in list(range(ilo, il)) + list(range(ih + 1, ihi + 1)):
                nreject += _grow(a, n, index[j], rgrow, sums)
            nsky = n - nreject
            # ap_mode looks for the median up to the old upper limit.
            med = _imed(a, index, il, ihi if use_mode else ih, (nsky + 1) // 2)
        else:
            nreject += (il - ilo) + (ihi - ih)
            nsky = n - nreject
            med = (ih + il) // 2
            for j in list(range(ilo, il)) + list(range(ih + 1, ihi + 1)):
                sums.remove(a.pix[index[j]])
        if nsky <= 0:
            break
        medcut = _nint(f32(f32(0.025) * f32(nsky)))
        med_value = _clamp(_wsmed(a, index, n, med, medcut), dmin, dmax)
        mean, sigma, skew = sums.moments(nsky)
        mean = _clamp(mean, dmin, dmax)
        value = mode_of(med_value, mean) if use_mode else med_value
        if sigma <= 0:
            break
        ilo, ihi = il, ih
    if nsky == 0 or nreject == n:
        return Sky(nsky=0, nreject=n, ier=SKY_NSKY_TOO_SMALL)
    return Sky(float(value), float(sigma), float(skew), nsky, nreject, SKY_OK)


# ------------------------------------------------------ histograms


class _Histogram:
    """The set-up the histogram algorithms share: zero point, limits, the
    median as the first guess, and the histogram itself."""

    def __init__(self, a, index, n, k1, binsize, kind):
        self.zero = _zero(a, index, n)
        self.dmin, self.dmax = _ialim(a, index, n)
        self.sums = _Sums(a.pix[index[1:n + 1]], self.zero)
        _, self.sigma, self.skew = self.sums.moments(n)
        centre = _clamp(_medr(a, index, n), self.dmin, self.dmax)
        self.median = centre
        k1, binsize = f32(k1), f32(binsize)
        cut = min(f32(centre - self.dmin), f32(self.dmax - centre), f32(k1 * self.sigma))
        self.hmin, self.hmax = f32(centre - cut), f32(centre + cut)
        dh = f32(f32(binsize * cut) / k1)
        self.nker = 1
        if dh <= 0:
            self.nbins, dh = 1, f32(0.0)
        else:
            self.nbins = 2 * _nint(f32(f32(self.hmax - centre) / dh)) + 1
            if kind == "crosscor":
                ratio = f32(f32(f32(self.hmax - centre) * f32(2.0)) / f32(dh * f32(3.0)))
                self.nker = 2 * _nint(ratio) + 1
            dh = f32(f32(self.hmax - self.hmin) / f32(self.nbins - 1))
        self.dh = dh
        self.k1 = k1

    def accumulate(self, a, index, n):
        """aphigmr: the histogram; pixels off it get weight zero."""
        nbins, z1, z2 = self.nbins, self.hmin, self.hmax
        self.hgm = np.zeros(nbins, dtype=f32)
        dz = f32(f32(nbins - 1) / f32(z2 - z1))
        nreject = 0
        for j in range(1, n + 1):
            v = a.pix[index[j]]
            if v < z1 or v > z2:
                nreject += 1
                a.wgt[index[j]] = 0
                continue
            self.hgm[int(f32(f32(v - z1) * dz))] += 1
        return nreject

    def initial_rejection(self, a, index, n, nreject):
        for j in range(1, n + 1):
            if a.wgt[index[j]] <= 0:
                self.sums.remove(a.pix[index[j]])
        return self.sums.moments(n - nreject)

    def subtract(self, v):
        """ap_hgmsub2."""
        z1, z2 = self.hmin, self.hmax
        if v < z1 or v > z2:
            return
        self.sums.remove(v)
        dh = f32(f32(self.nbins - 1) / f32(z2 - z1))
        self.hgm[int(f32(f32(v - z1) * dh))] -= 1

    def reject(self, a, index, n, locut, hicut, rgrow):
        nreject = 0
        for j in range(1, n + 1):
            p = index[j]
            if locut <= a.pix[p] <= hicut:
                continue
            if rgrow > 0:
                nreject += _grow(a, n, p, rgrow, self.sums,
                                 (self.hgm, self.hmin, self.hmax), histogram=True)
            elif a.wgt[p] > 0:
                self.subtract(a.pix[p])
                a.wgt[p] = 0
                nreject += 1
        return nreject

    def x(self):
        """amapr of 1..nbins onto the bin centres."""
        nbins = self.nbins
        b1 = f32(self.hmin + f32(self.dh * f32(0.5)))
        b2 = f32(self.hmax + f32(self.dh * f32(0.5)))
        scalar = f32(f32(b2 - b1) / f32(f32(nbins) - f32(1.0)))
        x = np.arange(1, nbins + 1, dtype=f32)
        out = (((x - f32(1.0)).astype(f32) * scalar).astype(f32) + b1).astype(f32)
        return np.clip(out, min(b1, b2), max(b1, b2)).astype(f32)


def _boxcar(h: np.ndarray, nsmooth: int) -> np.ndarray:
    """ap_sboxr, which divides by the full width even at the ends."""
    n = h.size
    ns = f32(2 * nsmooth + 1)
    out = np.zeros(n, dtype=f32)
    for i in range(n):
        out[i] = f32(_ssum(h[max(0, i - nsmooth):min(n, i + nsmooth + 1)]) / ns)
    return out


def _bsmooth(h: np.ndarray, nker: int) -> np.ndarray:
    return _boxcar(_boxcar(h, nker), nker)


def _imode(hgm: np.ndarray, z1, z2, smooth: bool):
    """ap_imode: the centroid of the histogram above a level."""
    nbins = hgm.size
    if smooth:
        hmean = f32(f32(hgm.max() * f32(2.0)) / f32(3.0))
    else:
        hmean = f32(_ssum(hgm) / f32(nbins))
    dz = f32(f32(z2 - z1) / f32(nbins - 1))
    x = float(f32(z1 + f32(dz * f32(0.5))))
    sumi = sumix = 0.0
    for v in hgm:
        dh = float(f32(v - hmean))
        if dh > 0:
            sumi += dh
            sumix += dh * x
        x += float(dz)
    if sumi > 0:
        return f32(sumix / sumi), SKY_OK
    return INDEFR, SKY_NOHISTOGRAM


def _centroid(a, index, n, k1, binsize, smooth, lo, hi, rgrow, maxiter):
    """ap_centroid."""
    if n <= 0:
        return Sky(ier=SKY_NOSKYAREA)
    h = _Histogram(a, index, n, k1, binsize, "centroid")
    sigma, skew, dh = h.sigma, h.skew, h.dh
    if h.nbins < 2 or h.k1 <= 0 or sigma < dh or dh <= 0 or sigma <= 0:
        return Sky(float(h.median), 0.0, 0.0, n, 0, SKY_NOHISTOGRAM)
    nreject = h.accumulate(a, index, n)
    nsky = n - nreject
    if nreject > 0:
        _, sigma, skew = h.initial_rejection(a, index, n, nreject)

    def peak(sigma):
        if smooth:
            nker = max(1, _nint(f32(sigma / dh)))
            return _imode(_bsmooth(h.hgm, nker), h.hmin, h.hmax, True)
        return _imode(h.hgm, h.hmin, h.hmax, False)

    mode, ier = peak(sigma)
    mode = _clamp(mode, h.dmin, h.dmax)
    if (lo is None and hi is None) or sigma < dh or maxiter < 1:
        return Sky(_undef(mode), float(sigma), float(skew), nsky, nreject, SKY_OK)
    for _ in range(maxiter):
        locut = -MAX_REAL if lo is None else f32(mode - f32(f32(lo) * sigma))
        hicut = MAX_REAL if hi is None else f32(mode + f32(f32(hi) * sigma))
        rejected = h.reject(a, index, n, locut, hicut, rgrow)
        if rejected == 0:
            break
        nreject += rejected
        nsky = n - nreject
        if nsky <= 0:
            break
        _, sigma, skew = h.sums.moments(nsky)
        mode, ier = peak(sigma)
        mode = _clamp(mode, h.dmin, h.dmax)
        if sigma <= dh or ier != SKY_OK:
            break
    if nsky == 0 or nreject == n or ier == SKY_NOHISTOGRAM:
        return Sky(nsky=0, nreject=n, ier=SKY_NSKY_TOO_SMALL)
    return Sky(_undef(mode), float(sigma), float(skew), nsky, nreject, SKY_OK)


def _gausskew(x, p):
    dx = f32(x - p[1])
    if p[2] == 0:
        r2 = f32(36.0)
    else:
        r2 = f32(f32(dx * dx) / f32(f32(2.0) * p[2]))
        r3 = f32(float(f32(r2 * dx)) / math.sqrt(float(f32(f32(2.0) * abs(p[2])))))
    if abs(r2) > 25.0:
        return f32(0.0)
    return f32(float(f32(f32(f32(1.0) + f32(p[3] * r3)) * p[0])) * math.exp(-float(r2)))


def _dgausskew(x, p):
    dx = f32(x - p[1])
    r2 = f32(36.0) if p[2] == 0 else f32(f32(dx * dx) / f32(f32(2.0) * p[2]))
    if abs(r2) > 25.0:
        return f32(0.0), (f32(0), f32(0), f32(0), f32(0))
    r = f32(float(dx) / math.sqrt(float(f32(f32(2.0) * abs(p[2])))))
    r3 = f32(r2 * r)
    d1 = f32(math.exp(-float(r2)))
    z = f32(d1 * p[0])
    d2 = f32(f32(z * dx) / p[2])
    d3 = f32(f32(z * r2) / p[2])
    rint = f32(f32(1.0) + f32(p[3] * r3))
    shape = f32(rint - f32(f32(f32(1.5) * p[3]) * r))
    der = (f32(d1 * rint), f32(d2 * shape), f32(d3 * shape), f32(z * r3))
    return f32(z * rint), der


def _hist_mode(x, hgm, maxfit):
    """ap_hist_mode: a skewed gaussian fitted to the histogram by nlfit."""
    nbins = hgm.size
    imax = int(np.argmax(hgm)) + 1
    p = [hgm.max(), x[imax - 1], f32(abs(f32(f32(x[-1] - x[0]) / f32(6.0))) ** 2), f32(0.0)]
    fit = nlfit.NLFit(lambda v, q: _gausskew(v, q), _dgausskew, p, [0, 1, 2, 3],
                      0.001, maxfit)
    fier = fit.fit(x, hgm)
    q = fit.param
    if fier == nlfit.NO_DEG_FREEDOM:
        return INDEFR, INDEFR, INDEFR, SKY_NSKY_TOO_SMALL
    ier = SKY_OK
    if fier == nlfit.SINGULAR:
        ier = SKY_SINGULAR
    elif fier == nlfit.NOT_DONE:
        ier = SKY_NOCONVERGE
    if q[1] < x[0] or q[1] > x[nbins - 1]:
        mode, ier = x[imax - 1], SKY_BADPARAMS
    else:
        mode = q[1]
    if q[2] <= 0:
        return mode, f32(0.0), f32(0.0), SKY_BADPARAMS
    sigma = f32(math.sqrt(float(q[2])))
    skew = f32(abs(float(q[3])) ** (1.0 / 3.0) * float(f32(1.743875281)) * float(sigma))
    return mode, sigma, (-skew if q[3] < 0 else skew), ier


def _gauss(a, index, n, maxfit, k1, binsize, smooth, lo, hi, rgrow, maxiter):
    """ap_gauss."""
    if n <= 0:
        return Sky(ier=SKY_NOSKYAREA)
    h = _Histogram(a, index, n, k1, binsize, "gauss")
    msigma, dh = h.sigma, h.dh
    if msigma <= dh or dh <= 0 or h.k1 <= 0 or msigma <= 0 or h.nbins < 4:
        return Sky(float(h.median), 0.0, 0.0, n, 0, SKY_NOHISTOGRAM)
    x = h.x()
    nreject = h.accumulate(a, index, n)
    nsky = n - nreject
    skew = h.skew
    if nreject > 0:
        _, msigma, skew = h.initial_rejection(a, index, n, nreject)

    def fit(msigma):
        hist = h.hgm
        if smooth:
            hist = _bsmooth(hist, max(1, _nint(f32(msigma / dh))))
        return _hist_mode(x, hist, maxfit)

    mode, sigma, skew, ier = fit(msigma)
    mode = _clamp(mode, h.dmin, h.dmax)
    if ier != SKY_OK:
        return Sky(_undef(mode), _undef(sigma), _undef(skew), nsky, nreject, ier)
    if (lo is None and hi is None) or sigma <= dh or maxiter < 1:
        return Sky(_undef(mode), float(sigma), float(skew), nsky, nreject, SKY_OK)
    for _ in range(maxiter):
        locut = -MAX_REAL if lo is None else f32(mode - f32(f32(lo) * msigma))
        hicut = MAX_REAL if hi is None else f32(mode + f32(f32(hi) * msigma))
        rejected = h.reject(a, index, n, locut, hicut, rgrow)
        if rejected == 0:
            break
        nreject += rejected
        nsky = n - nreject
        if nsky <= 0:
            break
        _, msigma, _ = h.sums.moments(nsky)
        mode, sigma, skew, ier = fit(msigma)
        mode = _clamp(mode, h.dmin, h.dmax)
        if ier != SKY_OK or sigma <= dh:
            break
    if nsky == 0 or nreject == n:
        return Sky(nsky=0, nreject=n, ier=SKY_NSKY_TOO_SMALL)
    return Sky(_undef(mode), _undef(sigma), _undef(skew), nsky, nreject, ier)


def _mapr(a, a1, a2, b1, b2):
    """apmapr."""
    scalar = f32(f32(b2 - b1) / f32(a2 - a1))
    return max(min(b1, b2), min(max(b1, b2), f32(f32(f32(a - a1) * scalar) + b1)))


def _ofilter(a, index, n, k1, binsize, smooth, lo, hi, rgrow, maxiter):
    """ap_lgsky: optimal filtering of the histogram with a triangle."""
    if n <= 0:
        return Sky(ier=SKY_NOSKYAREA)
    h = _Histogram(a, index, n, k1, binsize, "ofilter")
    sigma, skew, dh = h.sigma, h.skew, h.dh
    if h.nbins < 2 or h.k1 <= 0 or sigma <= 0 or dh <= 0 or sigma <= dh:
        return Sky(float(h.median), 0.0, 0.0, n, 0, SKY_NOHISTOGRAM)
    nreject = h.accumulate(a, index, n)
    nsky = n - nreject
    if nreject > 0:
        _, sigma, skew = h.initial_rejection(a, index, n, nreject)
    one, top = f32(1.0), f32(h.nbins)
    b1 = f32(h.hmin + f32(dh * f32(0.5)))
    b2 = f32(h.hmax + f32(dh * f32(0.5)))
    centre = _mapr(f32(f32(h.hmin + h.hmax) / f32(2.0)), b1, b2, one, top)

    def peak(sigma, centre):
        hist = h.hgm
        if smooth:
            hist = _bsmooth(hist, max(1, _nint(f32(sigma / dh))))
        return apphot.aptopt(hist, centre, f32(sigma / dh), 0.001, maxiter, False)

    iteration, centre = peak(sigma, centre)
    mode = _clamp(_mapr(centre, one, top, b1, b2), h.dmin, h.dmax)
    if iteration < 0:
        return Sky(float(mode), float(sigma), float(skew), nsky, nreject, SKY_NOCONVERGE)
    if (lo is None and hi is None) or sigma <= dh or maxiter < 1:
        return Sky(float(mode), float(sigma), float(skew), nsky, nreject, SKY_OK)
    for _ in range(maxiter):
        locut = -MAX_REAL if lo is None else f32(mode - f32(f32(lo) * sigma))
        hicut = MAX_REAL if hi is None else f32(mode + f32(f32(hi) * sigma))
        rejected = h.reject(a, index, n, locut, hicut, rgrow)
        if rejected == 0:
            break
        nreject += rejected
        nsky = n - nreject
        if nsky <= 0:
            break
        _, sigma, skew = h.sums.moments(nsky)
        if sigma <= dh:
            break
        iteration, centre = peak(sigma, centre)
        mode = _clamp(_mapr(centre, one, top, b1, b2), h.dmin, h.dmax)
        if iteration < 0:
            break
    if nsky == 0 or nreject == n:
        return Sky(nsky=0, nreject=n, ier=SKY_NSKY_TOO_SMALL)
    if sigma <= 0:
        return Sky(float(mode), 0.0, 0.0, nsky, nreject, SKY_OK)
    return Sky(float(mode), float(sigma), float(skew), nsky, nreject,
               SKY_NOCONVERGE if iteration < 0 else SKY_OK)


def _gauss_kernel(nker, kmin, kmax, sigma):
    if nker == 1:
        return np.ones(1, dtype=f32)
    kernel = np.zeros(nker, dtype=f32)
    total = f32(0.0)
    x = kmin
    dk = f32(f32(kmax - kmin) / f32(nker - 1))
    for i in range(nker):
        kernel[i] = f32(math.exp(float(f32(-f32(x * x) / f32(f32(sigma * sigma) * f32(2.0))))))
        total = f32(total + kernel[i])
        x = f32(x + dk)
    return (kernel / total).astype(f32)


def _convolve(hgm, nker, nsmooth, kernel):
    """acnvr into the middle of a cleared array, as ap_crosscor uses it."""
    out = np.zeros(hgm.size, dtype=f32)
    for i in range(nsmooth):
        total = f32(0.0)
        for j in range(nker):
            total = f32(total + f32(kernel[j] * hgm[i + j]))
        out[nker // 2 + i] = total
    return out


def _corfit(x, shgm):
    """ap_corfit: the peak of the correlation, by a parabola."""
    nbins = shgm.size
    b = int(np.argmax(shgm)) + 1
    if shgm[b - 1] <= 0:
        return INDEFR, SKY_FLAT_HIST
    if b == 1:
        return x[0], SKY_OK
    if b == nbins:
        return x[nbins - 1], SKY_OK
    xo = f32(f32(x[b - 1] + x[b - 2]) * f32(0.5))
    dh1 = f32(shgm[b - 1] - shgm[b - 2])
    dh2 = f32(shgm[b - 1] - shgm[b])
    return f32(xo + f32(f32(x[b - 1] - x[b - 2]) * f32(dh1 / f32(dh1 + dh2)))), SKY_OK


def _crosscor(a, index, n, k1, binsize, lo, hi, rgrow, maxiter):
    """ap_crosscor: the peak of the histogram's correlation with a
    gaussian of the sky's sigma."""
    if n <= 0:
        return Sky(ier=SKY_NOSKYAREA)
    h = _Histogram(a, index, n, k1, binsize, "crosscor")
    sigma, skew, dh = h.sigma, h.skew, h.dh
    nker = h.nker
    nsmooth = h.nbins - nker + 1
    half = f32(f32(nker // 2) + f32(0.5))
    kmin, kmax = f32(-dh * half), f32(dh * half)
    if h.nbins < 2 or h.k1 <= 0 or sigma <= 0 or dh <= 0 or sigma <= dh:
        return Sky(float(h.median), 0.0, 0.0, n, 0, SKY_NOHISTOGRAM)
    x = h.x()
    nreject = h.accumulate(a, index, n)
    nsky = n - nreject
    mean = h.median            # ap_crosscor's sky_mean is the median until the moments move it
    if nreject > 0:
        mean, sigma, skew = h.initial_rejection(a, index, n, nreject)
    if sigma > 0:
        shgm = _convolve(h.hgm, nker, nsmooth, _gauss_kernel(nker, kmin, kmax, sigma))
    else:
        shgm = h.hgm.copy()
    mode, ier = _corfit(x, shgm)
    mode = _clamp(mode, h.dmin, h.dmax)
    if ier != SKY_OK:
        return Sky(_undef(mode), float(sigma), float(skew), nsky, nreject, ier)
    if (lo is None and hi is None) or sigma <= dh or maxiter < 1:
        return Sky(float(mode), float(sigma), float(skew), nsky, nreject, SKY_OK)
    for _ in range(maxiter):
        locut = -MAX_REAL if lo is None else f32(mode - f32(f32(lo) * sigma))
        hicut = MAX_REAL if hi is None else f32(mode + f32(f32(hi) * sigma))
        rejected = h.reject(a, index, n, locut, hicut, rgrow)
        if rejected == 0:
            break
        nreject += rejected
        nsky = n - nreject
        if nsky <= 0:
            break
        mean, sigma, skew = h.sums.moments(nsky)
        if sigma <= dh:
            break
        shgm = _convolve(h.hgm, nker, nsmooth, _gauss_kernel(nker, kmin, kmax, sigma))
        mode, ier = _corfit(x, shgm)
        mode = _clamp(mode, h.dmin, h.dmax)
        if ier != SKY_OK:
            break
    if nsky == 0 or nreject == n:
        return Sky(nsky=0, nreject=n, ier=SKY_NSKY_TOO_SMALL)
    if ier != SKY_OK:
        return Sky(float(mean), 0.0, 0.0, nsky, nreject, ier)
    return Sky(float(mode), float(sigma), float(skew), nsky, nreject, SKY_OK)


# ------------------------------------------------------------ apfitsky

ALGORITHMS = ("centroid", "mode", "median", "mean", "gauss", "ofilter", "crosscor",
              "histplot", "radplot", "constant", "file")

SKY_NOGRAPHICS, SKY_NOSKYFILE, SKY_EOFSKYFILE, SKY_BADSKYSCAN = 208, 209, 210, 211


class Cursor:
    """The graphics cursor histplot and radplot read: IRAF's gcommands, a
    file of 'x y [wcs] key' lines that every star reads on from where the
    last stopped -- or, given `show`, the window, where the marks are made
    by hand. Each star reads until q, or the end."""

    def __init__(self, path=None, show=None) -> None:
        self.lines: list[tuple[float, float, str]] = []
        self.show = show
        self.image = ""     #: for the title of the graphs
        if path:
            from pathlib import Path

            for line in Path(path).read_text().splitlines():
                words = line.split()
                if not words or words[0].startswith("#"):
                    continue
                if len(words) == 1:
                    self.lines.append((float("nan"), float("nan"), words[0]))
                    continue
                key = words[3] if len(words) > 3 else words[2] if len(words) > 2 else " "
                self.lines.append((float(words[0]), float(words[1]), key))
        self.at = 0

    def events(self, graph, start):
        if self.show is not None:
            yield from self.show(graph, start)
            return
        while self.at < len(self.lines):
            event = self.lines[self.at]
            self.at += 1
            yield event
            if event[2] == "q":
                return


class SkyFile:
    """phot's skyfile, for salgorithm=file: one 'x y msky stdev sskew nsky
    nsrej' line per star, read in order (ap_readsky)."""

    def __init__(self, path) -> None:
        from pathlib import Path

        self.lines = [line for line in Path(path).read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith("#")]
        self.at = 0

    def read(self) -> Sky:
        if self.at >= len(self.lines):
            return Sky(ier=SKY_EOFSKYFILE)
        words = self.lines[self.at].split()
        self.at += 1
        values: list[float] = []
        for k, word in enumerate(words[:7]):
            try:
                values.append(int(word) if k >= 5 else float(f32(float(word))))
            except ValueError:
                break
        if len(values) != 7:
            return Sky(ier=SKY_BADSKYSCAN)
        _, _, mode, sigma, skew, nsky, nreject = values
        if nsky <= 0:
            return Sky(float(mode), float(sigma), float(skew), int(nsky), int(nreject),
                       SKY_NOSKYAREA)
        return Sky(float(mode), float(sigma), float(skew), int(nsky), int(nreject), SKY_OK)


def fitsky(data: np.ndarray, x: float, y: float, pars: SkyPars,
           dpars: DataPars, cursor: Cursor | None = None,
           skyfile: SkyFile | None = None) -> Sky:
    """apfitsky."""
    if pars.salgorithm == "constant":
        return Sky(pars.skyvalue, dpars.sigma if dpars.sigma is not None
                   else float("nan"), float("nan"), 0, 0, SKY_OK)
    if pars.salgorithm == "file":
        return skyfile.read() if skyfile is not None else Sky(ier=SKY_NOSKYFILE)
    if pars.salgorithm in ("histplot", "radplot") and cursor is None:
        return Sky(ier=SKY_NOGRAPHICS)
    a, nbad, ier = sky_buffer(data, x, y, pars.annulus, pars.dannulus, dpars)
    if a is None:
        return Sky(nreject=nbad, ier=ier)
    sorted_order = np.concatenate((np.zeros(1, dtype=np.int64),
                                   np.argsort(a.pix[1:], kind="stable") + 1))
    nclip = 0
    if pars.sloclip > 0 or pars.shiclip > 0:
        # ap_clip: sort, and drop the percentiles from either end.
        loindex = _nint(f32(f32(f32(pars.sloclip) * f32(0.01)) * f32(a.n))) + 1
        hiindex = a.n - _nint(f32(f32(f32(pars.shiclip) * f32(0.01)) * f32(a.n)))
        nclip = a.n if hiindex - loindex + 1 <= 0 else loindex - 1 + a.n - hiindex
        if nclip >= a.n:
            return Sky(nreject=nbad, ier=SKY_NSKY_TOO_SMALL)
        index = np.concatenate(([0], sorted_order[loindex:hiindex + 1]))
    elif pars.salgorithm in ("median", "mode"):
        index = sorted_order
    elif pars.salgorithm in ("histplot", "radplot"):
        index = np.arange(a.n + 1)
    else:
        index = np.arange(a.n + 1)
    n = a.n - nclip
    rgrow = f32(f32(pars.rgrow) * f32(dpars.scale))
    lo, hi = pars.sloreject, pars.shireject
    algorithm = pars.salgorithm
    if algorithm == "mean":
        sky = _mean(a, index, n, lo, hi, rgrow, pars.snreject)
    elif algorithm in ("median", "mode"):
        sky = _median_mode(a, index, n, lo, hi, rgrow, pars.snreject, algorithm == "mode")
    elif algorithm == "centroid":
        sky = _centroid(a, index, n, pars.khist, pars.binsize, pars.smooth, lo, hi,
                        rgrow, pars.smaxiter)
    elif algorithm == "gauss":
        sky = _gauss(a, index, n, pars.smaxiter, pars.khist, pars.binsize, pars.smooth,
                     lo, hi, rgrow, pars.snreject)
    elif algorithm == "histplot":
        sky = _histplot(a, index, n, pars.khist, pars.binsize, pars.smooth, cursor, x, y)
    elif algorithm == "radplot":
        sky = _radplot(a, index, n, cursor, x, y, dpars.scale)
    elif algorithm == "ofilter":
        sky = _ofilter(a, index, n, pars.khist, pars.binsize, pars.smooth, lo, hi,
                       rgrow, pars.smaxiter)
    else:
        sky = _crosscor(a, index, n, pars.khist, pars.binsize, lo, hi, rgrow,
                        pars.smaxiter)
    sky.nreject += nbad + nclip
    return sky


# ------------------------------------------------------ marked by hand


def _histplot(a, index, n, k1, binsize, smooth, cursor, x, y):
    """ap_histplot: the histogram drawn, its peak marked with the cursor --
    every key but q moves the mark there, clamped to the histogram. Sigma
    and skew are the moments of the pixels on the histogram."""
    from .graph import Graph

    if n <= 0:
        return Sky(nsky=0, ier=apphot.SKY_OUTOFBOUNDS)
    h = _Histogram(a, index, n, k1, binsize, "centroid")
    mode = h.median
    if h.dh <= 0 or h.k1 <= 0 or h.sigma <= 0 or h.sigma <= h.dh or h.nbins < 2:
        return Sky(float(mode), 0.0, 0.0, n, 0, SKY_NOHISTOGRAM)
    nreject = h.accumulate(a, index, n)
    nsky = n - nreject
    _, sigma, skew = h.initial_rejection(a, index, n, nreject)
    graph = Graph([f"Image: {cursor.image}:  {x:.2f} {y:.2f}",
                   f"Sky Histogram: nbins = {h.nbins} hmin = {iraffmt.g(float(h.hmin))} "
                   f"hmax = {iraffmt.g(float(h.hmax))} smooth={'YES' if smooth else 'NO'}"],
                  xlabel="Sky Values", ylabel="Number of Pixels")
    graph.hint = f"Mark histogram peak ({float(mode):g}) [space=mark, q=quit]"
    centres = h.x()
    graph.steps(np.append(centres - h.dh / 2, centres[-1] + h.dh / 2), h.hgm)
    if smooth:
        nker = max(1, _nint(f32(sigma / h.dh)))
        graph.line(centres, _bsmooth(h.hgm, nker))
    graph.scale_to(float(h.hmin), float(h.hmax))
    for wx, _wy, key in cursor.events(graph, float(mode)):
        if key == "q":
            break
        mode = max(h.hmin, min(f32(wx), h.hmax))
    return Sky(float(mode), float(sigma), float(skew), nsky, nreject, SKY_OK)


def _radplot(a, index, n, cursor, x, y, scale):
    """ap_radplot: the sky pixels against their distance from the star,
    the sky level marked with the cursor's y; sigma and skew of them all."""
    from .graph import Graph

    if n <= 0:
        return Sky(nsky=0, ier=apphot.SKY_OUTOFBOUNDS)
    zero = _zero(a, index, n)
    mean, sigma, skew = _Sums(a.pix[index[1:n + 1]], zero).moments(n)
    mode = mean
    coords = a.coords[index[1:n + 1]] - 1
    col = coords % a.snx + 1
    row = coords // a.snx + 1
    r = np.hypot(col - a.sxc, row - a.syc) / scale
    graph = Graph([f"Image: {cursor.image}:  {x:.2f} {y:.2f}"],
                  xlabel="Radial Distance (lower-pixels, upper-scale units)",
                  ylabel="Intensity")
    graph.hint = f"Mark sky level ({float(mode):g}) [space=mark, q=quit]"
    pixels = a.pix[index[1:n + 1]]
    graph.marks = (r, pixels)
    graph.window = (float(r.min()), float(r.max()), float(pixels.min()), float(pixels.max()))
    for _wx, wy, key in cursor.events(graph, float(mode)):
        if key == "q":
            break
        mode = f32(wy)
    return Sky(float(mode), float(sigma), float(skew), n, 0, SKY_OK)
