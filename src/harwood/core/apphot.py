"""IRAF's apphot arithmetic: centring, sky fitting and aperture sums.

`center`, `fitsky` and `phot` in IRAF are one pipeline with three entry
points: phot centres the star (centerpars), fits the sky around the *new*
centre (fitskypars), then sums the apertures and computes magnitudes
(photpars). This module is that pipeline, ported from iraf-community 2.18
(`noao/digiphot/apphot/{center,fitsky,phot,aputil}`), so a magnitude from
harwood is the magnitude IRAF would have printed for the same star -- and a
test holds it to IRAF's own output.

What is ported, and what is not yet:

=============  ========================================================
centring       centroid (IRAF's default) and none. gauss and ofilter
               are refused by name.
sky            centroid (the default), mode, median, mean and constant,
               with k-sigma rejection. Region growing (rgrow > 0), the
               percentile clips (sloclip, shiclip) and histogram
               smoothing are refused by name.
apertures      the sums with IRAF's fractional edge pixels, the
               three-term error, the itime correction; positive
               features only (datapars emission = yes).
=============  ========================================================

Positions are 1-based (x, y) image coordinates, as in IRAF.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# IRAF's error codes, as phot writes them in the CIER, SIER and PIER columns.
# From noao/digiphot/apphot/lib/{center,fitsky,phot}.h.
CTR_OK, CTR_NOAREA, CTR_OUTOFBOUNDS, CTR_LOWSNRATIO = 0, 101, 102, 103
CTR_BADSHIFT, CTR_BADDATA = 107, 108
SKY_OK, SKY_NOSKYAREA, SKY_OUTOFBOUNDS, SKY_NOHISTOGRAM = 0, 201, 202, 203
SKY_NSKY_TOO_SMALL = 205
APERT_OK, APERT_NOAPERT, APERT_OUTOFBOUNDS, APERT_NOSKYMODE = 0, 301, 302, 303
APERT_BADDATA = 305

MESSAGES = {
    CTR_NOAREA: "the centring box is off the image",
    CTR_OUTOFBOUNDS: "the centring box is partly off the image",
    CTR_LOWSNRATIO: "the signal-to-noise in the centring box is below minsnratio",
    CTR_BADDATA: "bad pixels (outside datamin/datamax) in the centring box",
    CTR_BADSHIFT: "the centre moved more than maxshift from where it started",
    SKY_NOSKYAREA: "the sky annulus has no area",
    SKY_OUTOFBOUNDS: "the sky annulus is off the image",
    SKY_NSKY_TOO_SMALL: "too few sky pixels survived rejection",
    SKY_NOHISTOGRAM: "the sky histogram could not be built",
    APERT_NOAPERT: "the aperture is off the image",
    APERT_OUTOFBOUNDS: "the largest apertures are partly off the image",
    APERT_NOSKYMODE: "no sky value, so no magnitude",
    APERT_BADDATA: "bad pixels (outside datamin/datamax) in the aperture",
}


# ------------------------------------------------------------------ pars


@dataclass
class DataPars:
    """datapars: what the data are."""

    scale: float = 1.0
    fwhmpsf: float = 2.5
    sigma: float | None = None        #: INDEF
    datamin: float | None = None
    datamax: float | None = None
    epadu: float = 1.0
    readnoise: float = 0.0
    itime: float = 1.0


@dataclass
class CenterPars:
    """centerpars."""

    calgorithm: str = "centroid"
    cbox: float = 5.0
    cthreshold: float = 0.0
    minsnratio: float = 1.0
    cmaxiter: int = 10
    maxshift: float = 1.0


@dataclass
class SkyPars:
    """fitskypars."""

    salgorithm: str = "centroid"
    annulus: float = 10.0
    dannulus: float = 10.0
    skyvalue: float = 0.0
    smaxiter: int = 10
    snreject: int = 50
    sloreject: float = 3.0
    shireject: float = 3.0
    khist: float = 3.0
    binsize: float = 0.1


@dataclass
class Center:
    x: float
    y: float
    xshift: float = 0.0
    yshift: float = 0.0
    xerr: float = float("nan")
    yerr: float = float("nan")
    ier: int = CTR_OK


@dataclass
class Sky:
    value: float = float("nan")
    sigma: float = float("nan")
    skew: float = float("nan")
    nsky: int = 0
    nreject: int = 0
    ier: int = SKY_OK


@dataclass
class Apertures:
    radii: list[float]
    sums: np.ndarray = field(repr=False)
    areas: np.ndarray = field(repr=False)
    flux: np.ndarray = field(repr=False)
    mag: np.ndarray = field(repr=False)
    merr: np.ndarray = field(repr=False)
    ier: int = APERT_OK


def _trunc(value: float) -> int:
    """SPP's real-to-int assignment: truncation, as in Fortran."""
    return int(value)


# --------------------------------------------------------------- centring


def center(data: np.ndarray, x: float, y: float, pars: CenterPars | None = None,
           dpars: DataPars | None = None) -> Center:
    """apfitcenter, for the centroid and none algorithms."""
    from . import advice

    pars, dpars = pars or CenterPars(), dpars or DataPars()
    if pars.calgorithm == "none":
        return Center(x, y)
    if pars.calgorithm != "centroid":
        advice.fail(f"calgorithm '{pars.calgorithm}' is not ported from IRAF yet",
                    "use calgorithm=centroid, IRAF's default, or none")

    ny, nx = data.shape
    cthreshold = (0.0 if dpars.sigma is None or pars.cthreshold <= 0
                  else pars.cthreshold * dpars.sigma)
    capert = pars.cbox / 2.0
    icpix = 2 * _trunc(max(1.0, capert * dpars.scale)) + 1
    half = (icpix - 1) // 2

    owx, owy = x, y
    xc_new, yc_new = x, y
    ier = CTR_OK
    lowsnr = False
    for _ in range(pars.cmaxiter):
        # ap_ctrpix: the box around the current centre.
        if (owx - half > nx or owx + half < 1 or owy - half > ny or owy + half < 1):
            return Center(x, y, ier=CTR_NOAREA)
        c1 = _trunc(max(1.0, min(float(nx), owx - half)) + 0.5)
        c2 = _trunc(min(float(nx), max(1.0, owx + half)) + 0.5)
        l1 = _trunc(max(1.0, min(float(ny), owy - half)) + 0.5)
        l2 = _trunc(min(float(ny), max(1.0, owy + half)) + 0.5)
        box = data[l1 - 1:l2, c1 - 1:c2].astype(float)
        bnx, bny = c2 - c1 + 1, l2 - l1 + 1
        cxc, cyc = owx - c1 + 1, owy - l1 + 1

        if not np.isfinite(box).all():
            # Blank pixels: IRAF has none; treat them as bad data and keep the
            # position rather than centring on an undefined marginal.
            return Center(x, y, ier=CTR_BADDATA)
        ier = CTR_OK
        low = -np.inf if dpars.datamin is None else dpars.datamin
        high = np.inf if dpars.datamax is None else dpars.datamax
        if box.min() < low or box.max() > high:
            ier = CTR_BADDATA
        elif bnx < icpix or bny < icpix:
            ier = CTR_OUTOFBOUNDS

        # The threshold: the box minimum, or the lowest row median when a
        # cthreshold is set; everything at or below it counts for nothing.
        floor = box.min() if cthreshold <= 0 else float(np.min(np.median(box, axis=1)))
        box = np.maximum(box - (floor + cthreshold), 0.0)

        lowsnr = _snratio(box, dpars) < pars.minsnratio

        if cthreshold <= 0:
            xc, xerr = _cmmarg(box.mean(axis=0), dpars.epadu)
            yc, yerr = _cmmarg(box.mean(axis=1), dpars.epadu)
        else:
            xc, xerr = _cmarg(box.mean(axis=0), dpars.epadu)
            yc, yerr = _cmarg(box.mean(axis=1), dpars.epadu)
        if np.isnan(xerr):
            xc = cxc
        if np.isnan(yerr):
            yc = cyc

        xc = max(0.5, min(bnx + 0.5, xc))
        yc = max(0.5, min(bny + 0.5, yc))
        xshift, yshift = xc - cxc, yc - cyc
        xc_new, yc_new = xshift + owx, yshift + owy
        owx, owy = xc_new, yc_new
        if abs(xshift) < 1.0 and abs(yshift) < 1.0:
            break

    result = Center(xc_new, yc_new, xc_new - x, yc_new - y, xerr, yerr)
    if ier == CTR_BADDATA:
        result.ier = CTR_BADDATA
    elif lowsnr:
        result.ier = CTR_LOWSNRATIO
    elif (abs(result.xshift) > pars.maxshift * dpars.scale
          or abs(result.yshift) > pars.maxshift * dpars.scale):
        result.ier = CTR_BADSHIFT
    elif ier == CTR_OUTOFBOUNDS:
        result.ier = CTR_OUTOFBOUNDS
    return result


def _snratio(box: np.ndarray, dpars: DataPars) -> float:
    """ap_pratio: the poisson noise model, with the sky taken as zero."""
    signal = float(box.sum())
    noise2 = abs(signal / dpars.epadu)
    if dpars.sigma is not None:
        noise2 += box.size * dpars.sigma ** 2
    if signal <= 0:
        return 0.0
    if noise2 <= 0:
        return np.inf
    return signal / np.sqrt(noise2)


def _cmmarg(a: np.ndarray, norm: float) -> tuple[float, float]:
    """ap_cmmarg: centroid of a marginal, using only what is above its mean."""
    n = a.size
    mean = a.sum() / n
    i = np.arange(1, n + 1, dtype=float)
    val = a - mean
    pos = val > 0
    if not pos.any():
        return (1.0 + n) / 2.0, float("nan")
    sumi = val[pos].sum()
    xc = float((val[pos] * i[pos]).sum() / sumi)
    err = float((val[pos] * i[pos] ** 2).sum() / sumi - xc * xc)
    if err <= 0:
        return xc, 0.0
    err = float(np.sqrt(err / ((sumi + mean * n) * norm)))
    return xc, (err if err <= n else float("nan"))


def _cmarg(a: np.ndarray, norm: float) -> tuple[float, float]:
    """ap_cmarg: centroid of a marginal, every point weighted by its value."""
    n = a.size
    i = np.arange(1, n + 1, dtype=float)
    if not (a > 0).any():
        return (1.0 + n) / 2.0, float("nan")
    sumi = a.sum()
    xc = float((a * i).sum() / sumi)
    err = float((a * i ** 2).sum() / sumi - xc * xc)
    if err <= 0:
        return xc, 0.0
    err = float(np.sqrt(err / (sumi * norm)))
    return xc, (err if err <= n else float("nan"))


# ---------------------------------------------------------------- the sky


def sky_pixels(data: np.ndarray, x: float, y: float, annulus: float, dannulus: float,
               dpars: DataPars | None = None) -> tuple[np.ndarray, int, int]:
    """ap_skypix / ap_bskypix: the annulus, row by row, and how many were bad.

    Returns the pixel values, the number rejected as bad, and an error code.
    """
    dpars = dpars or DataPars()
    rin, rout = annulus * dpars.scale, (annulus + dannulus) * dpars.scale
    if dannulus * dpars.scale <= 0 or rout <= rin:
        return np.array([]), 0, SKY_NOSKYAREA
    ny, nx = data.shape
    if x + rout < 1 or x - rout > nx or y + rout < 1 or y - rout > ny:
        return np.array([]), 0, SKY_OUTOFBOUNDS
    c1 = _trunc(max(1.0, min(float(nx), x - rout)) + 0.5)
    c2 = _trunc(min(float(nx), max(1.0, x + rout)) + 0.5)
    l1 = _trunc(max(1.0, min(float(ny), y - rout)) + 0.5)
    l2 = _trunc(min(float(ny), max(1.0, y + rout)) + 0.5)
    j, i = np.mgrid[l1:l2 + 1, c1:c2 + 1]
    r2 = (x - i) ** 2 + (y - j) ** 2
    ring = (r2 > rin ** 2) & (r2 <= rout ** 2)
    values = data[l1 - 1:l2, c1 - 1:c2][ring].astype(float)
    finite = np.isfinite(values)
    bad = int((~finite).sum())          # blank pixels count as bad, as datamin's do
    values = values[finite]
    if dpars.datamin is not None or dpars.datamax is not None:
        low = -np.inf if dpars.datamin is None else dpars.datamin
        high = np.inf if dpars.datamax is None else dpars.datamax
        good = (values >= low) & (values <= high)
        bad += int((~good).sum())
        values = values[good]
    if values.size == 0:
        return values, bad, (SKY_NSKY_TOO_SMALL if bad else SKY_OUTOFBOUNDS)
    return values, bad, SKY_OK


def fitsky(data: np.ndarray, x: float, y: float, pars: SkyPars | None = None,
           dpars: DataPars | None = None) -> Sky:
    """apfitsky."""
    from . import advice

    pars, dpars = pars or SkyPars(), dpars or DataPars()
    if pars.salgorithm == "constant":
        return Sky(pars.skyvalue, dpars.sigma if dpars.sigma is not None
                   else float("nan"), float("nan"), 0, 0, SKY_OK)
    algorithms = {"centroid": _sky_centroid, "mode": _sky_mode,
                  "median": _sky_median, "mean": _sky_mean}
    if pars.salgorithm not in algorithms:
        advice.fail(f"salgorithm '{pars.salgorithm}' is not ported from IRAF yet",
                    "use centroid (IRAF's default), mode, median, mean or constant")

    values, bad, ier = sky_pixels(data, x, y, pars.annulus, pars.dannulus, dpars)
    if ier != SKY_OK:
        return Sky(nreject=bad, ier=ier)
    sky = algorithms[pars.salgorithm](values, pars)
    sky.nreject += bad
    return sky


def _moments(sums: np.ndarray, n: int, zero: float) -> tuple[float, float, float]:
    """apmoments: mean, sigma and skew from running sums about `zero`."""
    s1, s2, s3 = sums
    mean = s1 / n
    var = s2 / n - mean ** 2
    if var <= 0:
        return zero + mean, 0.0, 0.0
    skew3 = s3 / n - 3.0 * mean * var - mean ** 3
    skew = float(np.sign(skew3) * abs(skew3) ** (1.0 / 3.0))
    return zero + mean, float(np.sqrt(var)), skew


def _sums(values: np.ndarray, zero: float) -> np.ndarray:
    d = values.astype(np.float64) - zero
    return np.array([d.sum(), (d * d).sum(), (d * d * d).sum()])


def _sky_centroid(pix: np.ndarray, pars: SkyPars) -> Sky:
    """ap_centroid: the moments of the histogram of the annulus."""
    n = pix.size
    zero = float(np.float32(pix.sum() / n))
    dmin, dmax = float(pix.min()), float(pix.max())
    sums = _sums(pix, zero)
    _, sigma, skew = _moments(sums, n, zero)
    centre = float(np.sort(pix)[(n + 1) // 2 - 1]) if n > 2 else float(pix[0])
    centre = max(dmin, min(centre, dmax))

    k1 = pars.khist
    cut = min(centre - dmin, dmax - centre, k1 * sigma)
    hmin = np.float32(centre - cut)
    hmax = np.float32(centre + cut)
    dh = pars.binsize * cut / k1 if k1 > 0 else 0.0
    if dh <= 0:
        nbins, dh = 1, 0.0
    else:
        nbins = 2 * int(np.floor((hmax - centre) / dh + 0.5)) + 1
        dh = float(hmax - hmin) / (nbins - 1)
    if nbins < 2 or k1 <= 0 or sigma < dh or dh <= 0 or sigma <= 0:
        return Sky(centre, 0.0, 0.0, n, 0, SKY_NOHISTOGRAM)

    # aphigmr: pixels off the histogram are rejected from the start.
    scale = np.float32(nbins - 1) / np.float32(hmax - hmin)
    pix32 = pix.astype(np.float32)
    inside = (pix32 >= hmin) & (pix32 <= hmax)
    bins = ((pix32[inside] - hmin) * scale).astype(np.int64)
    hist = np.bincount(bins, minlength=nbins).astype(float)
    weight = inside.copy()
    nreject = int((~inside).sum())
    nsky = n - nreject
    if nreject:
        sums -= _sums(pix[~inside], zero)
        _, sigma, skew = _moments(sums, nsky, zero)

    def mode() -> float | None:
        level = hist.sum() / nbins
        excess = hist - level
        above = excess > 0
        if not above.any():
            return None
        centres = hmin + (np.arange(nbins) + 0.5) * dh
        return float((excess[above] * centres[above]).sum() / excess[above].sum())

    value = mode()
    if value is None:
        return Sky(nsky=0, nreject=n, ier=SKY_NSKY_TOO_SMALL)
    value = max(dmin, min(value, dmax))
    if sigma < dh or pars.smaxiter < 1:
        return Sky(value, sigma, skew, nsky, nreject, SKY_OK)

    for _ in range(pars.smaxiter):
        locut = value - pars.sloreject * sigma
        hicut = value + pars.shireject * sigma
        out = weight & ((pix < locut) | (pix > hicut))
        if not out.any():
            break
        for v in pix32[out]:
            if hmin <= v <= hmax:
                hist[int((v - hmin) * scale)] -= 1
        sums -= _sums(pix[out], zero)
        weight &= ~out
        nreject += int(out.sum())
        nsky = n - nreject
        if nsky <= 0:
            break
        _, sigma, skew = _moments(sums, nsky, zero)
        found = mode()
        if found is None:
            return Sky(nsky=0, nreject=n, ier=SKY_NSKY_TOO_SMALL)
        value = max(dmin, min(found, dmax))
        if sigma <= dh:
            break
    if nsky <= 0:
        return Sky(nsky=0, nreject=n, ier=SKY_NSKY_TOO_SMALL)
    return Sky(value, sigma, skew, nsky, nreject, SKY_OK)


def _smoothed_median(sorted_pix: np.ndarray, n: int) -> float:
    """apsmed: the median averaged over 2.5% of the points either side."""
    medcut = int(np.floor(0.025 * n + 0.5))
    med = (n + 1) // 2
    lo = max(1, med - medcut)
    hi = min(n, med + medcut + (0 if med % 2 == 1 else 1))
    return float(sorted_pix[lo - 1:hi].mean())


def _weighted_smoothed_median(sorted_pix: np.ndarray, med: int, nsky: int) -> float:
    """apwsmed, with every weight one -- which is what it sees without rgrow."""
    medcut = int(np.floor(0.025 * nsky + 0.5))
    maxmed = 2 * medcut + (1 if med % 2 == 1 else 2)
    lo = max(1, med - medcut)
    taken = med - lo + 1
    hi = min(sorted_pix.size, med + (maxmed - taken))
    return float(sorted_pix[lo - 1:hi].mean())


def _sky_sorted(pix: np.ndarray, pars: SkyPars, use_mode: bool) -> Sky:
    """ap_mode and ap_median: rejection from the ends of the sorted pixels."""
    s = np.sort(pix)
    n = s.size
    zero = float(np.float32(s.sum() / n))
    dmin, dmax = float(s[0]), float(s[-1])
    med_value = max(dmin, min(_smoothed_median(s, n), dmax))
    sums = _sums(s, zero)
    mean, sigma, skew = _moments(sums, n, zero)
    mean = max(dmin, min(mean, dmax))

    def estimate(med_value, mean):
        if not use_mode:
            return med_value
        value = mean if mean < med_value else 3.0 * med_value - 2.0 * mean
        return max(dmin, min(value, dmax))

    value = estimate(med_value, mean)
    if pars.snreject < 1 or sigma <= 0:
        return Sky(value, sigma, skew, n, 0, SKY_OK)

    ilo, ihi, nreject, nsky = 1, n, 0, n
    centre = med_value
    for iteration in range(1, pars.snreject + 1):
        reference = centre if iteration == 1 or not use_mode else value
        if not use_mode:
            reference = centre
        if iteration == 1:
            span = min(centre - dmin, dmax - centre)
            locut = centre - min(span, pars.sloreject * sigma)
            hicut = centre + min(span, pars.shireject * sigma)
        else:
            locut = reference - pars.sloreject * sigma
            hicut = reference + pars.shireject * sigma
        il = ilo
        while il <= n and s[il - 1] < locut:
            il += 1
        ih = ihi
        while ih >= 1 and s[ih - 1] > hicut:
            ih -= 1
        if il == ilo and ih == ihi:
            break
        nreject += (il - ilo) + (ihi - ih)
        nsky = n - nreject
        med = (ih + il) // 2
        if ilo < il:
            sums -= _sums(s[ilo - 1:il - 1], zero)
        if ih < ihi:
            sums -= _sums(s[ih:ihi], zero)
        if nsky <= 0:
            break
        centre = max(dmin, min(_weighted_smoothed_median(s, med, nsky), dmax))
        mean, sigma, skew = _moments(sums, nsky, zero)
        mean = max(dmin, min(mean, dmax))
        value = estimate(centre, mean)
        if sigma <= 0:
            break
        ilo, ihi = il, ih
    if nsky <= 0 or nreject >= n:
        return Sky(nsky=0, nreject=n, ier=SKY_NSKY_TOO_SMALL)
    return Sky(value, sigma, skew, nsky, nreject, SKY_OK)


def _sky_mode(pix, pars):
    return _sky_sorted(pix, pars, use_mode=True)


def _sky_median(pix, pars):
    return _sky_sorted(pix, pars, use_mode=False)


def _sky_mean(pix: np.ndarray, pars: SkyPars) -> Sky:
    """ap_mean: k-sigma rejection about the mean."""
    n = pix.size
    zero = float(np.float32(pix.sum() / n))
    dmin, dmax = float(pix.min()), float(pix.max())
    sums = _sums(pix, zero)
    mean, sigma, skew = _moments(sums, n, zero)
    mean = max(dmin, min(mean, dmax))
    if pars.snreject < 1 or sigma <= 0:
        return Sky(mean, sigma, skew, n, 0, SKY_OK)
    weight = np.ones(n, bool)
    nreject = 0
    for iteration in range(1, pars.snreject + 1):
        if iteration == 1:
            span = min(mean - dmin, dmax - mean)
            locut = mean - min(span, pars.sloreject * sigma)
            hicut = mean + min(span, pars.shireject * sigma)
        else:
            locut, hicut = mean - pars.sloreject * sigma, mean + pars.shireject * sigma
        out = weight & ((pix < locut) | (pix > hicut))
        if not out.any():
            break
        sums -= _sums(pix[out], zero)
        weight &= ~out
        nreject += int(out.sum())
        if n - nreject <= 0:
            break
        mean, sigma, skew = _moments(sums, n - nreject, zero)
        if sigma <= 0:
            break
        mean = max(dmin, min(mean, dmax))
    if n - nreject <= 0:
        return Sky(nsky=0, nreject=n, ier=SKY_NSKY_TOO_SMALL)
    return Sky(mean, sigma, skew, n - nreject, nreject, SKY_OK)


# --------------------------------------------------------------- apertures


def apertures(data: np.ndarray, x: float, y: float, radii: list[float], sky: Sky,
              zmag: float = 25.0, dpars: DataPars | None = None) -> Apertures:
    """apmag: the sums, areas, magnitudes and errors of a set of apertures.

    A pixel at distance r from the centre counts with weight
    clip(radius - (r - 0.5), 0, 1): whole inside, nothing outside, and a
    linear share across the last pixel -- IRAF's approximation to the area
    of the pixel that the circle covers.
    """
    dpars = dpars or DataPars()
    radii = sorted(float(r) for r in radii)
    scaled = np.array(radii) * dpars.scale
    ny, nx = data.shape
    nap = len(radii)
    sums, areas = np.zeros(nap), np.zeros(nap)
    flux = np.full(nap, np.nan)
    mag, merr = np.full(nap, np.nan), np.full(nap, np.nan)

    # ap_photpix: the largest aperture whose box is on the image.
    nmax = 0
    for k in range(nap, 0, -1):
        half = scaled[k - 1]
        if (x - half >= 0.5 and x + half <= nx + 0.5
                and y - half >= 0.5 and y + half <= ny + 0.5):
            nmax = k
            break
    if nmax == 0:
        return Apertures(radii, sums, areas, flux, mag, merr, APERT_NOAPERT)
    half = scaled[nmax - 1]
    c1 = _trunc(max(1.0, min(float(nx), x - half)))
    c2 = _trunc(min(float(nx), max(1.0, x + half + 0.5)))
    l1 = _trunc(max(1.0, min(float(ny), y - half)))
    l2 = _trunc(min(float(ny), max(1.0, y + half + 0.5)))
    box = data[l1 - 1:l2, c1 - 1:c2].astype(np.float64)
    j, i = np.mgrid[l1:l2 + 1, c1:c2 + 1]
    r = np.sqrt((i - x) ** 2 + (j - y) ** 2) - 0.5
    low = -np.inf if dpars.datamin is None else dpars.datamin
    high = np.inf if dpars.datamax is None else dpars.datamax
    nmin = nmax + 1
    for k in range(nmax):
        weight = np.clip(scaled[k] - r, 0.0, 1.0) * (r <= scaled[k])
        sums[k] = float((weight * box).sum())
        areas[k] = float(weight.sum())
        covered = weight > 0
        inside = box[covered]
        if nmin > nmax and ((inside < low) | (inside > high) | ~np.isfinite(inside)).any():
            nmin = k + 1

    ier = APERT_OK if nmax == nap else APERT_OUTOFBOUNDS
    if not np.isfinite(sky.value):
        return Apertures(radii, sums, areas, flux, mag, merr, APERT_NOSKYMODE)
    if ier == APERT_OK and nmin <= nmax:
        ier = APERT_BADDATA
    usable = min(nmin - 1, nmax)
    sigma = sky.sigma if np.isfinite(sky.sigma) else None
    for k in range(usable):
        f = sums[k] - areas[k] * sky.value
        flux[k] = f
        if f <= 0:
            continue
        err = (areas[k] * sigma ** 2 if sigma is not None else 0.0) + f / dpars.epadu
        if sky.nsky > 0 and sigma is not None:
            err += sigma ** 2 * areas[k] ** 2 / sky.nsky
        merr[k] = 1.0857 * np.sqrt(err) / f if err > 0 else 0.0
        mag[k] = zmag - 2.5 * np.log10(f) + 2.5 * np.log10(dpars.itime)
    return Apertures(radii, sums, areas, flux, mag, merr, ier)
