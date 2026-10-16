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
centring       centroid (IRAF's default), gauss, ofilter and none, with
               the symmetry clean; gauss through IRAF's own nlfit.
sky            every algorithm (core/fitsky.py): centroid (the default),
               mode, median, mean, gauss, ofilter, crosscor and
               constant, with k-sigma rejection, region growing,
               percentile clips and histogram smoothing. Not the
               interactive histplot and radplot, nor file.
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

import math
from dataclasses import dataclass, field

import numpy as np

f32 = np.float32
CONVERT = 0.424660900          # FWHM to sigma, as apfitcen.x has it
SQRTOFPI = 1.7724538509055160273

# IRAF's error codes, as phot writes them in the CIER, SIER and PIER columns.
# From noao/digiphot/apphot/lib/{center,fitsky,phot}.h.
CTR_OK, CTR_NOAREA, CTR_OUTOFBOUNDS, CTR_LOWSNRATIO = 0, 101, 102, 103
CTR_NTOO_SMALL, CTR_SINGULAR, CTR_NOCONVERGE = 104, 105, 106
CTR_BADSHIFT, CTR_BADDATA = 107, 108
SKY_OK, SKY_NOSKYAREA, SKY_OUTOFBOUNDS, SKY_NOHISTOGRAM = 0, 201, 202, 203
SKY_FLAT_HIST, SKY_NSKY_TOO_SMALL, SKY_SINGULAR, SKY_NOCONVERGE = 204, 205, 206, 207
SKY_BADPARAMS = 212
APERT_OK, APERT_NOAPERT, APERT_OUTOFBOUNDS, APERT_NOSKYMODE = 0, 301, 302, 303
APERT_BADDATA = 305

MESSAGES = {
    CTR_NOAREA: "the centring box is off the image",
    CTR_OUTOFBOUNDS: "the centring box is partly off the image",
    CTR_LOWSNRATIO: "the signal-to-noise in the centring box is below minsnratio",
    CTR_NTOO_SMALL: "the centring box is too small for the fit",
    CTR_SINGULAR: "the centring fit is singular",
    CTR_NOCONVERGE: "the centring fit did not converge",
    CTR_BADDATA: "bad pixels (outside datamin/datamax) in the centring box",
    CTR_BADSHIFT: "the centre moved more than maxshift from where it started",
    SKY_NOSKYAREA: "the sky annulus has no area",
    SKY_OUTOFBOUNDS: "the sky annulus is off the image",
    SKY_NSKY_TOO_SMALL: "too few sky pixels survived rejection",
    SKY_NOHISTOGRAM: "the sky histogram could not be built",
    SKY_FLAT_HIST: "the sky histogram is flat",
    SKY_SINGULAR: "the fit to the sky histogram is singular",
    SKY_NOCONVERGE: "the fit to the sky histogram did not converge",
    SKY_BADPARAMS: "the fit to the sky histogram gave a non-physical answer",
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
    noise: str = "poisson"            #: poisson or constant


@dataclass
class CenterPars:
    """centerpars."""

    calgorithm: str = "centroid"
    cbox: float = 5.0
    cthreshold: float = 0.0
    minsnratio: float = 1.0
    cmaxiter: int = 10
    maxshift: float = 1.0
    clean: bool = False
    rclean: float = 1.0
    rclip: float = 2.0
    kclean: float = 3.0


@dataclass
class SkyPars:
    """fitskypars."""

    salgorithm: str = "centroid"
    annulus: float = 10.0
    dannulus: float = 10.0
    skyvalue: float = 0.0
    smaxiter: int = 10
    snreject: int = 50
    sloreject: float | None = 3.0     #: None is IRAF's INDEF: no lower rejection
    shireject: float | None = 3.0
    khist: float = 3.0
    binsize: float = 0.1
    smooth: bool = False
    sloclip: float = 0.0
    shiclip: float = 0.0
    rgrow: float = 0.0


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
    """apfitcenter: centroid, gauss, ofilter or none."""
    from . import advice

    pars, dpars = pars or CenterPars(), dpars or DataPars()
    if pars.calgorithm == "none":
        return Center(x, y)
    if pars.calgorithm not in ("centroid", "gauss", "ofilter"):
        advice.fail(f"calgorithm '{pars.calgorithm}' is not one of IRAF's",
                    "use centroid (IRAF's default), gauss, ofilter or none")

    ny, nx = data.shape
    cthreshold = (0.0 if dpars.sigma is None or pars.cthreshold <= 0
                  else pars.cthreshold * dpars.sigma)
    capert = pars.cbox / 2.0
    icpix = 2 * _trunc(max(1.0, capert * dpars.scale)) + 1
    half = (icpix - 1) // 2
    # The PSF's sigma for gauss and ofilter, in IRAF's single precision.
    psf_sigma = f32(f32(CONVERT) * f32(dpars.fwhmpsf)) * f32(dpars.scale)

    owx, owy = x, y
    xc_new, yc_new = x, y
    xerr = yerr = float("nan")
    ier = fier = CTR_OK
    lowsnr = False
    niter = 0
    while True:
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

        if pars.clean:
            box = _clean(box.astype(f32), cxc, cyc, pars, dpars).astype(float)

        # The threshold: the box minimum, or the lowest row median when a
        # cthreshold is set; everything at or below it counts for nothing.
        floor = box.min() if cthreshold <= 0 else float(np.min(np.median(box, axis=1)))
        box = np.maximum(box - (floor + cthreshold), 0.0)

        lowsnr = _snratio(box, dpars) < pars.minsnratio

        if pars.calgorithm == "centroid":
            # ap_mctr1d unless a cthreshold is asked for -- even when there is
            # no sigma to turn it into counts, as in IRAF.
            marginal = _cmmarg if pars.cthreshold <= 0 else _cmarg
            xc, xerr = marginal(box.mean(axis=0), dpars.epadu)
            yc, yerr = marginal(box.mean(axis=1), dpars.epadu)
            if np.isnan(xerr):
                xc = cxc
            if np.isnan(yerr):
                yc = cyc
            fier = CTR_OK
        elif pars.calgorithm == "gauss":
            fier, xc, yc, xerr, yerr = _gctr1d(box.astype(f32), psf_sigma, pars.cmaxiter)
        else:
            fier, xc, yc, xerr, yerr = _lgctr1d(box.astype(f32), cxc, cyc, psf_sigma,
                                                pars.cmaxiter, dpars.epadu, dpars.sigma)
        if xc is None:              # the fit did not run: IRAF keeps the last centre
            xc, yc = cxc, cyc

        xc = max(0.5, min(bnx + 0.5, xc))
        yc = max(0.5, min(bny + 0.5, yc))
        xshift, yshift = xc - cxc, yc - cyc
        xc_new, yc_new = xshift + owx, yshift + owy
        owx, owy = xc_new, yc_new
        niter += 1
        if ((fier not in (CTR_OK, CTR_NOCONVERGE)) or niter >= pars.cmaxiter
                or (abs(xshift) < 1.0 and abs(yshift) < 1.0)):
            break

    if fier != CTR_OK:
        return Center(x, y, ier=fier)
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


def _clean(pix: np.ndarray, cxc: float, cyc: float, pars: CenterPars,
           dpars: DataPars) -> np.ndarray:
    """ap_clean: replace a pixel by its mirror image through the brightest
    pixel when it is brighter by more than the noise allows (ap_cclean for
    noise = constant, ap_pclean for poisson). Nothing without a sigma."""
    sigma = dpars.sigma
    if sigma is None or (dpars.noise == "constant" and sigma <= 0):
        return pix
    pix = pix.copy()
    ny, nx = pix.shape
    # ap_2dalimr -- which, as IRAF's does, never looks at the first row.
    ixc = jyc = 1
    mindat = maxdat = pix[0, 0]
    for j in range(2, ny + 1):
        for i in range(1, nx + 1):
            v = pix[j - 1, i - 1]
            if v < mindat:
                mindat = v
            elif v > maxdat:
                maxdat, ixc, jyc = v, i, j
    if abs(cxc - ixc) > pars.maxshift or abs(cyc - jyc) > pars.maxshift:
        ixc, jyc = int(cxc), int(cyc)
    rclean2 = f32(f32(dpars.scale * pars.rclean) ** 2)
    rclip2 = f32(f32(dpars.scale * pars.rclip) ** 2)
    ksigma = f32(f32(pars.kclean) * f32(sigma))
    ksigma2 = f32(ksigma ** 2)
    padu = f32(dpars.epadu)
    poisson = dpars.noise != "constant"
    for j in range(1, ny + 1):
        jj = 2 * jyc - j
        if jj < 1 or jj > ny:
            continue
        for i in range(1, ixc + 1):
            ii = 2 * ixc - i
            if ii < 1 or ii > nx:
                continue
            r2 = (i - ixc) ** 2 + (j - jyc) ** 2
            a, b = pix[j - 1, i - 1], pix[jj - 1, ii - 1]
            if poisson and rclean2 < r2 <= rclip2:
                if b > f32(a + np.sqrt(f32(f32(a / padu) + ksigma2))):
                    pix[jj - 1, ii - 1] = a
                elif a > f32(b + np.sqrt(f32(f32(b / padu) + ksigma2))):
                    pix[j - 1, i - 1] = b
            if r2 > rclip2:
                a, b = pix[j - 1, i - 1], pix[jj - 1, ii - 1]
                if b > f32(a + ksigma):
                    pix[jj - 1, ii - 1] = a
                elif a > f32(b + ksigma):
                    pix[j - 1, i - 1] = b
    return pix


def _marginals(pix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ap_mkmarg: column and row sums, summed in order in single precision."""
    ny, nx = pix.shape
    xm = np.zeros(nx, dtype=f32)
    ym = np.zeros(ny, dtype=f32)
    for j in range(ny):
        xm = (xm + pix[j]).astype(f32)
    for i in range(nx):
        ym = (ym + pix[:, i]).astype(f32)
    return xm, ym


def _expf(v) -> np.float32:
    """A REAL's exp as IRAF's f2c-built code takes it: C's double exp,
    rounded when stored."""
    return f32(math.exp(float(v)))


def _cgauss1d(x, p):
    """cgauss1d: amplitude, centre, variance and a constant."""
    r2 = f32(36.0) if p[2] == 0 else f32(f32(f32(x - p[1]) ** 2) / f32(f32(2.0) * p[2]))
    if abs(r2) > 25.0:
        return p[3]
    # C promotes from the exp on: one rounding, at the store.
    return f32(float(p[0]) * math.exp(-float(r2)) + float(p[3]))


def _cdgauss1d(x, p):
    dx = f32(x - p[1])
    r2 = f32(36.0) if p[2] == 0 else f32(f32(dx * dx) / f32(f32(2.0) * p[2]))
    if abs(r2) > 25.0:
        return p[3], (f32(0), f32(0), f32(0), f32(1))
    d1 = _expf(-r2)
    z = f32(p[0] * d1)
    return f32(z + p[3]), (d1, f32(f32(z * dx) / p[2]), f32(f32(z * r2) / p[2]), f32(1))


def _alimr(a: np.ndarray) -> tuple[np.float32, np.float32, int, int]:
    """ap_alimr: min, max and the 1-based index of each (first occurrence)."""
    return a.min(), a.max(), int(np.argmin(a)) + 1, int(np.argmax(a)) + 1


def _gctr1d(pix: np.ndarray, sigma, maxiter: int):
    """ap_gctr1d: a gaussian of the PSF's sigma, plus a constant, fitted to
    each marginal by IRAF's nlfit."""
    from . import nlfit

    ny, nx = pix.shape
    if nx < 3 or ny < 3:
        return CTR_NTOO_SMALL, None, None, float("nan"), float("nan")
    xm, ym = _marginals(pix)
    # Each marginal is divided by its own length -- unlike the centroid's,
    # which are means.
    xm = (xm / f32(nx)).astype(f32)
    ym = (ym / f32(ny)).astype(f32)
    out, stats = [], []
    for m, n in ((xm, nx), (ym, ny)):
        lo, hi, _, imax = _alimr(m)
        p = [f32(hi - lo), f32(imax), f32(sigma ** 2), lo]
        fit = nlfit.NLFit(_cgauss1d, _cdgauss1d, p, [0, 1, 3], 0.001, maxiter)
        xs = np.arange(1, n + 1, dtype=f32)
        stats.append(fit.fit(xs, m))
        dp = fit.errors(m, fit.vector(xs))
        err = float(f32(float(dp[1]) / math.sqrt(n)))
        out += [float(fit.param[1]), err if err <= n else float("nan")]
    if nlfit.NO_DEG_FREEDOM in stats:
        fier = CTR_NTOO_SMALL
    elif nlfit.SINGULAR in stats:
        fier = CTR_SINGULAR
    elif nlfit.NOT_DONE in stats:
        fier = CTR_NOCONVERGE
    else:
        fier = CTR_OK
    return fier, out[0], out[2], out[1], out[3]


def _lgctr1d(pix: np.ndarray, cx: float, cy: float, sigma, maxiter: int, norm: float,
             skysigma: float | None):
    """ap_lgctr1d: optimal filtering of each marginal with a triangle the
    width of the PSF (aptopt)."""
    ny, nx = pix.shape
    xm, ym = _marginals(pix)
    results = []
    iters = []
    for m, n, c in ((xm, nx, cx), (ym, ny, cy)):
        total = _asum(m)
        m = (m / f32(n)).astype(f32)
        niter, centre = aptopt(m, f32(c), sigma, 0.001, maxiter, True)
        iters.append(niter)
        if total <= 0:
            err = float("nan")
        else:
            # 4 sqrt(pi), folded by f2c into one single precision constant.
            constant = (f32(0.0) if skysigma is None else
                        f32(f32(sigma * f32(4.0 * SQRTOFPI)) * f32(f32(skysigma) ** 2)))
            ratio = f32(constant / total)
            spread = f32(f32(sigma ** 2) / f32(total * f32(norm)))
            err = float(f32(math.sqrt(float(max(spread, f32(ratio * spread))))))
            if err > n:
                err = float("nan")
        results += [float(centre), err]
    if min(iters) < 0:
        fier = CTR_SINGULAR
    elif max(iters) > maxiter:
        fier = CTR_NOCONVERGE
    else:
        fier = CTR_OK
    return fier, results[0], results[2], results[1], results[3]


def _asum(a) -> np.float32:
    """asumr: a sum in order, in single precision."""
    total = f32(0.0)
    for v in a:
        total = f32(total + v)
    return total


def _adot(a, b) -> np.float32:
    total = f32(0.0)
    for u, v in zip(a, b, strict=True):
        total = f32(total + f32(u * v))
    return total


def _profile_weight(npix: int, centre, sigma, ortho: bool) -> np.ndarray:
    """mkt_prof_derv: the derivative of a triangle of the PSF's width,
    optionally made orthogonal to a constant and to the triangle itself."""
    width = f32(sigma * f32(2.35))
    x = ((np.arange(1, npix + 1, dtype=f32) - f32(centre)) / width).astype(f32)
    tri = np.where(np.abs(x) <= 1.0, f32(1.0) - np.abs(x), f32(0.0)).astype(f32)
    weight = (x * tri).astype(f32)
    if ortho:
        weight = (weight + f32(-_asum(weight) / f32(npix))).astype(f32)
        tri = (tri + f32(-_asum(tri) / f32(npix))).astype(f32)
        coef = _adot(tri, tri)
        coef = f32(1.0) if coef <= 0 else f32(_adot(weight, tri) / coef)
        weight = (weight - (tri * coef).astype(f32)).astype(f32)
        coef = _adot(weight, weight)
        coef = f32(1.0) if coef <= 0 else np.sqrt(f32(f32(1.0) / coef))
        weight = (weight * coef).astype(f32)
    return weight


def _qzero(x, y):
    """apqzero: the root of the parabola through three points, from x[0]."""
    x2, x3 = f32(x[1] - x[0]), f32(x[2] - x[0])
    y2, y3 = f32(y[1] - y[0]), f32(y[2] - y[0])
    det = f32(f32(x2 * x3) * f32(x2 - x3))
    if abs(det) > 0.0:
        a = f32(f32(f32(x3 * y2) - f32(x2 * y3)) / det)
        b = f32(-f32(f32(f32(x3 * x3) * y2) - f32(f32(x2 * x2) * y3)) / det)
        c = f32(f32(a * y[0]) / f32(b * b))
        if abs(c) > 0.125:
            root = float(f32(f32(1.0) - f32(f32(4.0) * c)))
            root = math.sqrt(root) if root >= 0 else float("nan")
            return f32(float(f32(-b / f32(f32(2.0) * a))) * (1.0 - root))
        return f32(f32(-(y[0] / b)) * f32(f32(1.0) + c))
    if abs(y3) > 0.0:
        return f32(f32(-y[0] * x3) / y3)
    return f32(0.0)


def aptopt(data: np.ndarray, centre, sigma, tol: float, maxiter: int,
           ortho: bool) -> tuple[int, np.float32]:
    """aptopt: the zero of the data's correlation with the derivative of a
    triangle, bracketed and then found by repeated parabolas. Returns the
    iteration count (-1 when it cannot be bracketed, > maxiter when it did
    not converge) and the centre."""
    data = np.asarray(data, dtype=f32)
    npix = data.size
    sigma = f32(sigma)
    gap = f32(tol)
    if sigma <= 0:
        return -1, f32(centre)
    x = [f32(centre), f32(0), f32(0)]
    s = [f32(0), f32(0), f32(0)]
    s[0] = _adot(_profile_weight(npix, x[0], sigma, ortho), data)
    if s[0] == 0.0:
        return 0, x[0]
    s[2] = s[0]
    i = 1
    while s[2] * s[0] >= 0.0 and i <= 3:
        s[2], x[2] = s[0], x[0]
        x[0] = f32(x[2] + (sigma if s[2] >= 0 else -sigma))
        s[0] = _adot(_profile_weight(npix, x[0], sigma, ortho), data)
        if s[0] == 0.0:
            return 0, x[0]
        i += 1
    if s[2] * s[0] > 0.0:
        return -1, f32(centre)
    delx = f32(x[0] - x[2])
    x[1] = f32(x[2] - f32(f32(s[2] * delx) / f32(s[0] - s[2])))
    s[1] = _adot(_profile_weight(npix, x[1], sigma, ortho), data)
    if s[1] == 0.0:
        return 1, x[1]
    iteration = 2
    while iteration <= maxiter:
        if s[1] == 0.0:
            break
        if abs(f32(x[1] - x[0])) <= gap or abs(f32(x[2] - x[1])) <= gap:
            break
        newx = f32(x[0] + _qzero(x, s))
        news = _adot(_profile_weight(npix, newx, sigma, ortho), data)
        if s[0] * s[1] > 0.0:
            s[0], x[0] = s[1], x[1]
        else:
            s[2], x[2] = s[1], x[1]
        s[1], x[1] = news, newx
        iteration += 1
    return iteration, x[1]


def _snratio(box: np.ndarray, dpars: DataPars) -> float:
    """ap_csnratio, with the sky taken as zero: ap_pratio for the poisson
    noise model, ap_cratio (the sky's sigma alone) for constant."""
    signal = float(box.sum())
    noise2 = 0.0 if dpars.noise == "constant" else abs(signal / dpars.epadu)
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


def fitsky(data: np.ndarray, x: float, y: float, pars: SkyPars | None = None,
           dpars: DataPars | None = None) -> Sky:
    """apfitsky: every algorithm of IRAF's fitskypars but the interactive
    ones (histplot, radplot) and file. See core/fitsky.py."""
    from . import advice, fitsky

    pars, dpars = pars or SkyPars(), dpars or DataPars()
    if pars.salgorithm not in fitsky.ALGORITHMS:
        advice.fail(f"salgorithm '{pars.salgorithm}' is not available",
                    "use " + ", ".join(fitsky.ALGORITHMS[:-1]) + " or constant; "
                    "histplot and radplot are IRAF's interactive markers, and file "
                    "reads the sky from a list -- give it as constant per star")
    return fitsky.fitsky(data, x, y, pars, dpars)


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
