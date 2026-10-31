# ruff: noqa: E741
"""IRAF's daofind (noao.digiphot.apphot), ported.

From iraf-community 2.18: apegkernel.x (the kernel), apconvolve.x (the
density image), apfind.x (detection, sharpness, the two roundnesses and the
centres), apwres1.x (the magnitude written out). harwood used photutils'
DAOStarFinder, which is DAOPHOT's FIND rather than apphot's daofind; on the
1987 M92 frame it found 139 stars where IRAF finds 162, the difference
being the threshold scaling and the edge.

In IRAF's order:

1. an elliptical gaussian of sigma = 0.42467 fwhmpsf, cut at nsigma, is
   turned into a kernel that measures the amplitude of a gaussian fitted to
   the pixels under it (its mean subtracted, divided by the sum of squares);
2. the image, extended at its edges by the nearest pixel (boundary =
   nearest), is correlated with it: the density image;
3. a pixel is a star if its density is at least relerr * threshold * sigma
   and no other pixel under the kernel exceeds it; after one is found the
   next nxk/2 columns of that line are skipped;
4. sharpness is (central pixel - mean of the others) / density; sround
   compares the density's two-fold and four-fold symmetry; ground compares
   the heights of gaussians fitted to the x and y marginals, which also give
   the centre;
5. a star is kept within sharplo..sharphi and with both roundnesses within
   roundlo..roundhi (ground is not tested on a saturated star), and its
   magnitude is -2.5 log10(density / (relerr * threshold)), zero at the
   detection limit.

IRAF reads and computes in single precision; so does this, where it
matters for which pixel is the local maximum.

Positions are 1-based (x, y). Translated from IRAF, Copyright (c) 1986
Association of Universities for Research in Astronomy Inc.; the notice it is
distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .nlfit import square

FWHM_TO_SIGMA = 0.42467
RMIN = 2.001
MAX_REAL = 3.4028235e38


@dataclass
class Kernel:
    gauss: np.ndarray        #: the gaussian, everywhere in the box (gker2d)
    normal: np.ndarray       #: the amplitude-fitting kernel (ngker2d)
    skip: np.ndarray         #: True outside nsigma
    relerr: float


def kernel(fwhmpsf: float, ratio: float = 1.0, theta: float = 0.0,
           nsigma: float = 1.5, scale: float = 1.0) -> Kernel:
    """ap_egparams and ap_egkernel."""
    f32 = np.float32
    sigma = f32(FWHM_TO_SIGMA) * f32(fwhmpsf) * f32(scale)
    sx2 = sigma * sigma
    sy2 = square(f32(ratio) * sigma)
    cost, sint = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    f = nsigma ** 2 / 2.0
    a: float
    b: float
    c: float
    if ratio == 0:
        if theta in (0.0, 180.0):
            a, b, c = float(1.0 / sx2), 0.0, 0.0
        elif theta == 90.0:
            a, b, c = 0.0, 0.0, float(1.0 / sx2)
        else:
            raise ValueError("a 1-D gaussian kernel needs theta of 0, 90 or 180")
        nx = 2 * int(max(sigma * nsigma * abs(cost), RMIN)) + 1
        ny = 2 * int(max(sigma * nsigma * abs(sint), RMIN)) + 1
    else:
        a = cost ** 2 / sx2 + sint ** 2 / sy2
        b = 2.0 * (1.0 / sx2 - 1.0 / sy2) * cost * sint
        c = sint ** 2 / sx2 + cost ** 2 / sy2
        discrim = b ** 2 - 4.0 * a * c
        nx = 2 * int(max(np.sqrt(-8.0 * c * f / discrim), RMIN)) + 1
        ny = 2 * int(max(np.sqrt(-8.0 * a * f / discrim), RMIN)) + 1

    x0, y0 = nx // 2 + 1, ny // 2 + 1
    j, i = np.mgrid[1:ny + 1, 1:nx + 1].astype(float)
    x, y = i - x0, j - y0
    rsq = np.sqrt(x ** 2 + y ** 2)
    ef = 0.5 * (a * x ** 2 + c * y ** 2 + b * x * y)
    gauss = np.exp(-ef).astype(np.float32)
    inside = (ef <= f) | (rsq <= RMIN)
    g = gauss.astype(np.float64)
    npts = inside.sum()
    sumg, sumgsq = g[inside].sum(), (g[inside] ** 2).sum()
    denom = sumgsq - sumg ** 2 / npts
    normal = np.where(inside, (g - sumg / npts) / denom, 0.0).astype(np.float32)
    return Kernel(gauss, normal, ~inside, float(np.sqrt(1.0 / denom)))


def density(data: np.ndarray, k: Kernel) -> np.ndarray:
    """The density image: the data, nearest-extended, correlated with the
    amplitude kernel."""
    ny, nx = data.shape
    nyk, nxk = k.normal.shape
    padded = np.pad(np.asarray(data, dtype=np.float32), ((nyk // 2, nyk // 2),
                                                         (nxk // 2, nxk // 2)),
                    mode="edge")
    out = np.zeros((ny, nx), dtype=np.float32)
    for j in range(nyk):
        for i in range(nxk):
            if not k.skip[j, i]:
                out += k.normal[j, i] * padded[j:j + ny, i:i + nx]
    return out


def find(data: np.ndarray, fwhmpsf: float, sigma: float | None, threshold: float = 4.0,
         nsigma: float = 1.5, ratio: float = 1.0, theta: float = 0.0,
         sharplo: float = 0.2, sharphi: float = 1.0, roundlo: float = -1.0,
         roundhi: float = 1.0, epadu: float = 1.0, readnoise: float = 0.0,
         scale: float = 1.0, refit: bool = False) -> dict:
    """ap_bfdfind + ap_find: the star list, in IRAF's order (line by line).

    Returns arrays x, y, mag, sharpness, sround, ground and the relerr and
    absolute threshold used. `refit`: ap_find has run on this kernel already, in
    an interactive session; it marked the centre as skipped then, so it counts
    one pixel fewer now, and every sharpness comes out differently.
    """
    k = kernel(fwhmpsf, ratio, theta, nsigma, scale)
    data = np.asarray(data, dtype=np.float32)
    ny, nx = data.shape
    nyk, nxk = k.normal.shape
    dens = density(data, k)
    if sigma is None:
        skymode, thresh = 0.0, 0.0
    else:
        skymode = max(0.0, epadu * (sigma ** 2 - (readnoise / epadu) ** 2))
        thresh = threshold * sigma
    limit = np.float32(k.relerr * thresh)

    xh, yh = nxk // 2, nyk // 2
    dpad = np.pad(data, ((yh, yh), (xh, xh)), mode="edge")
    cpad = np.pad(dens, ((yh, yh), (xh, xh)), mode="edge")
    skip = k.skip.copy()
    pixels = int((~skip).sum()) - 1
    nonzero = pixels - 1 if refit else pixels
    skip_centre = skip.copy()
    skip_centre[yh, xh] = True
    xsigsq = (scale * fwhmpsf / 2.35482) ** 2
    ysigsq = (scale * ratio * fwhmpsf / 2.35482) ** 2
    gk = k.gauss.astype(np.float64)

    out: dict[str, list[float]] = {
        name: [] for name in ("x", "y", "mag", "sharpness", "sround", "ground")}
    for line in range(ny):
        row = dens[line]
        candidates = np.flatnonzero(row >= limit)
        if candidates.size == 0:
            continue
        cols = []
        i = 0
        while i < nx:
            if row[i] < limit:
                i += 1
                continue
            box = cpad[line:line + nyk, i:i + nxk]
            if np.any(box[~skip_centre] > row[i]):
                i += 1
                continue
            cols.append(i)
            i += xh + 1
        for c in cols:
            dbox = dpad[line:line + nyk, c:c + nxk].astype(np.float64)
            cbox = cpad[line:line + nyk, c:c + nxk].astype(np.float64)
            centre = cbox[yh, xh]
            # sround: two-fold against four-fold symmetry of the density.
            nhalf = min(xh, yh)
            sum2 = sum4 = 0.0
            for kk in range(nhalf + 1):
                for jj in range(1, nhalf + 1):
                    v = (cbox[yh - jj, xh - kk], cbox[yh + jj, xh + kk],
                         cbox[yh + kk, xh - jj], cbox[yh - kk, xh + jj])
                    sum2 += v[0] + v[1] - v[2] - v[3]
                    sum4 += abs(v[0]) + abs(v[1]) + abs(v[2]) + abs(v[3])
            sround = 0.0 if sum2 == 0 else (np.nan if sum4 <= 0 else 2.0 * sum2 / sum4)
            # sharpness
            midpix = dbox[yh, xh]
            others = dbox[~skip_centre].sum() - pixels * skymode
            sharp = (np.nan if centre <= 0 or nonzero <= 0
                     else (midpix - skymode - others / nonzero) / centre)
            # ground and the centre, from the marginals
            fitted = _marginals(dbox, gk, skymode, xh, yh, xsigsq, ysigsq)
            if fitted is None:
                continue
            dx, dy, ground = fitted
            x = (c + 1) + dx
            y = (line + 1) + dy
            if not np.isnan(sharp) and (sharp < sharplo or sharp > sharphi):
                continue
            if np.isnan(sround) or sround < roundlo or sround > roundhi:
                continue
            if np.isnan(ground) or ground < roundlo or ground > roundhi:
                continue
            if not (0.5 <= x <= nx + 0.5 and 0.5 <= y <= ny + 0.5):
                continue
            out["x"].append(float(np.float32(x)))
            out["y"].append(float(np.float32(y)))
            out["mag"].append(float(-2.5 * np.log10(centre / limit)) if limit > 0
                              else float("nan"))
            out["sharpness"].append(float(np.float32(sharp)))
            out["sround"].append(float(np.float32(sround)))
            out["ground"].append(float(np.float32(ground)))
    result: dict[str, Any] = {name: np.array(values) for name, values in out.items()}
    result["relerr"] = k.relerr
    result["threshold"] = float(limit)
    return result


def _marginals(dbox, gk, skymode, xh, yh, xsigsq, ysigsq):
    """ap_xy_round for one star: heights and centres of gaussians fitted to
    the weighted x and y marginals. None when either height is not positive."""
    nyk, nxk = dbox.shape
    xmiddle, ymiddle = xh + 1, yh + 1
    wy = (ymiddle - np.abs(np.arange(1, nyk + 1) - ymiddle)).astype(float)
    wx = (xmiddle - np.abs(np.arange(1, nxk + 1) - xmiddle)).astype(float)

    def fit(sd, sg, w, offsets, sigsq, half, fallback_sign):
        keep = sg > 0
        if keep.sum() <= 2:
            return None
        sd, sg, w, off = sd[keep], sg[keep], w[keep], offsets[keep]
        p = w.sum()
        if p <= 0:
            return None
        sumgd, sumgsq = (w * sg * sd).sum(), (w * sg ** 2).sum()
        sumg, sumd = (w * sg).sum(), (w * sd).sum()
        dgdx = sg * off
        sdgdxsq, sdgdx = (w * dgdx ** 2).sum(), (w * dgdx).sum()
        sddgdx, sgdgdx = (w * sd * dgdx).sum(), (w * sg * dgdx).sum()
        h = sumgsq - sumg ** 2 / p
        if h <= 0:
            return None
        h = (sumgd - sumg * sumd / p) / h
        if h <= 0:
            return None
        sky = (sumd - h * sumg) / p
        d = (sgdgdx - (sddgdx - sdgdx * (h * sumg + sky * p))) / (h * sdgdxsq / sigsq)
        if abs(d) > half:
            # IRAF's fallback centroid: +offset along x, -offset along y.
            sumdx = (w * sd * fallback_sign * off).sum()
            d = 0.0 if sumd == 0 else sumdx / sumd
            if abs(d) > half:
                d = 0.0
        return h, d

    signal = dbox - skymode
    # x: columns, each summed over lines weighted by wy
    sdx = (signal * wy[:, None]).sum(axis=0)
    sgx = (gk * wy[:, None]).sum(axis=0)
    xfit = fit(sdx, sgx, wx, xmiddle - np.arange(1, nxk + 1), xsigsq, xh + 0.5, 1.0)
    if xfit is None:
        return None
    sdy = (signal * wx[None, :]).sum(axis=1)
    sgy = (gk * wx[None, :]).sum(axis=1)
    yfit = fit(sdy, sgy, wy, ymiddle - np.arange(1, nyk + 1), ysigsq, yh + 0.5, -1.0)
    if yfit is None:
        return None
    hx, dx = xfit
    hy, dy = yfit
    return dx, dy, 2.0 * (hx - hy) / (hx + hy)
