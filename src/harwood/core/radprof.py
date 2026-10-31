"""IRAF's radprof (noao.digiphot.apphot), ported.

From iraf-community 2.18, apphot/radprof: apfrprof.x (the fit, the rejection and
the half width), aprpbuf.x (the pixels), aprmmeasure.x (the sums). Every pixel
within `radius` of the centre, less the sky, is a point of intensity against
distance; a cubic spline of `order` pieces is fitted through them, pixels beyond
kreject times the rms are taken out and the fit done again; and the curve is what
is written, normalised to its value at the centre, with its integral outwards
normalised to its value at the radius, and the radius at which it crosses half
doubled as the FWHM.

The quirks kept. The error code of a failed fit is never reset: a star after one
says so too. A star whose box runs off the image is fitted on the pixels there are,
and nothing says so, the out-of-bounds code being looked at only for NOPROFILE. The
apertures are summed over the profile's box and no further, so an aperture larger
than `radius` is short of light, and only the image's edge makes that an error.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import apphot
from .asi import Interpolant
from .curfit import NO_DEG_FREEDOM, SINGULAR, Curve

f32 = np.float32

OK, NOPROFILE, OUTOFBOUNDS, NPTS_TOO_SMALL, RP_SINGULAR, NOSKYMODE = 0, 901, 902, 903, 904, 905
MESSAGES = {OK: "NoError", NOPROFILE: "OffImage", OUTOFBOUNDS: "EdgeImage",
            NPTS_TOO_SMALL: "TooFewPts", RP_SINGULAR: "BadFit", NOSKYMODE: "NoSky"}


@dataclass
class Profile:
    """One star's profile: what ap_frprof leaves."""
    fwhm: float = float("nan")
    inorm: float = float("nan")
    tinorm: float = float("nan")
    ier: int = OK
    radius: np.ndarray = field(default_factory=lambda: np.zeros(0))
    intensity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tintensity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ndata: int = 0
    nreject: int = 0
    measured: apphot.Apertures | None = None


def _half(radius: np.ndarray, intensity: np.ndarray) -> f32:
    """ap_rphalf."""
    npts = len(radius)
    i = next((k for k in range(npts) if intensity[k] < 0.5), npts - 1)
    if i == 0:
        return f32(radius[0])
    if i == npts - 1 and intensity[npts - 1] >= 0.5:
        return f32(radius[npts - 1])
    return f32(f32(f32(radius[i] * f32(f32(0.5) - intensity[i - 1]))
                   + f32(radius[i - 1] * f32(intensity[i] - f32(0.5))))
               / f32(intensity[i] - intensity[i - 1]))


class Fitter:
    """ap_frprof over a list of stars: one, because its error code outlives the
    star it belongs to."""

    def __init__(self, radius: float, step: float, order: int = 5, kreject: float = 3.0,
                 nreject: int = 1) -> None:
        self.radius, self.step = f32(radius), f32(step)
        self.order, self.kreject, self.nreject = order, f32(kreject), nreject
        self.npts = int(f32(self.radius / self.step)) + 1
        self.ier = OK

    def fit(self, data: np.ndarray, x: float, y: float, sky: apphot.Sky, radii: list[float],
            zmag: float, dpars: apphot.DataPars) -> Profile:
        profile = Profile()
        if np.isnan(x) or np.isnan(y):
            profile.ier = NOPROFILE
            return profile
        if not np.isfinite(sky.value):
            profile.ier = NOSKYMODE
            return profile
        box = self._box(data, x, y, dpars.scale)
        if box is None:
            profile.ier = NOPROFILE
            return profile
        pixels, xc, yc = box
        profile.measured = _measure(pixels, xc, yc, x, y, data.shape, radii, sky, zmag, dpars)
        self._fit(pixels, xc, yc, sky, dpars, profile)
        profile.ier = self.ier
        return profile

    def _box(self, data, x, y, scale):
        """ap_rpbuf and ap_rppix: the subraster and the centre in it."""
        half = f32(f32(f32(f32(f32(2.0) * self.radius) * f32(scale)) + f32(1.0)) / f32(2.0))
        ny, nx = data.shape
        wx, wy = f32(x), f32(y)
        xc1, xc2 = f32(wx - half), f32(wx + half)
        yl1, yl2 = f32(wy - half), f32(wy + half)
        if xc1 > nx or xc2 < 1.0 or yl1 > ny or yl2 < 1.0:
            return None
        c1 = int(max(f32(1.0), min(f32(nx), xc1)))
        c2 = int(min(f32(nx), max(f32(1.0), xc2)))
        l1 = int(max(f32(1.0), min(f32(ny), yl1)))
        l2 = int(min(f32(ny), max(f32(1.0), yl2)))
        pixels = np.asarray(data[l1 - 1:l2, c1 - 1:c2], dtype=f32)
        return pixels, f32(f32(wx - f32(c1)) + f32(1.0)), f32(f32(wy - f32(l1)) + f32(1.0))

    def _fit(self, pixels, xc, yc, sky, dpars, profile: Profile) -> None:
        step = f32(f32(dpars.scale) * self.step)
        ny, nx = pixels.shape
        order = max(1, min(self.order, nx * ny - 3))
        rmax = f32(f32(self.npts - 1) * step)
        low = -np.finfo(f32).max if dpars.datamin is None else f32(f32(dpars.datamin)
                                                                   - f32(sky.value))
        high = np.finfo(f32).max if dpars.datamax is None else f32(f32(dpars.datamax)
                                                                   - f32(sky.value))
        pixels = (pixels - f32(sky.value)).astype(f32)
        jj, ii = np.mgrid[1:ny + 1, 1:nx + 1]
        dy2 = ((jj.astype(f32) - yc).astype(f32) ** 2).astype(f32)
        dx2 = ((ii.astype(f32) - xc).astype(f32) ** 2).astype(f32)
        r = np.sqrt(((dx2 + dy2).astype(f32)).astype(np.float64)).astype(f32)

        # ap_rpmeasure, a point at a time: the order of the sums is IRAF's.
        curve = Curve("spline3", order, f32(0.0), rmax)
        one = np.ones(1, f32)
        use = r <= rmax
        good = use & (pixels >= low) & (pixels <= high)
        for j, i in zip(*np.nonzero(good), strict=True):
            curve.accumulate(r[j, i:i + 1], pixels[j, i:i + 1], one)
        ndata, nbad = int(good.sum()), int((use & ~good).sum())
        fier = curve.solve()

        rejected = 0
        if fier != NO_DEG_FREEDOM and self.nreject > 0 and self.kreject > 0:
            rejected, fier = self._reject(curve, pixels, r, good, fier)
        profile.ndata = ndata - rejected
        profile.nreject = rejected + nbad
        if fier == NO_DEG_FREEDOM:
            self.ier = NPTS_TOO_SMALL
            return
        if fier == SINGULAR:
            self.ier = RP_SINGULAR

        radius = (np.arange(self.npts, dtype=f32) * step).astype(f32)
        intensity = np.asarray(curve.vector_at(radius), dtype=f32)
        spline = Interpolant("spline3", (radius * intensity).astype(f32))
        tintensity = np.zeros(self.npts, dtype=f32)
        for k in range(1, self.npts):
            piece = f32(spline.integral(float(k), float(k + 1)))
            tintensity[k] = f32(tintensity[k - 1] + piece)
        tintensity = (tintensity * f32(2.0 * np.pi) * step).astype(f32)
        inorm = curve(f32(0.0))
        if inorm != 0:
            intensity = (intensity / inorm).astype(f32)
        tinorm = tintensity[-1]
        if tinorm != 0:
            tintensity = (tintensity / tinorm).astype(f32)
        half = _half(radius, intensity)
        profile.fwhm = float(f32(f32(f32(2.0) * half) / f32(dpars.scale)))
        profile.inorm, profile.tinorm = float(inorm), float(tinorm)
        profile.radius = (radius / f32(dpars.scale)).astype(f32)
        profile.intensity, profile.tintensity = intensity, tintensity

    def _reject(self, curve, pixels, r, good, fier):
        """ap_rpiter."""
        keep = good.copy()
        total = 0
        for _ in range(self.nreject):
            chisqr, npts = f32(0.0), 0
            for j, i in zip(*np.nonzero(keep), strict=True):
                diff = f32(pixels[j, i] - curve(r[j, i]))
                chisqr = f32(chisqr + f32(diff * diff))
                npts += 1
            chisqr = (np.sqrt(f32(chisqr / f32(npts - 1))) if npts > 1 else f32(0.0))
            hicut = f32(self.kreject * chisqr)
            locut = -hicut
            nreject = 0
            for j, i in zip(*np.nonzero(keep), strict=True):
                diff = f32(pixels[j, i] - curve(r[j, i]))
                if locut <= diff <= hicut:
                    continue
                curve.reject(r[j, i:i + 1], pixels[j, i:i + 1], np.ones(1, f32))
                nreject += 1
                keep[j, i] = False
            if nreject == 0:
                break
            total += nreject
            fier = curve.solve()
            if fier == NO_DEG_FREEDOM:
                break
        return total, fier


def _measure(pixels, xc, yc, x, y, shape, radii, sky, zmag, dpars) -> apphot.Apertures:
    """ap_rmag and ap_rmmeasure: the sums over the box, whatever the aperture."""
    radii = sorted(float(v) for v in radii)
    scaled = np.array(radii) * dpars.scale
    nap = len(radii)
    ny, nx = shape
    nmax = 0
    for k in range(nap, 0, -1):
        half = scaled[k - 1]
        if (x - half >= 0.5 and x + half <= nx + 0.5
                and y - half >= 0.5 and y + half <= ny + 0.5):
            nmax = k
            break
    sums, areas = np.zeros(nap), np.zeros(nap)
    if nmax == 0:
        return apphot.magnitudes(radii, sums, areas, sky, zmag, dpars, apphot.APERT_NOAPERT,
                                 nmax, 1)
    box = pixels.astype(np.float64)
    jj, ii = np.mgrid[1:pixels.shape[0] + 1, 1:pixels.shape[1] + 1]
    dx = (ii.astype(f32) - xc).astype(f32)
    dy = (jj.astype(f32) - yc).astype(f32)
    r2 = ((dx * dx).astype(f32) + (dy * dy).astype(f32)).astype(f32)
    r = (np.sqrt(r2.astype(np.float64)) - 0.5).astype(f32)
    top = f32(f32(scaled[nmax - 1]) + f32(0.5))
    within = r2 <= f32(top * top)
    low = -np.inf if dpars.datamin is None else dpars.datamin
    high = np.inf if dpars.datamax is None else dpars.datamax
    nmin = nmax + 1
    for k in range(nmax):
        aperture = f32(scaled[k])
        weight = (np.clip((aperture - r).astype(f32), 0.0, 1.0)
                  * ((r <= aperture) & within)).astype(np.float64)
        sums[k] = float((weight * box).sum())
        areas[k] = float(weight.sum())
        inside = box[weight > 0]
        if nmin > nmax and ((inside < low) | (inside > high)).any():
            nmin = k + 1
    ier = apphot.APERT_OK if nmax == nap else apphot.APERT_OUTOFBOUNDS
    if ier == apphot.APERT_OK and nmin <= nmax:
        ier = apphot.APERT_BADDATA
    return apphot.magnitudes(radii, sums, areas, sky, zmag, dpars, ier, nmax, nmin)
