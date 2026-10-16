"""The measurements behind the imexamine keys -- no Qt in this file.

Section 8.2 keeps the analysis out of the window so that the same numbers come
out whether they were asked for by a keystroke, by a task, or by a test.
`core.examine` is what the keys call; this is the arithmetic underneath it.

`radial_profile` is a port of IRAF's `ie_rimexam`
(`pkg/images/tv/imexamine/ierimexam.x`, iraf-community 2.18), which serves
both the r and the a key, so that a FWHM read off harwood means what one read
off IRAF meant. In IRAF's order:

1. recentre on the marginal distributions above their mean (`ie_center`);
2. call starfocus for the direct FWHM and the FWHM of a fit to the enclosed
   flux -- and, with `iterations > 1`, to reset the aperture to three times
   the direct FWHM (`core.starfocus`);
3. take the median of the annulus from `radius + buffer` to
   `radius + buffer + width` as the sky;
4. sum the sky-subtracted pixels within `radius` for the flux, and their
   second moments for the ellipticity and position angle;
5. fit a gaussian or Moffat of fixed centre to the pixels within both
   `radius` and `rplot`, weighted by exp(-r3^2) / max(0.1, r^2) with
   r3 = clip(2r / dfwhm - 1, 0, 5): the core counts as much as the many
   pixels in the wings, and the far wings hardly at all.

One deliberate difference: IRAF prints as SKY the sky summed over the
aperture divided by the pixels within min(radius, rplot), which exceeds the
sky whenever the aperture is wider than rplot. harwood reports the median.

Positions in and out are 1-based (x, y), as everywhere a user can see.

The ie_rimexam port is translated from IRAF, Copyright (c) 1986
Association of Universities for Research in Astronomy Inc.; the notice it
is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core import pixels


@dataclass
class RadialProfile:
    """What the r and a keys measure at a position."""

    x: float
    y: float
    radii: np.ndarray = field(repr=False)     #: plotted points, out to rplot
    values: np.ndarray = field(repr=False)
    radius: float = 5.0                  #: the aperture actually used
    sky: float = 0.0
    #: Scatter of the sky annulus, robust (1.4826 x the MAD).
    noise: float = 0.0
    peak: float = 0.0                    #: amplitude of the fit
    fwhm: float = float("nan")           #: from the fit to the pixels
    enclosed_fwhm: float = float("nan")  #: from the fit to the enclosed flux
    direct_fwhm: float = float("nan")    #: from the radial profile, no model
    beta: float = float("nan")           #: Moffat only
    flux: float = 0.0
    area: int = 0
    ellipticity: float = float("nan")
    pa: float = float("nan")             #: degrees, -90 to 90, 0 along x
    fit: np.ndarray | None = field(default=None, repr=False)
    fit_radii: np.ndarray | None = field(default=None, repr=False)
    fittype: str = "moffat"
    converged: bool = False
    message: str = ""

    @property
    def total(self) -> float:
        """Flux above sky inside the aperture."""
        return self.flux

    def magnitude(self, magzero: float = 25.0) -> float:
        return magzero - 2.5 * np.log10(self.flux) if self.flux > 0 else float("nan")


def centroid(data: np.ndarray, x: float, y: float, radius: float = 5.0) -> tuple[float, float]:
    """IRAF's ie_center: moments of the marginal distributions.

    A box from x - radius to x + radius; of each marginal, only the values
    above its mean count, which is what keeps a sloping sky from pulling the
    centre. Repeated, up to three times, while the centre changes pixel -- a
    cursor dropped a couple of pixels off a star still finds it.
    """
    ny, nx = data.shape
    for _ in range(3):
        xlast, ylast = x, y
        x1, x2 = max(1, int(x - radius + 0.5)), min(nx, int(x + radius + 0.5))
        y1, y2 = max(1, int(y - radius + 0.5)), min(ny, int(y + radius + 0.5))
        if x2 < x1 or y2 < y1:
            return xlast, ylast
        box = data[y1 - 1:y2, x1 - 1:x2]
        if not np.isfinite(box).all():
            # A blank (NaN) pixel in the box: IRAF has none, so there is no
            # rule to follow. Leaving the position alone is the honest answer;
            # the measurement that follows says the data were bad.
            return xlast, ylast
        total = box.sum()
        columns = np.clip(box.sum(axis=0) - total / box.shape[1], 0, None)
        rows = np.clip(box.sum(axis=1) - total / box.shape[0], 0, None)
        if columns.sum() <= 0 or rows.sum() <= 0:
            return xlast, ylast
        x = float((columns * np.arange(x1, x2 + 1)).sum() / columns.sum())
        y = float((rows * np.arange(y1, y2 + 1)).sum() / rows.sum())
        if int(x) == int(xlast) and int(y) == int(ylast):
            break
    return x, y


def radial_profile(
    data: np.ndarray,
    x: float,
    y: float,
    radius: float = 5.0,
    buffer: float = 5.0,
    width: float = 5.0,
    rplot: float = 8.0,
    fittype: str = "moffat",
    beta: float | None = None,
    iterations: int = 3,
    recentre: bool = True,
    background: bool = True,
    integer: bool = False,
) -> RadialProfile:
    """Everything IRAF's r and a keys report about the star near (x, y)."""
    from ..core import starfocus

    data = np.asarray(data, dtype=float)
    if recentre:
        x, y = centroid(data, x, y, radius=radius)
    if not (0.5 <= x < data.shape[1] + 0.5 and 0.5 <= y < data.shape[0] + 0.5):
        return RadialProfile(x, y, np.array([]), np.array([]), radius=radius,
                             fittype=fittype, message="that position is outside the image")

    try:
        star = starfocus.measure(data, x, y, radius=radius, buffer=buffer, width=width,
                                 iterations=iterations, beta=beta, integer=integer)
        radius, dfwhm = star.radius, star.dfwhm
        enclosed = star.mfwhm if fittype == "moffat" else star.gfwhm
    except Exception as error:               # no star: say so below, with the sky
        star, dfwhm, enclosed = None, float("nan"), float("nan")
        reason = str(error)
    else:
        reason = ""

    if not background:
        # rimexam background = no: nothing subtracted, no annulus read.
        buffer = width = 0.0
    reach = max(rplot, radius + buffer + width)
    ny, nx = data.shape
    x1, x2 = max(1, int(x - reach)), min(nx, int(x + reach))
    y1, y2 = max(1, int(y - reach)), min(ny, int(y + reach))
    box = data[y1 - 1:y2, x1 - 1:x2]
    j, i = np.mgrid[y1:y2 + 1, x1:x2 + 1]
    dx, dy = i - x, j - y
    # IRAF's radii are single precision: sqrt(dx^2 + dy^2) of REALs.
    fx, fy = (i - np.float32(x)).astype(np.float32), (j - np.float32(y)).astype(np.float32)
    r = np.sqrt((fx * fx + fy * fy).astype(np.float32)).astype(float)

    # The sky: the median of the annulus -- or nothing subtracted, as in
    # IRAF, when background is off or the annulus holds no pixel.
    ring = (r >= radius + buffer) & (r <= radius + buffer + width)
    sky = noise = 0.0
    if background and width > 0 and ring.any():
        sky_pixels = box[ring]
        sky_pixels = sky_pixels[np.isfinite(sky_pixels)]
        if sky_pixels.size == 0:
            return RadialProfile(x, y, np.array([]), np.array([]), radius=float(radius),
                                 fittype=fittype,
                                 message="no usable sky pixels around here")
        # amedr: the lower middle value, not the mean of the two middle ones.
        sky = (float(sky_pixels[0]) if sky_pixels.size <= 2
               else float(np.sort(sky_pixels)[(sky_pixels.size + 1) // 2 - 1]))
        noise = float(1.4826 * np.median(np.abs(sky_pixels - sky)))
    signal = box - sky

    if not np.isfinite(signal[r <= radius]).all():
        profile = RadialProfile(x, y, np.array([]), np.array([]), radius=float(radius),
                                sky=sky, fittype=fittype,
                                message="blank (NaN) pixels inside the aperture -- "
                                        "fix them first, or measure another star")
        return profile
    inside = r <= radius
    shown = r <= rplot
    order = np.argsort(r[shown])
    profile = RadialProfile(
        x, y, r[shown][order], box[shown][order], radius=float(radius), sky=sky,
        noise=noise, flux=float(signal[inside].sum()), area=int(inside.sum()),
        peak=float(signal[inside].max()) if inside.any() else 0.0,
        direct_fwhm=dfwhm, enclosed_fwhm=enclosed, fittype=fittype)
    _moments(profile, dx[inside], dy[inside], signal[inside])

    if star is None:
        profile.message = reason or "no star here"
        return profile
    # ie_rimexam fits every pixel inside the aperture, whatever rplot is.
    _fit(profile, r[inside], signal[inside], dfwhm, rplot, beta)
    return profile


def _moments(profile: RadialProfile, dx, dy, signal) -> None:
    """Ellipticity and position angle from the second intensity moments."""
    total = signal.sum()
    if total <= 0:
        return
    sxx, syy, sxy = (dx * dx * signal).sum(), (dy * dy * signal).sum(), \
        (dx * dy * signal).sum()
    if sxx + syy <= 0:
        return
    e = float(np.hypot(sxx - syy, 2 * sxy) / (sxx + syy))
    if e < 0.01:
        # IRAF prints e = 0 and no angle: below 1% the angle is noise.
        profile.ellipticity, profile.pa = 0.0, float("nan")
    else:
        profile.ellipticity = e
        profile.pa = float(np.degrees(0.5 * np.arctan2(2 * sxy, sxx - syy)))


def _fit(profile: RadialProfile, radii: np.ndarray, values: np.ndarray, dfwhm: float,
         rplot: float, beta: float | None) -> None:
    """The fit to the individual pixels, weighted as ie_rimexam weights it."""
    if radii.size < 4:
        profile.message = "too few pixels to fit -- try a larger radius"
        return
    amplitude = profile.peak
    if amplitude <= 0:
        profile.message = "no source above the sky here"
        return
    # The brightest pixel of any patch of sky stands a few sigma above the
    # rest, and a profile will happily fit it and report a FWHM. Five sigma
    # is the floor below which that number describes the noise, not a star.
    # IRAF has no such floor; this is harwood refusing to print nonsense.
    if profile.noise > 0 and amplitude < 5 * profile.noise:
        profile.message = (f"nothing stands above the sky noise here (peak "
                           f"{amplitude:.1f}, sky sigma {profile.noise:.1f}) -- "
                           "point closer to a star")
        return

    from ..core import nlfit, starfocus

    f32 = np.float32
    if not np.isfinite(dfwhm) or dfwhm <= 0:
        dfwhm = 2.0
    xs = radii.astype(f32)
    ys = values.astype(f32)
    r3 = np.clip(f32(2) * xs / f32(dfwhm) - f32(1), 0, 5).astype(f32)
    # exp(-r3^2) / max(.1, r^2): double from the exp on, as IRAF's C has it.
    weights = np.array([math.exp(-float(f32(v * v))) / float(max(f32(0.1), f32(r * r)))
                        for v, r in zip(r3, xs, strict=True)], dtype=f32)

    if profile.fittype == "moffat":
        if beta is not None and beta <= 0:
            profile.message = (f"a Moffat needs a positive beta, not {beta:g} -- "
                               ":beta INDEF fits it")
            return
        power = f32(-3.0 if beta is None else -beta)
        alpha0 = f32(dfwhm / 2.0 / math.sqrt(2.0 ** float(f32(f32(-1) / power)) - 1.0))
        fit = nlfit.NLFit(starfocus.moffat, starfocus.dmoffat,
                          [f32(amplitude), alpha0, power],
                          [0, 1, 2] if beta is None else [0, 1], 0.001, 100)
    else:
        fit = nlfit.NLFit(_ie_gauss, _ie_dgauss,
                          [f32(amplitude), f32(dfwhm ** 2 / (8 * math.log(2.0)))],
                          [0, 1], 0.001, 100)
    status = fit.fit(xs, ys, weights)
    if status in (nlfit.SINGULAR, nlfit.NO_DEG_FREEDOM):
        profile.message = (f"the {profile.fittype} fit did not converge -- "
                           "is there a star here?")
        return
    p = fit.param
    if p[1] < 0:
        profile.message = (f"the {profile.fittype} fit gave a negative width -- "
                           "is there a star here?")
        return
    if profile.fittype == "moffat":
        if p[2] >= 0:
            # A rising "Moffat": IRAF prints INDEF for its FWHM.
            profile.message = ("the Moffat fit gave a non-physical beta -- is there a "
                               "star here? :fittype gaussian may fit it")
            return
        profile.beta = float(-p[2])
        profile.fwhm = float(f32(abs(float(p[1])) * 2.0
                                 * math.sqrt(2.0 ** float(f32(f32(-1) / p[2])) - 1.0)))
    else:
        profile.fwhm = float(f32(math.sqrt(8.0 * float(f32(math.log(2.0))) * float(p[1]))))
    profile.peak = float(p[0])
    grid = np.linspace(0, rplot, 51).astype(f32)
    profile.fit_radii = grid.astype(float)
    profile.fit = fit.vector(grid).astype(float) + profile.sky
    profile.converged = True


def _ie_gauss(x, p):
    """ie_gauss: amplitude and sigma squared, of the radius."""
    r2 = f32(f32(x * x) / f32(p[1] * f32(2)))
    return f32(0) if abs(r2) > 20 else f32(float(p[0]) * math.exp(-float(r2)))


def _ie_dgauss(x, p):
    r2 = f32(f32(x * x) / f32(p[1] * f32(2)))
    if abs(r2) > 20:
        return f32(0), (f32(0), f32(0))
    d1 = f32(math.exp(-float(r2)))
    z = f32(p[0] * d1)
    return z, (d1, f32(f32(z * r2) / p[1]))


f32 = np.float32


def quick_phot(
    data: np.ndarray, x: float, y: float,
    aperture: float = 5.0, annulus: float = 10.0, dannulus: float = 5.0,
    epadu: float = 1.0, zmag: float = 25.0,
) -> dict:
    """One star, one aperture, straight away -- in apphot's terms.

    Kept beside the rimexam measurement because a few callers think in
    aperture, annulus and dannulus rather than radius, buffer and width.
    """
    x, y = centroid(data, x, y, radius=aperture)
    row, col = pixels.to_numpy(x, y)
    reach = int(np.ceil(annulus + dannulus)) + 1
    r0, r1 = max(0, int(row) - reach), min(data.shape[0], int(row) + reach + 1)
    c0, c1 = max(0, int(col) - reach), min(data.shape[1], int(col) + reach + 1)
    box = data[r0:r1, c0:c1]
    rows = np.arange(r0, r1)[:, None]
    cols = np.arange(c0, c1)[None, :]
    distance = np.hypot(rows - row, cols - col)

    ring = (distance >= annulus) & (distance <= annulus + dannulus)
    sky = float(np.median(box[ring])) if ring.any() else 0.0
    inside = distance <= aperture
    area = float(inside.sum())
    flux = float(box[inside].sum() - sky * area)

    with np.errstate(divide="ignore", invalid="ignore"):
        mag = zmag - 2.5 * np.log10(flux) if flux > 0 else float("nan")
    return {"x": x, "y": y, "sky": sky, "flux": flux, "area": area, "mag": mag}


def statistics(data: np.ndarray, x: float, y: float, ncols: int = 5,
               nlines: int = 5) -> dict:
    """What the `m` key reports: statistics of a box around the cursor.

    IRAF's ncstat by nlstat, 5 by 5 unless told otherwise.
    """
    row, col = pixels.to_numpy(x, y)
    r, c = int(round(row)), int(round(col))
    r0 = max(0, r - (nlines - 1) // 2)
    c0 = max(0, c - (ncols - 1) // 2)
    box = data[r0:min(data.shape[0], r0 + nlines), c0:min(data.shape[1], c0 + ncols)]
    return {"npix": int(box.size), "mean": float(box.mean()),
            "median": float(np.median(box)), "stddev": float(box.std(ddof=1)),
            "min": float(box.min()), "max": float(box.max())}
