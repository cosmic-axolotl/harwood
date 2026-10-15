"""The measurements behind the imexamine keys -- no Qt in this file.

Section 8.2 keeps the analysis out of the window so that the same numbers come
out whether they were asked for by a keystroke, by a task, or by a test. The
prototype's `r` key is `radial_profile`; `a` is `quick_phot`; `m` is
`statistics`.

Positions in and out are 1-based (x, y), as everywhere a user can see.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import pixels


@dataclass
class RadialProfile:
    """What the `r` key measures at a position."""

    x: float
    y: float
    radii: np.ndarray = field(repr=False)
    values: np.ndarray = field(repr=False)
    sky: float = 0.0
    #: Scatter of the sky annulus, robust (1.4826 x the MAD).
    noise: float = 0.0
    peak: float = 0.0
    fwhm: float = float("nan")
    fit: np.ndarray | None = field(default=None, repr=False)
    fit_radii: np.ndarray | None = field(default=None, repr=False)
    fittype: str = "gaussian"
    converged: bool = False
    message: str = ""

    @property
    def total(self) -> float:
        """Flux above sky inside the profile radius, summed over the pixels."""
        return float(np.sum(self.values - self.sky))


def centroid(data: np.ndarray, x: float, y: float, radius: float = 5.0) -> tuple[float, float]:
    """Marginal centroid in a box around a 1-based position."""
    row, col = pixels.to_numpy(x, y)
    half = int(np.ceil(radius))
    r0, r1 = max(0, int(row) - half), min(data.shape[0], int(row) + half + 1)
    c0, c1 = max(0, int(col) - half), min(data.shape[1], int(col) + half + 1)
    box = data[r0:r1, c0:c1]
    if box.size == 0:
        return x, y

    floor = np.median(box)
    weights = np.clip(box - floor, 0, None)
    if weights.sum() <= 0:
        return x, y
    rows = np.arange(r0, r1)[:, None]
    cols = np.arange(c0, c1)[None, :]
    cy = float((weights * rows).sum() / weights.sum())
    cx = float((weights * cols).sum() / weights.sum())
    return pixels.from_numpy(cy, cx)


def radial_profile(
    data: np.ndarray,
    x: float,
    y: float,
    radius: float = 12.0,
    buffer: float = 5.0,
    width: float = 5.0,
    fittype: str = "gaussian",
    recentre: bool = True,
) -> RadialProfile:
    """Pixel values against distance from a star, with a fitted profile.

    `radius` is how far out the profile goes; the sky is measured in the
    annulus from `radius + buffer` to `radius + buffer + width`, which is why
    those two parameters exist separately -- the buffer keeps the star's own
    wings out of its sky.
    """
    if recentre:
        x, y = centroid(data, x, y, radius=max(3.0, radius / 3))

    row, col = pixels.to_numpy(x, y)
    reach = int(np.ceil(radius + buffer + width)) + 1
    r0, r1 = max(0, int(row) - reach), min(data.shape[0], int(row) + reach + 1)
    c0, c1 = max(0, int(col) - reach), min(data.shape[1], int(col) + reach + 1)
    box = data[r0:r1, c0:c1]
    if box.size == 0:
        return RadialProfile(x, y, np.array([]), np.array([]),
                             message="that position is outside the image")

    rows = np.arange(r0, r1)[:, None]
    cols = np.arange(c0, c1)[None, :]
    distance = np.hypot(rows - row, cols - col)

    ring = (distance >= radius + buffer) & (distance <= radius + buffer + width)
    sky_pixels = box[ring] if ring.any() else box.ravel()
    sky = float(np.median(sky_pixels))
    noise = float(1.4826 * np.median(np.abs(sky_pixels - sky)))

    inside = distance <= radius
    radii = distance[inside].ravel()
    values = box[inside].ravel()
    order = np.argsort(radii)
    radii, values = radii[order], values[order]

    profile = RadialProfile(x, y, radii, values, sky=sky, noise=noise,
                            peak=float(values.max() - sky) if values.size else 0.0,
                            fittype=fittype)
    _fit(profile)
    return profile


def _fit(profile: RadialProfile) -> None:
    """Fit the requested analytic profile; say so plainly when it fails."""
    if profile.radii.size < 6:
        profile.message = "too few pixels to fit"
        return

    amplitude = profile.peak
    if amplitude <= 0:
        profile.message = "no source above the sky here"
        return
    # The brightest pixel of any patch of sky stands a few sigma above the
    # rest, and a gaussian will happily fit it and report a FWHM. Five sigma
    # is the floor below which that number describes the noise, not a star.
    if profile.noise > 0 and amplitude < 5 * profile.noise:
        profile.message = (f"nothing stands above the sky noise here (peak "
                           f"{amplitude:.1f}, sky sigma {profile.noise:.1f}) -- "
                           "point closer to a star")
        return

    y = profile.values - profile.sky
    grid = np.linspace(0, profile.radii.max(), 120)

    try:
        from scipy.optimize import curve_fit
    except ImportError:
        profile.message = "scipy is not installed; showing the profile without a fit"
        return

    if profile.fittype == "moffat":
        def model(r, a, alpha, beta):
            return a * (1 + (r / alpha) ** 2) ** (-beta)

        guess = (amplitude, 2.0, 2.5)
    else:
        def model(r, a, sigma):
            return a * np.exp(-(r ** 2) / (2 * sigma ** 2))

        guess = (amplitude, 2.0)

    try:
        popt, _ = curve_fit(model, profile.radii, y, p0=guess, maxfev=4000)
    except Exception:
        profile.message = "the fit did not converge -- is there a star here?"
        return

    if profile.fittype == "moffat":
        _, alpha, beta = popt
        profile.fwhm = float(2 * abs(alpha) * np.sqrt(2 ** (1 / beta) - 1))
    else:
        profile.fwhm = float(2.3548 * abs(popt[1]))
    profile.fit_radii = grid
    profile.fit = model(grid, *popt) + profile.sky
    profile.converged = True


def quick_phot(
    data: np.ndarray, x: float, y: float,
    aperture: float = 5.0, annulus: float = 10.0, dannulus: float = 5.0,
    epadu: float = 1.0, zmag: float = 25.0,
) -> dict:
    """What the `a` key reports: one star, one aperture, straight away."""
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


def statistics(data: np.ndarray, x: float, y: float, size: int = 11) -> dict:
    """What the `m` key reports: statistics of a box around the cursor."""
    row, col = pixels.to_numpy(x, y)
    half = size // 2
    r0, r1 = max(0, int(row) - half), min(data.shape[0], int(row) + half + 1)
    c0, c1 = max(0, int(col) - half), min(data.shape[1], int(col) + half + 1)
    box = data[r0:r1, c0:c1]
    return {"npix": int(box.size), "mean": float(box.mean()),
            "median": float(np.median(box)), "stddev": float(box.std(ddof=1)),
            "min": float(box.min()), "max": float(box.max())}
