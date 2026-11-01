"""PSF photometry with an effective PSF: Anderson & King (2000, PASP 112, 1360), the method of
HST's and of most of what has come since. IRAF's daophot fits an analytic function and a table
of residuals; an effective PSF is the image of a star as the pixels themselves saw it, built
from the stars in the field with no function assumed, and so it needs no correction for the
pixel it is sampled on.

Each star gives a sample of the PSF at pixel offsets, and with stars at different phases those
samples fill a grid finer than the pixels. It starts from nothing and goes round: what each
sample has left over against the PSF so far, clipped median at each grid point, smoothed by the
quartic kernel of the paper and added on, and each star's centre measured anew. Its flux only
in the first passes, and a star does not go far: otherwise the two lock onto the noise, passes
later, and the PSF collapses into a spike. Finer grids want more stars: with a few dozen, 2 to
a pixel is what got nearest a PSF known beforehand, and 4 was three times worse.

A star then fits three numbers, its flux and its position, on the pixels near it, its
neighbours taken out with the fits so far and all of them gone round again, as daophot's
NSTAR does.

Positions in and out are 1-based (x, y), as everywhere a user can see.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import advice

#: Anderson & King's smoothing: a quartic fit over five grid points
KERNEL = np.array([[+0.041632, -0.080816, -0.078368, -0.080816, +0.041632],
                   [-0.080816, +0.078368, +0.200816, +0.078368, -0.080816],
                   [-0.078368, +0.200816, +0.441632, +0.200816, -0.078368],
                   [-0.080816, +0.078368, +0.200816, +0.078368, -0.080816],
                   [+0.041632, -0.080816, -0.078368, -0.080816, +0.041632]])


@dataclass
class EPSF:
    grid: np.ndarray                 #: its value at each grid point, flux per pixel
    oversampling: int
    size: int                        #: in image pixels
    nstars: int = 0
    used: list = field(default_factory=list)

    def at(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """The fraction of a star's flux a pixel gets, at offsets from its centre: bicubic
        through the grid, zero outside it."""
        from scipy.ndimage import map_coordinates

        centre = (self.grid.shape[0] - 1) / 2.0
        coordinates = np.array([np.asarray(dy) * self.oversampling + centre,
                                np.asarray(dx) * self.oversampling + centre])
        return map_coordinates(self.grid, coordinates, order=3, mode="constant", cval=0.0)

    def sampled(self, shape: tuple[int, int] | None = None) -> np.ndarray:
        """On the image's pixels, centred, summing to one: for what wants an array."""
        half = self.size // 2
        y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
        image = self.at(x, y)
        image = image / image.sum()
        if shape is None:
            return image
        out = np.zeros(shape)
        cy, cx = shape[0] // 2, shape[1] // 2
        out[cy - half:cy + half + 1, cx - half:cx + half + 1] = image
        return out


def smooth(image: np.ndarray) -> np.ndarray:
    """An elliptical Moffat fitted to a PSF on the image's pixels, tapered to zero at the edge
    of its box. An effective PSF has the noise of its stars in it, and a sharp end; what
    divides by a PSF in Fourier space, as ZOGY does, turns both into rings around bright
    stars."""
    from scipy.optimize import least_squares

    half = image.shape[0] // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)

    def model(p):
        amplitude, alpha, beta, e1, e2 = p
        q = (x * x * (1 + e1) + y * y * (1 - e1) + 2 * e2 * x * y) / (alpha * alpha)
        return amplitude * (1 + q) ** -beta

    found = least_squares(lambda p: (model(p) - image).ravel(),
                          [float(image.max()), 3.0, 2.5, 0.0, 0.0],
                          bounds=([0.0, 0.3, 1.01, -0.6, -0.6],
                                  [np.inf, 50.0, 30.0, 0.6, 0.6]))
    radius = np.hypot(x, y) / half
    taper = np.where(radius < 0.8, 1.0, np.where(
        radius >= 1.0, 0.0, 0.5 * (1 + np.cos(np.pi * (radius - 0.8) / 0.2))))
    out = model(found.x) * taper
    return out / out.sum()


def _cutout(data, x, y, half):
    """1-based position; the integer box around it, or None off the image."""
    ny, nx = data.shape
    ix, iy = int(round(x)) - 1, int(round(y)) - 1
    if ix - half < 0 or iy - half < 0 or ix + half >= nx or iy + half >= ny:
        return None
    box = data[iy - half:iy + half + 1, ix - half:ix + half + 1]
    jj, ii = np.mgrid[iy - half:iy + half + 1, ix - half:ix + half + 1]
    return box, ii + 1.0, jj + 1.0


def _centroid(box, xs, ys, x0, y0, reach=3.0):
    """Near where it was said to be, as a neighbour in the box is not the star."""
    near = (xs - x0) ** 2 + (ys - y0) ** 2 <= reach * reach
    above = np.clip(box - np.median(box), 0, None) * near
    total = above.sum()
    if total <= 0:
        return None
    return float((above * xs).sum() / total), float((above * ys).sum() / total)


def build(data, stars, sky=None, size: int = 21, oversampling: int = 2, iterations: int = 6,
          radius: float | None = None, clip: float = 3.0) -> EPSF:
    """The effective PSF of an image, from `stars`, their (x, y). `sky` is a number or an
    image, and with none each star's is the median at the edge of its box."""
    data = np.asarray(data, dtype=np.float64)
    if size % 2 == 0:
        size += 1
    half = size // 2
    radius = radius if radius is not None else half * 0.6
    fit_radius = max(3.0, half * 0.4)
    cuts = []
    for x, y in stars:
        cut = _cutout(data, x, y, half)
        if cut is None or not np.isfinite(cut[0]).all():
            continue
        box, xs, ys = cut
        if sky is None:
            edge = np.concatenate([box[0], box[-1], box[:, 0], box[:, -1]])
            local = float(np.median(edge))
        elif np.ndim(sky) == 0:
            local = float(sky)
        else:
            local = sky[ys.astype(int) - 1, xs.astype(int) - 1]
        box = box - local
        found = _centroid(box, xs, ys, x, y)
        if found is None:
            continue
        cuts.append([box, xs, ys, found[0], found[1], None, found[0], found[1]])
    if len(cuts) < 3:
        advice.fail(f"only {len(cuts)} usable stars for the PSF",
                    "it needs stars well away from the edges: give more, or a smaller --size")
    n = size * oversampling + 1
    if oversampling % 2:
        n += 1
    psf = EPSF(np.zeros((n, n)), oversampling, size, len(cuts))
    centre = (n - 1) / 2.0
    for star in cuts:
        box, xs, ys, x0, y0, _, _, _ = star
        r2 = (xs - x0) ** 2 + (ys - y0) ** 2
        star[5] = float(box[r2 <= radius * radius].sum())
    yg, xg = np.mgrid[0:n, 0:n]
    window = ((xg - centre) ** 2 + (yg - centre) ** 2
              <= ((half - 0.5) * oversampling) ** 2)
    for iteration in range(iterations):
        offsets = [[] for _ in range(n * n)]
        for box, xs, ys, x0, y0, flux, _, _ in cuts:
            if flux <= 0:
                continue
            dx, dy = xs - x0, ys - y0
            value = box / flux - psf.at(dx, dy)
            ig = np.rint(dx * oversampling + centre).astype(int)
            jg = np.rint(dy * oversampling + centre).astype(int)
            inside = (ig >= 0) & (ig < n) & (jg >= 0) & (jg < n)
            for k, v in zip((jg[inside] * n + ig[inside]).ravel(), value[inside].ravel(),
                            strict=True):
                offsets[k].append(v)
        change = np.zeros((n, n))
        for k, samples in enumerate(offsets):
            if not samples:
                continue
            values = np.array(samples)
            if values.size >= 3:
                for _ in range(3):
                    middle, spread = np.median(values), np.std(values)
                    if spread == 0:
                        break
                    kept = values[np.abs(values - middle) <= clip * spread]
                    if kept.size == values.size or kept.size == 0:
                        break
                    values = kept
            change[k // n, k % n] = np.median(values)
        from scipy.ndimage import convolve

        grid = psf.grid + convolve(change, KERNEL, mode="constant", cval=0.0)
        grid = np.where(window, grid, 0.0)
        # Its centre is where the fits will put a star. Only when it has wandered: each
        # shift through the interpolant blurs it a little, and that adds up.
        core = (xg - centre) ** 2 + (yg - centre) ** 2 <= (2.0 * oversampling) ** 2
        positive = np.clip(grid, 0, None) * core
        total = positive.sum()
        if total > 0:
            cx = (positive * xg).sum() / total - centre
            cy = (positive * yg).sum() / total - centre
            if max(abs(cx), abs(cy)) > 0.25:
                from scipy.ndimage import shift

                grid = shift(grid, (-cy, -cx), order=3, mode="constant", cval=0.0)
        grid = grid * oversampling ** 2 / grid[window].sum()
        moved = float(np.abs(grid - psf.grid).max())
        psf.grid = grid
        if iteration >= 2 and moved < 0.002 * float(grid.max()):
            break
        for star in cuts:
            box, xs, ys, x0, y0, flux, _, _ = star
            fitted = _fit_one(psf, box, xs, ys, np.ones_like(box), x0, y0, flux, 0.0,
                              fit_radius)
            # A star that has wandered stays where it was, and past the first passes its flux
            # stays too: left free to the end, the two lock onto the noise, and the PSF
            # collapses into a spike.
            near = (fitted is not None
                    and np.hypot(fitted[0] - star[6], fitted[1] - star[7]) < 1)
            if near:
                star[3], star[4] = fitted[:2]
                if iteration < 3 and fitted[2] > 0:
                    star[5] = fitted[2]
    psf.used = [(star[3], star[4]) for star in cuts]
    return psf


def _fit_one(psf, box, xs, ys, weight, x0, y0, flux, sky, radius, fit_sky=False,
             iterations=25):
    """Gauss-Newton on x, y, flux, and if asked the sky, with the steps held to a pixel."""
    r2 = (xs - x0) ** 2 + (ys - y0) ** 2
    inside = (r2 <= radius * radius) & (weight > 0)
    if inside.sum() < (4 if fit_sky else 3):
        return None
    X, Y, D, W = xs[inside], ys[inside], box[inside], weight[inside]
    h = 0.05
    step_limit = 1.0
    for _ in range(iterations):
        model = psf.at(X - x0, Y - y0)
        ddx = (psf.at(X - x0 - h, Y - y0) - psf.at(X - x0 + h, Y - y0)) / (2 * h)
        ddy = (psf.at(X - x0, Y - y0 - h) - psf.at(X - x0, Y - y0 + h)) / (2 * h)
        columns = [model, flux * ddx, flux * ddy]
        if fit_sky:
            columns.append(np.ones_like(model))
        A = np.array(columns).T
        resid = D - (flux * model + sky)
        AtA = (A * W[:, None]).T @ A
        try:
            delta = np.linalg.solve(AtA, (A * W[:, None]).T @ resid)
        except np.linalg.LinAlgError:
            return None
        flux = flux + delta[0]
        x0 = x0 + float(np.clip(delta[1], -step_limit, step_limit))
        y0 = y0 + float(np.clip(delta[2], -step_limit, step_limit))
        if fit_sky:
            sky = sky + delta[3]
        if abs(delta[1]) < 1e-3 and abs(delta[2]) < 1e-3 and abs(delta[0]) < 1e-4 * abs(flux):
            break
    model = psf.at(X - x0, Y - y0)
    resid = D - (flux * model + sky)
    dof = max(1, resid.size - len(columns))
    chi = float(np.sqrt((W * resid * resid).sum() / dof))
    try:
        cov = np.linalg.inv(AtA)
        ferr = float(np.sqrt(max(cov[0, 0], 0.0)))
        xerr, yerr = float(np.sqrt(max(cov[1, 1], 0.0))), float(np.sqrt(max(cov[2, 2], 0.0)))
    except np.linalg.LinAlgError:
        ferr = xerr = yerr = float("nan")
    return x0, y0, flux, sky, ferr, xerr, yerr, chi


@dataclass
class Fit:
    x: float
    y: float
    flux: float
    sky: float
    ferr: float
    xerr: float
    yerr: float
    chi: float
    ier: int = 0


def photometry(data, psf: EPSF, stars, sky, variance=None, radius: float = 4.0,
               passes: int = 3, fit_sky: bool = False) -> tuple[list[Fit], np.ndarray]:
    """The stars of `stars`, (x, y, first guess of the flux or None), fitted together: the
    brightest first, each with the others as they stand taken out of the image, three times
    round. `sky` is a number or an image; `variance` an image, or else each pixel counts the
    same and the errors are scaled by what is left over. With the image less all of them."""
    data = np.asarray(data, dtype=np.float64)
    sky_image = (np.full(data.shape, float(sky)) if np.ndim(sky) == 0
                 else np.asarray(sky, float))
    residual = data - sky_image
    half = psf.size // 2
    fits = []
    for x, y, guess in stars:
        flux = float(guess) if guess is not None else 0.0
        cut = _cutout(residual, x, y, int(radius) + 1)
        if guess is None and cut is not None:
            box, xs, ys = cut
            flux = float(np.clip(box[(xs - x) ** 2 + (ys - y) ** 2 <= radius * radius], 0,
                                 None).sum())
        fits.append(Fit(float(x), float(y), max(flux, 1.0), 0.0, float("nan"), float("nan"),
                        float("nan"), float("nan")))
    order = sorted(range(len(fits)), key=lambda k: -fits[k].flux)
    ny, nx = data.shape

    def add(fit: Fit, sign: float) -> None:
        ix, iy = int(round(fit.x)) - 1, int(round(fit.y)) - 1
        x1, x2, y1, y2 = max(0, ix - half), min(nx, ix + half + 1), max(0, iy - half), min(
            ny, iy + half + 1)
        if x2 <= x1 or y2 <= y1:
            return
        jj, ii = np.mgrid[y1:y2, x1:x2]
        residual[y1:y2, x1:x2] += sign * fit.flux * psf.at(ii + 1.0 - fit.x, jj + 1.0 - fit.y)

    # Everyone out to start with, from the first guess.
    for k in order:
        add(fits[k], -1.0)
    for _ in range(passes):
        for k in order:
            fit = fits[k]
            add(fit, +1.0)
            cut = _cutout(residual, fit.x, fit.y, int(radius) + 1)
            if cut is None:
                fit.ier = 1
                continue
            box, xs, ys = cut
            if variance is None:
                weight = np.ones_like(box)
            else:
                iy, ix = ys.astype(int) - 1, xs.astype(int) - 1
                var = variance[iy, ix]
                weight = np.where(var > 0, 1.0 / np.where(var > 0, var, 1.0), 0.0)
            good = np.isfinite(box)
            weight = np.where(good, weight, 0.0)
            box = np.where(good, box, 0.0)
            done = _fit_one(psf, box, xs, ys, weight, fit.x, fit.y, fit.flux, 0.0, radius,
                            fit_sky)
            if done is None:
                fit.ier = 2
            else:
                fit.x, fit.y, fit.flux, fit.sky, fit.ferr, fit.xerr, fit.yerr, fit.chi = done
                fit.ier = 0
                if variance is None:
                    fit.ferr *= fit.chi
                    fit.xerr *= fit.chi
                    fit.yerr *= fit.chi
            add(fit, -1.0)
    return fits, residual
