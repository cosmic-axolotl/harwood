"""IRAF's starfocus profile code, which imexamine's r and a keys stand on.

`images.tv.imexamine` does not measure the direct FWHM itself: it calls
`stf_measure`, a cut-down psfmeasure, and prints what that returns. This is a
line-for-line port of that path, from iraf-community 2.18
(`pkg/images/tv/imexamine/stfmeasure.x` and `stfprofile.x`), because the
direct FWHM also *decides the aperture* -- the a key resets the radius to
three times it -- so an approximation shifts the magnitude too. A first
version that averaged an interpolated image came out 5% wider than IRAF on
the 1987 M92 frames; this one is checked against IRAF's own output on them.

What it does, in IRAF's order:

``find``      recentre on the marginal distributions above their mean
``bkgd``      the sky: midpoint of the narrowest interval holding half the
              annulus pixels (a mode, robust to stars in the annulus)
``profile``   the enclosed flux, summed over subpixels of a bicubic spline
              corrected so each pixel keeps its own total. A star narrower
              than five pixels has a gaussian taken out first, integrated over
              the pixel, and only the residuals are interpolated -- which is
              what stops the pixel itself from widening the profile
``dfwhm``     differentiate the enclosed flux into a radial profile; the
              direct FWHM is twice the radius where it falls to half its peak
``fit``       gaussian and Moffat fits to the enclosed flux: the ENCLOSED
              column of imexamine's output

Two things are not carried over: the dither IRAF adds to integer pixel types
(a pseudo-random sequence from its own urand), and the saturation warning.

Positions are 1-based (x, y) image coordinates, as in IRAF.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The surface IRAF uses to correct the width of a star narrower than two
#: pixels for where its centre falls in the pixel (gsrestore format: type 1 =
#: chebyshev, orders 4 x 4, full cross terms, x = centre offset 0..0.67,
#: y = true FWHM 1..2, then the coefficients with x varying fastest).
_GSDATA = (1., 4., 4., 1., 0., 0.6726812, 1., 2., 1.630641, 0.088787,
           0.00389378, -0.001457133, 0.3932125, -0.1267456, -0.004864541,
           0.00249941, 0.03078612, 0.02731274, -4.875850E-4, 2.307464E-4,
           -0.002134843, 0.007603908, -0.002552385, -8.010564E-4)


@dataclass
class Measurement:
    """What stf_measure hands back to imexamine."""

    x: float
    y: float
    radius: float           #: the radius of the last pass
    background: float
    dfwhm: float            #: direct FWHM
    gfwhm: float            #: gaussian fit to the enclosed flux
    mfwhm: float            #: Moffat fit to the enclosed flux
    beta: float             #: of that Moffat fit
    renclosed: float        #: radius enclosing half the flux


# ------------------------------------------------------------ sample grid


def i2r(i):
    """stf_i2r: sample index -> radius. Fine near the centre, coarse outside."""
    i = np.asarray(i, dtype=float)
    return np.select([i < 20, i < 30, i < 40, i < 50],
                     [0.05 * i, 0.1 * i - 1, 0.2 * i - 4, 0.5 * i - 16], i - 41)


def r2i(r):
    """stf_r2i: radius -> sample index."""
    r = np.asarray(r, dtype=float)
    return np.select([r < 1, r < 2, r < 4, r < 9],
                     [20 * r, 10 * (r + 1), 5 * (r + 4), 2 * (r + 16)], r + 41)


def r2n(r):
    """stf_r2n: subpixels per side for a pixel at this radius."""
    r = np.asarray(r, dtype=float)
    return np.select([r < 1, r < 2, r < 4, r < 9], [20, 10, 5, 2], 1).astype(int)


def _nint(value: float) -> int:
    return int(np.floor(value + 0.5))


# ------------------------------------------------------------------ steps


def measure(data: np.ndarray, x: float, y: float, radius: float = 5.0,
            buffer: float = 5.0, width: float = 5.0, iterations: int = 3,
            beta: float | None = None, level: float = 0.5) -> Measurement:
    """stf_measure: find, sky, profile and widths, `iterations` times.

    After the first pass the radius becomes three times the direct FWHM of
    the pass before, which is how imexamine's a key sizes its aperture.
    """
    data = np.asarray(data, dtype=float)
    current = radius
    result = None
    for number in range(iterations):
        if number > 0:
            current = 3.0 * result.dfwhm
        npmax = int(r2i(current)) + 1
        raster, x1, y1, xc, yc = _find(data, x, y, current, buffer, width)
        background, miso = _background(raster, x1, y1, xc, yc, current, buffer, width)
        enclosed = _profile(raster, x1, y1, xc, yc, background, miso, current, npmax)
        derivative = _derivative(enclosed, npmax)
        dfwhm = _dfwhm(derivative, npmax)
        gfwhm, mfwhm, fitted_beta = _fit(enclosed, npmax, beta)
        result = Measurement(xc, yc, current, background, dfwhm, gfwhm, mfwhm,
                             fitted_beta, _radius(enclosed, npmax, level))
    return result


def _find(data, x, y, radius, buffer, width):
    """stf_find: centre on the marginals, then cut the raster for the rest."""
    radius = max(3.0, radius)
    reach = radius + buffer + width
    ny, nx = data.shape
    xc, yc = x, y
    for _ in range(3):
        xlast, ylast = xc, yc
        x1, x2 = _nint(xc - radius), _nint(xc + radius)
        y1, y2 = _nint(yc - radius), _nint(yc + radius)
        box = _cut(data, x1, x2, y1, y2)
        total = box.sum()
        columns = box.sum(axis=0) - total / box.shape[1]
        rows = box.sum(axis=1) - total / box.shape[0]
        if not (columns > 0).any() or not (rows > 0).any():
            raise ValueError("centering failed to converge -- is there a star here?")
        w = np.clip(columns, 0, None)
        xc = float((w * np.arange(x1, x2 + 1)).sum() / w.sum())
        w = np.clip(rows, 0, None)
        yc = float((w * np.arange(y1, y2 + 1)).sum() / w.sum())
        xc = float(np.clip(xc, xlast - 0.2 * box.shape[1], xlast + 0.2 * box.shape[1]))
        yc = float(np.clip(yc, ylast - 0.2 * box.shape[0], ylast + 0.2 * box.shape[0]))
        if _nint(xc) == _nint(xlast) and _nint(yc) == _nint(ylast):
            break
    del nx, ny
    x1, x2 = _nint(xc - reach), _nint(xc + reach)
    y1, y2 = _nint(yc - reach), _nint(yc + reach)
    return _cut(data, x1, x2, y1, y2), x1, y1, xc, yc


def _cut(data, x1, x2, y1, y2):
    """A 1-based inclusive raster, extended past the edge with the nearest
    pixel -- IRAF's default boundary, so a star near the edge still works."""
    rows = np.clip(np.arange(y1, y2 + 1) - 1, 0, data.shape[0] - 1)
    cols = np.clip(np.arange(x1, x2 + 1) - 1, 0, data.shape[1] - 1)
    return data[np.ix_(rows, cols)].astype(float)


def _background(raster, x1, y1, xc, yc, radius, buffer, width):
    """stf_bkgd: the narrowest half of the sorted annulus, and its midpoint."""
    j, i = np.mgrid[y1:y1 + raster.shape[0], x1:x1 + raster.shape[1]]
    r = (xc - i) ** 2 + (yc - j) ** 2
    ring = np.sort(raster[(r >= (radius + buffer) ** 2)
                          & (r <= (radius + buffer + width) ** 2)])
    ns = ring.size
    if ns <= 9:
        return 0.0, 0.0
    span = ring[-1] - ring[0]
    background, miso = ring[0] + span / 2, span / 2
    half = int(1 + 0.5 * ns)
    widths = ring[half - 1:] - ring[:ns - half + 1]
    best = int(np.argmin(widths))
    if widths[best] < span:
        background, miso = ring[best] + widths[best] / 2, widths[best] / 2
    return float(background), float(miso)


def _subpixels(dx: np.ndarray, dy: np.ndarray):
    """Every subpixel centre of every pixel, with the pixel it belongs to."""
    n = r2n(np.hypot(dx, dy))
    owner, sx, sy, area = [], [], [], []
    for index, (px, py, k) in enumerate(zip(dx, dy, n, strict=True)):
        ds = 1.0 / k
        offsets = -0.5 - 0.5 * ds + ds * np.arange(1, k + 1)
        gx, gy = np.meshgrid(px + offsets, py + offsets)
        owner.append(np.full(k * k, index))
        sx.append(gx.ravel())
        sy.append(gy.ravel())
        area.append(np.full(k * k, ds * ds))
    return (np.concatenate(owner), np.concatenate(sx), np.concatenate(sy),
            np.concatenate(area))


def _profile(raster, x1, y1, xc, yc, background, miso, radius, npmax):
    """stf_profile: the enclosed flux at each sample radius, normalised."""
    from scipy.interpolate import RectBivariateSpline

    ny, nx = raster.shape
    xs = np.arange(x1, x1 + nx, dtype=float)
    ys = np.arange(y1, y1 + ny, dtype=float)
    # The edge pixels only anchor the interpolator.
    j, i = np.mgrid[y1 + 1:y1 + ny - 1, x1 + 1:x1 + nx - 1]
    pixel = raster[1:-1, 1:-1].ravel()
    dx, dy = (i - xc).ravel().astype(float), (j - yc).ravel().astype(float)
    owner, sx, sy, area = _subpixels(dx, dy)

    def enclosed(image, analytic=None):
        spline = RectBivariateSpline(ys, xs, image, kx=3, ky=3, s=0)
        values = spline.ev(sy + yc, sx + xc)
        # Each pixel keeps its own total: the offset between the pixel and
        # the mean of its subpixels is spread evenly over them.
        mean = np.bincount(owner, values * area, minlength=pixel.size)
        own = image[1:-1, 1:-1].ravel() - background - mean
        flux = area * (values + own[owner] + (analytic if analytic is not None else 0))
        r = np.maximum(0.0, np.hypot(sx, sy) - np.sqrt(area) / 2)
        keep = r < radius
        samples = i2r(np.arange(1, npmax + 1))
        ds = np.sqrt(area[keep])
        weight = np.clip((samples[:, None] - r[keep][None, :]) / ds[None, :], 0.0, 1.0)
        profile = weight @ flux[keep]
        top = profile.max()
        if top <= 0:
            raise ValueError("invalid flux profile -- is there a star here?")
        return profile / top

    profile = enclosed(raster)

    # A narrow star: take a gaussian out before interpolating.
    fwhm = 2 * _radius(profile, npmax, 0.8) * np.sqrt(np.log(2) / np.log(1 / (1 - 0.8)))
    if fwhm < 5.0:
        if fwhm <= 2.0:
            fwhm = _narrow_correction(fwhm, xc, yc)
        sigma2 = (fwhm / (2 * np.sqrt(np.log(2)))) ** 2

        def integrated(px, py):
            o, gx, gy, a = _subpixels(np.atleast_1d(px), np.atleast_1d(py))
            e = (gx ** 2 + gy ** 2) / sigma2
            return np.bincount(o, np.where(e < 25, np.exp(-np.minimum(e, 25)), 0) * a,
                               minlength=np.size(px))

        ci, cj = _nint(xc), _nint(yc)
        central = raster[cj - y1, ci - x1] - background
        unit = integrated(ci - xc, cj - yc)[0]
        peak = central / unit
        model = np.zeros_like(raster)
        model[1:-1, 1:-1] = (peak * integrated(dx, dy)).reshape(ny - 2, nx - 2)
        e = (sx ** 2 + sy ** 2) / sigma2
        analytic = np.where(e < 25, peak * np.exp(-np.minimum(e, 25)), 0.0)
        profile = enclosed(raster - model, analytic)
    return profile


def _narrow_correction(measured: float, xc: float, yc: float) -> float:
    """The true FWHM, 1 to 2 pixels, whose pixelated width is `measured`."""
    from numpy.polynomial import chebyshev

    g = _GSDATA
    coeff = np.array(g[8:]).reshape(int(g[2]), int(g[1]))       # [y][x]
    offset = np.hypot(xc - _nint(xc), yc - _nint(yc))
    xn = (offset - (g[4] + g[5]) / 2) * 2 / (g[5] - g[4])
    candidates = np.round(np.arange(1.0, 2.0 + 1e-9, 0.01), 2)
    yn = (candidates - (g[6] + g[7]) / 2) * 2 / (g[7] - g[6])
    predicted = chebyshev.chebval2d(np.full_like(yn, xn), yn, coeff.T)
    return float(candidates[np.argmin(np.abs(measured - predicted))])


def _spline(values: np.ndarray):
    """IRAF's asi spline3 over the sample index, which starts at 1."""
    from scipy.interpolate import CubicSpline

    return CubicSpline(np.arange(1, values.size + 1), values, bc_type="natural")


def _radius(profile, npmax, level) -> float:
    """stf_radius: the radius where the enclosed flux first reaches `level`."""
    spline = _spline(profile)
    grid = np.arange(1.0, npmax + 1e-9, 0.01)
    values = spline(grid)
    reached = np.flatnonzero(values >= level)
    index = grid[reached[0]] if reached.size else grid[int(np.argmax(values))]
    return float(i2r(index))


def _derivative(profile, npmax):
    """The radial profile, as the derivative of the enclosed flux."""
    spline = _spline(profile)
    out = np.zeros(npmax)
    dx, count, k, j = 0.25, 0, npmax + 1, 1
    for i in range(1, npmax + 1):
        r = float(i2r(i))
        upper = float(r2i(r + dx))
        if upper > npmax:
            k = i
            break
        lower = float(r2i(r - dx))
        if lower < 1:
            if i > 1:
                value = float(spline(i)) / r ** 2
                out[0] = (count * out[0] + value) / (count + 1)
                count += 1
            j = i
        else:
            out[i - 1] = (float(spline(upper)) - float(spline(lower))) / (4 * r * dx)
    for i in range(2, j + 1):
        out[i - 1] = (out[j] - out[0]) / j * (i - 1) + out[0]
    for i in range(k, npmax + 1):
        out[i - 1] = out[k - 2]
    return out


def _dfwhm(derivative, npmax) -> float:
    """stf_dfwhm: twice the radius at half the peak of the radial profile."""
    spline = _spline(derivative)
    grid = np.arange(1.0, npmax + 1e-9, 0.01)
    values = spline(grid)
    start = int(np.argmax(values))
    half = values[start] / 2
    below = np.flatnonzero(values[start:] <= half)
    at = grid[start + below[0]] if below.size else grid[-1] + 0.01
    return float(2 * i2r(at))


def _fit(profile, npmax, beta):
    """stf_fit: gaussian and Moffat fits to one minus the enclosed flux."""
    from scipy.optimize import curve_fit

    spline = _spline(profile)
    x, y = [], []
    j = 0
    for i in range(1, npmax + 1):
        z = 1.0 - max(0.0, float(spline(i)))
        n = len(x)
        if n > npmax // 3 and z < 0.5:
            break
        if (n < npmax // 3 and z > 0.01) or z > 0.5:
            j = n
        x.append(float(i2r(i)))
        y.append(z)
    x, y = np.array(x), np.array(y)
    yj = min(0.99, y[j])

    def gauss(r, s):
        e = r ** 2 / (2 * s ** 2)
        return np.where(np.abs(e) > 20, 0.0, np.exp(-np.minimum(np.abs(e), 20)))

    sigma = x[j] / np.sqrt(2 * np.log(1 / yj))
    try:
        (fitted,), _ = curve_fit(gauss, x, y, p0=(sigma,), maxfev=200)
        if fitted > 0:
            sigma = fitted
    except Exception:
        pass
    gfwhm = float(2 * sigma * np.sqrt(2 * np.log(2)))

    b = 2.5 if beta is None else beta
    p3 = 1 - b
    alpha = x[j] / np.sqrt(yj ** (1 / p3) - 1)

    def moffat(r, a, power):
        t = 1 + (r / a) ** 2
        return np.where(np.abs(t) > 20, 0.0, np.minimum(t, 20) ** power)

    try:
        if beta is None:
            (a, power), _ = curve_fit(moffat, x, y, p0=(alpha, p3), maxfev=400)
        else:
            (a,), _ = curve_fit(lambda r, a: moffat(r, a, p3), x, y, p0=(alpha,),
                                maxfev=400)
            power = p3
        if a > 0:
            alpha, p3 = a, power
    except Exception:
        pass
    fitted_beta = 1 - p3
    mfwhm = float(2 * alpha * np.sqrt(2 ** (1 / fitted_beta) - 1))
    return gfwhm, mfwhm, float(fitted_beta)
