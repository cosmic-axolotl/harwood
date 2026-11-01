"""The sky of a long-slit frame as one spectrum: Kelson (2003, PASP 115, 688). IRAF's
background, and what apall does, fit each line across the slit, so that a sky line tilted on
the detector is cut differently by each line, and what is left of it is the sharp residual
everyone knows.
Here a tilted line is what helps. Each pixel of the slit samples the sky's spectrum at its own
place, a fraction of a pixel from its neighbours', and a spline in that one coordinate sees it
far better than any line does. It is evaluated back at each pixel, with nothing resampled, so
the noise of what remains is the noise of the pixel.

With s = y + d(x, y) where a pixel is along the sky's spectrum, d being how far a feature is
from where it is at the middle of the slit:

    sky(x, y) = S(s) f(x)

S a spline with knots every fraction of a pixel, and f the slit's illumination, found in turn.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import advice


def slit_columns(data) -> tuple[int, int]:
    """The columns the slit lights, 0-based and inclusive: the widest run above half of what is
    typical of the columns clearly above the dark ones at the frame's edges. A median, as a
    star on the slit is far above the rest of it."""
    profile = np.median(np.asarray(data, float), axis=0)
    dark = profile[profile <= np.percentile(profile, 25)]
    low = float(np.median(dark))
    spread = 1.4826 * float(np.median(np.abs(dark - low)))
    above = profile[profile - low > 8.0 * spread]
    plateau = float(np.median(above)) if above.size else float(profile.max())
    lit = profile - low > 0.5 * (plateau - low)
    best, start, run = (0, profile.size - 1), None, 0
    for at, on in enumerate(np.r_[lit, False]):
        if on and start is None:
            start = at
        elif not on and start is not None:
            if at - start > run:
                run, best = at - start, (start, at - 1)
            start = None
    return best


def object_columns(data, slit: tuple[int, int], sigma: float = 5.0,
                   grow: int = 2) -> np.ndarray:
    """What of the slit has an object on it: columns above the profile's level by `sigma` of
    how the columns differ, and those next to them."""
    profile = np.median(np.asarray(data, float)[:, slit[0]:slit[1] + 1], axis=0)
    level = float(np.median(profile))
    spread = 1.4826 * float(np.median(np.abs(profile - level))) or 1.0
    bright = profile > level + sigma * spread
    if grow:
        bright = np.convolve(bright, np.ones(2 * grow + 1), mode="same") > 0
    return bright


def _shift(reference, band, reach: int, least: float = 0.0) -> float | None:
    """How far `band` is along from `reference`: the peak of their correlation, to a fraction
    by the parabola through its top. None if it is no peak, or under `least` of what a perfect
    match gives, as where a piece of the spectrum has only noise."""
    a = reference - np.median(reference)
    b = band - np.median(band)
    if not np.any(a) or not np.any(b):
        return None
    lags = np.arange(-reach, reach + 1)
    inner = a[reach:-reach]
    values = np.array([np.dot(inner, np.roll(b, -lag)[reach:-reach]) for lag in lags])
    top = int(np.argmax(values))
    if top == 0 or top == lags.size - 1 or values[top] <= 0:
        return None
    if values[top] < least * np.sqrt((inner ** 2).sum() * (b[reach:-reach] ** 2).sum()):
        return None
    left, middle, right = values[top - 1:top + 2]
    denominator = left - 2 * middle + right
    return float(lags[top] + (0.5 * (left - right) / denominator if denominator else 0.0))


@dataclass
class Tilt:
    """d(x, y) = sum c[i, j] u^(i+1) v^j, with u and v over the slit and the dispersion in
    -1 to 1: no term in u^0, as a feature is where it is at the slit's middle."""

    coefficients: np.ndarray
    middle: float
    half_slit: float
    half_length: float
    rms: float = 0.0
    measured: int = 0
    reach: tuple[float, float] = (-1.0, 1.0)     #: of v, where there were features

    def __call__(self, x, y):
        u = (np.asarray(x, float) - self.middle) / self.half_slit
        # No further along than its features went: a polynomial wanders off beyond them
        v = np.clip((np.asarray(y, float) - self.half_length) / self.half_length, *self.reach)
        total = np.zeros(np.broadcast(u, v).shape)
        for i in range(self.coefficients.shape[0]):
            for j in range(self.coefficients.shape[1]):
                total = total + self.coefficients[i, j] * u ** (i + 1) * v ** j
        return total

    @property
    def largest(self) -> float:
        """How far a feature goes across the slit, at the ends, pixels."""
        length = 2 * self.half_length
        return float(max(abs(self(self.middle + side * self.half_slit, along))
                         for side in (-1.0, 1.0) for along in (0.0, length / 2, length)))


def measure_tilt(frame, slit: tuple[int, int], xorder: int = 2, yorder: int = 2,
                 bands: int = 8, blocks: int = 8, reach: int = 8, objects=None) -> Tilt:
    """From what features a frame has, an arc's or the sky's: bands across the slit against the
    middle one, block by block along the dispersion. `objects`, the columns of the slit with
    an object on, count for neither."""
    frame = np.asarray(frame, float)
    rows = frame.shape[0]
    edges = np.linspace(slit[0], slit[1] + 1, bands + 1).astype(int)
    centres = 0.5 * (edges[:-1] + edges[1:] - 1)
    middle = 0.5 * (slit[0] + slit[1])
    half_slit = max(0.5 * (slit[1] - slit[0]), 1.0)
    half_length = 0.5 * (rows - 1)
    free = np.ones(frame.shape[1], bool)
    if objects is not None:
        free[slit[0]:slit[1] + 1] = ~np.asarray(objects, bool)

    def band_of(a: int, b: int):
        chosen = np.arange(a, b)[free[a:b]]
        return np.median(frame[:, chosen], axis=1) if chosen.size else None

    columns = [band_of(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    # The free columns nearest the middle, which is where d is zero
    near = np.argsort(np.abs(np.arange(slit[0], slit[1] + 1) - middle), kind="stable")
    near = near[free[slit[0]:slit[1] + 1][near]][:3] + slit[0]
    central = np.median(frame[:, near], axis=1)
    middle = float(near.mean())
    length = rows // blocks
    points = []
    for block in range(blocks):
        a, b = block * length, (block + 1) * length
        # Where in the block its features are, not where the block is
        part = central[a:b] - np.median(central[a:b])
        place = a + (part ** 2 * np.arange(b - a)).sum() / max((part ** 2).sum(), 1e-30)
        for at, band in enumerate(columns):
            shift = None if band is None else _shift(central[a:b], band[a:b], reach, 0.5)
            if shift is not None:
                # A feature at y in the band is at y - shift along the sky's spectrum
                points.append((centres[at], place, -shift))
    if len(points) < (xorder * (yorder + 1)) + 2:
        advice.fail(f"{len(points)} measures of the tilt",
                    "the frame has too few features: give an arc with --tilt-frame")
    data = np.array(points)
    u = (data[:, 0] - middle) / half_slit
    v = (data[:, 1] - half_length) / half_length
    # The same tilt all along: features at few places would send a polynomial along the
    # dispersion wild between them, and the sky's own spline says the rest (_refined)
    design = np.column_stack([u ** (i + 1) for i in range(xorder)])
    keep = np.ones(data.shape[0], bool)
    solution = np.zeros(design.shape[1])
    for _ in range(5):
        solution = np.linalg.lstsq(design[keep], data[keep, 2], rcond=None)[0]
        left = data[:, 2] - design @ solution
        sigma = max(1.4826 * np.median(np.abs(left[keep])), 0.05)
        new = keep & (np.abs(left) < 3.5 * sigma)
        if new.sum() < design.shape[1] + 2 or (new == keep).all():
            break
        keep = new
    coefficients = np.zeros((xorder, yorder + 1))
    coefficients[:, 0] = solution
    return Tilt(coefficients, middle, half_slit, half_length, float(left[keep].std()),
                int(keep.sum()), (float(v[keep].min()), float(v[keep].max())))


def _fit_spline(s, values, weights, spacing: float):
    """A cubic spline in s, with knots every `spacing` where there are samples to hold them."""
    from scipy.interpolate import LSQUnivariateSpline

    order = np.argsort(s, kind="stable")
    s, values, weights = s[order], values[order], weights[order]
    # Each knot has to have samples between it and the next, or there is no solution
    knots = np.arange(s[0] + spacing, s[-1], spacing)
    counts = np.searchsorted(s, knots)
    keep, last = [], 0
    for knot, count in zip(knots, counts, strict=True):
        if count - last >= 6:
            keep.append(knot)
            last = count
    if keep and s.size - last < 6:
        keep.pop()
    return LSQUnivariateSpline(s, values, keep, w=weights, k=3, check_finite=False)


@dataclass
class SkyFit:
    sky: np.ndarray               #: at each pixel, zero off the slit
    subtracted: np.ndarray        #: the frame less it
    illumination: np.ndarray      #: f(x), one at the middle of the slit
    tilt: Tilt
    slit: tuple[int, int]
    objects: np.ndarray           #: columns of the slit left out
    rejected: int = 0
    rms: float = 0.0              #: of what remains of the sky pixels, in their noise
    used: int = 0


def noise(model, gain: float, rdnoise: float):
    """In counts, of a pixel of `model` counts, by the frame's gain and read noise."""
    return np.sqrt(np.maximum(model, 0.0) / gain + (rdnoise / gain) ** 2)


def _median_of_columns(values, use) -> np.ndarray:
    """Of each column's pixels in `use`, and NaN for a column with none."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.where(use, values, np.nan), axis=0)


def _refined(data, x, y, tilt: Tilt, use, gain, rdnoise, spacing, rounds: int):
    """A tenth of a pixel of tilt leaves several sigma on a sharp line, and a correlation of
    bands gets no closer. So the sky's own spline says it: where the residual follows the
    spline's slope, the tilt is off, data = S(s + e) = S(s) + S'(s) e, and e is a polynomial
    with no bins in it. The tilt comes back changed."""
    u = (x - tilt.middle) / tilt.half_slit
    v = np.clip((y - tilt.half_length) / tilt.half_length, *tilt.reach)
    powers = [(i, j) for i in range(tilt.coefficients.shape[0])
              for j in range(tilt.coefficients.shape[1])]
    s = y + tilt(x, y)
    # The slit's fall-off would pass for a tilt: each column at the level of the rest
    level = _median_of_columns(data, use)
    open_columns = np.isfinite(level) & (level > 0)
    wanted = np.polynomial.legendre.legfit(u[0, open_columns], level[open_columns], 3)
    fall = np.polynomial.legendre.legval(u[0], wanted)
    fall = np.where(fall > 0, fall / np.nanmedian(level[open_columns]), 1.0)
    data = data / fall[None, :]
    for _ in range(rounds):
        typical = np.maximum(np.median(data[use]), 1.0)
        weights = 1.0 / noise(np.maximum(data[use], typical), gain, rdnoise)
        spline = _fit_spline(s[use], data[use], weights, spacing)
        model = spline(s[use])
        slope = spline.derivative()(s[use])
        left = data[use] - model
        sigma = noise(np.maximum(model, 0.0), gain, rdnoise)
        good = np.abs(left) < 30.0 * sigma
        if good.sum() < 50:
            break
        scale = 1.0 / sigma[good]
        design = np.column_stack([(u[use] ** (i + 1) * v[use] ** j)[good] * slope[good] * scale
                                  for i, j in powers])
        change = np.linalg.lstsq(design, left[good] * scale, rcond=None)[0]
        for (i, j), value in zip(powers, change, strict=True):
            tilt.coefficients[i, j] += value
        s = y + tilt(x, y)
    return s


def fit(data, tilt: Tilt, slit: tuple[int, int], objects, gain: float = 1.0,
        rdnoise: float = 0.0, spacing: float = 0.5, order: int = 3, clip: float = 4.0,
        iterations: int = 5, mask=None, refine: int = 4) -> SkyFit:
    """`data` with its dispersion along the rows' index, `objects` the columns of the slit to
    leave out, `mask` pixels to. `refine` rounds make the tilt better from the sky itself, and
    it is changed."""
    data = np.asarray(data, float)
    rows, columns = data.shape
    y, x = np.mgrid[0:rows, 0:columns].astype(float)
    s = y + tilt(x, y)
    on_slit = np.zeros(columns, bool)
    on_slit[slit[0]:slit[1] + 1] = True
    open_columns = on_slit.copy()
    open_columns[slit[0]:slit[1] + 1] &= ~np.asarray(objects, bool)
    use = open_columns[None, :].repeat(rows, axis=0)
    if mask is not None:
        use &= ~np.asarray(mask, bool)
    use &= np.isfinite(data)
    if use.sum() < 100:
        advice.fail(f"{int(use.sum())} sky pixels on the slit",
                    "leave more of the slit out of --object, or check --slit")
    middle = 0.5 * (slit[0] + slit[1])
    half = max(0.5 * (slit[1] - slit[0]), 1.0)
    u = (x - middle) / half
    illumination = np.ones(columns)
    if refine:
        s = _refined(data, x, y, tilt, use, gain, rdnoise, spacing, refine)
    keep = use.copy()
    model = np.zeros(data.shape)
    spline = None
    for _ in range(iterations):
        level = illumination[x[keep].astype(int)]
        typical = np.maximum(np.median(data[keep]), 1.0)
        first = noise(typical if spline is None else model[keep], gain, rdnoise) / level
        weights = 1.0 / np.maximum(first, 1e-6)
        spline = _fit_spline(s[keep], data[keep] / level, weights, spacing)
        flat = spline(s.ravel()).reshape(data.shape)
        # What each column has against the spline, over the slit's columns that are open
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(flat > 0.05 * typical, data / flat, np.nan)
        wanted = _median_of_columns(ratio, keep)
        good = np.isfinite(wanted) & open_columns
        if good.sum() > order + 1:
            design = np.polynomial.legendre.legvander(u[0, good], order)
            coefficients = np.linalg.lstsq(design, wanted[good], rcond=None)[0]
            fitted = np.polynomial.legendre.legvander(u[0], order) @ coefficients
            illumination = fitted / np.polynomial.legendre.legval(0.0, coefficients)
        model = flat * illumination[None, :]
        sigma = noise(model, gain, rdnoise)
        keep = use & (np.abs(data - model) < clip * sigma)
    sky = np.where(on_slit[None, :], model, 0.0)
    left = (data - model)[keep] / noise(model, gain, rdnoise)[keep]
    return SkyFit(sky, data - sky, np.where(on_slit, illumination, 0.0), tilt, slit, objects,
                  int(use.sum() - keep.sum()), float(left.std()), int(use.sum()))


def by_lines(data, slit: tuple[int, int], objects) -> np.ndarray:
    """What a sky by lines across the slit gives, the median of each: to compare with."""
    data = np.asarray(data, float)
    open_columns = np.arange(slit[0], slit[1] + 1)[~np.asarray(objects, bool)]
    sky = np.median(data[:, open_columns], axis=1)
    return data - sky[:, None]


def sky_spectrum(data, dispaxis: int, columns: str = "") -> tuple[np.ndarray, str]:
    """The sky of a frame as one spectrum, the median across its columns: what `columns` says,
    or the slit with an object left out. Of a spectrum of one line, that line."""
    data = np.asarray(data, float)
    if data.ndim == 1 or 1 in data.shape:
        return data.ravel(), "one line"
    frame = data.T if dispaxis == 1 else data
    if columns:
        first, last = (int(word) - 1 for word in columns.split(":"))
        picked = np.arange(first, last + 1)
        said = columns
    else:
        slit = slit_columns(frame)
        objects = object_columns(frame, slit)
        picked = np.arange(slit[0], slit[1] + 1)[~objects]
        said = f"{slit[0] + 1}:{slit[1] + 1}, {int(objects.sum())} left out"
    return np.median(frame[:, picked], axis=1), said


def _flattened(spectrum, width: int) -> np.ndarray:
    from scipy.ndimage import median_filter

    flat = spectrum - median_filter(spectrum, size=width, mode="nearest")
    return flat / (flat.std() or 1.0)


@dataclass
class Shift:
    shift: float   #: pixels, of the frame's lines against the reference's
    error: float
    peak: float    #: the correlation, at its best: 1 for the same lines


def flexure(spectrum, reference, reach: int = 10, width: int = 61) -> Shift:
    """By the peak of the correlation of the lines, with each one's level off, and by the same
    on pieces of them for how well that is known."""
    a = _flattened(np.asarray(spectrum, float), width)
    b = _flattened(np.asarray(reference, float), width)
    found = _shift(b, a, reach)
    if found is None:
        advice.fail("no peak in the correlation within the reach",
                    "give a larger --reach, or check that both have lines")
    inner = a[reach:-reach], np.roll(a, -int(round(found)))[reach:-reach], b[reach:-reach]
    norm = np.sqrt((inner[1] ** 2).sum() * (inner[2] ** 2).sum()) or 1.0
    peak = float((inner[1] * inner[2]).sum() / norm)
    pieces = []
    for at in range(4):
        start, stop = at * a.size // 4, (at + 1) * a.size // 4
        if stop - start > 4 * reach:
            part = _shift(b[start:stop], a[start:stop], reach)
            if part is not None:
                pieces.append(part)
    error = float(np.std(pieces, ddof=1) / np.sqrt(len(pieces))) if len(pieces) > 1 else 0.0
    return Shift(found, error, peak)
