"""sensfunc, as IRAF's noao/onedspec/sensfunc makes it.

Each observation in the std file (one spectrum of one aperture) becomes a
Star: its bands' wavelengths, true fluxes, widths and counts, and from them
the sensitivity of each band in magnitudes,

    S = 2.5 log10(counts / (flux * width * exptime)) + airmass * k(lambda),

which, with the extinction taken out, belongs to the instrument. One curve
(curfit, in IRAF's single precision) is fitted through the stars of each
aperture and sampled into the sensitivity spectrum calibrate applies.

Before that the stars can be worked on, from the keys of IRAF's graphs or
a cursor file: points, whole stars or whole wavelengths deleted, restored,
moved or reweighted; points added; every star shifted to a common mean
(grey shift: clouds or a slit that lost light); the stars combined into one
composite per wavelength; a residual extinction fitted from the spread in
airmass and taken out. What is left is fitted when asked (f, g) and
written at q -- the fit as it stands then, as in IRAF.

From iraf-community 2.18: noao/onedspec/sensfunc/*.x, pkg/xtools/xtsort.x,
sys/etc/qsort.x. See LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import advice, flux
from .curfit import NO_DEG_FREEDOM, OK, SINGULAR, Curve

f32 = np.float32
f64 = np.float64
INCLUDE, EXCLUDE, DELETE = 1, 2, 3
RANGE_AIRMASS, SIGMA_AIRMASS = 0.1, 0.05
NEPOINTS = 100
MAX_REAL = 3.4028234663852886e38

#: sf_graph's viewports for one to four graphs, (x1, x2, y1, y2) each.
VIEWPORTS = {1: [(.10, .97, .1, .9)],
             2: [(.10, .97, .55, .9), (.10, .97, .10, .45)],
             3: [(.10, .97, .55, .9), (.10, .50, .1, .45), (.57, .97, .1, .45)],
             4: [(.10, .50, .55, .9), (.57, .97, .55, .9), (.10, .50, .1, .45),
                 (.57, .97, .1, .45)]}

KEYS = """SENSFUNC: Determine Sensitivity Function

? Help            a Add data         c Composite data      d Delete data
e Extinction      f Fit (overplot)   g Fit (redraw)        i Info
m Move data       o Original data    q Quit                r Redraw
s Shift data      u Undelete data    w Change weights      I Interrupt

:function [type]    :graphs [types]     :order [value]
:stats [file]       :vstats [file]

d, u, m and w ask p(oint), s(tar) or w(avelength) with the next key."""


@dataclass
class Star:
    """One observation of a standard (SENSFUNC's standard structure)."""

    image: str
    sky: str = ""
    title: str = ""
    flag: int = INCLUDE
    beam: int = 1
    npts: int = 0
    exptime: f32 = f32(1)
    airmass: f32 = f32(1)
    wstart: f32 = f32(0)
    wend: f32 = f32(0)
    shift: f32 = f32(0)
    waves: np.ndarray = field(default_factory=lambda: np.zeros(0, f32))
    fluxes: np.ndarray = field(default_factory=lambda: np.zeros(0, f32))
    dwaves: np.ndarray = field(default_factory=lambda: np.zeros(0, f32))
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0, f32))
    sens: np.ndarray = field(default_factory=lambda: np.zeros(0, f32))
    fit: np.ndarray = field(default_factory=lambda: np.zeros(0, f32))
    wts: np.ndarray = field(default_factory=lambda: np.zeros(0, f32))
    iwts: np.ndarray = field(default_factory=lambda: np.zeros(0, f32))
    x: np.ndarray = field(default_factory=lambda: np.zeros(0, f32))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0, f32))

    @property
    def nwaves(self) -> int:
        return int(self.waves.size)

    def set_bands(self, waves, fluxes, dwaves, counts) -> None:
        self.waves = np.asarray(waves, f32)
        self.fluxes = np.asarray(fluxes, f32)
        self.dwaves = np.asarray(dwaves, f32)
        self.counts = np.asarray(counts, f32)
        n = self.waves.size
        self.wts = np.ones(n, f32)
        self.iwts = self.wts.copy()
        self.sens = np.zeros(n, f32)
        self.fit = np.zeros(n, f32)
        self.x = np.zeros(n, f32)
        self.y = np.zeros(n, f32)


# --------------------------------------------------------- reading


_INDEF = object()


def _real(word: str, indef=None):
    """gargr: a real, `indef` for INDEF, None when it is not a number."""
    from .iraffmt import INDEF, ctod

    value = ctod(word)
    if value is None:
        return None
    return indef if abs(value) >= INDEF else f32(value)


def read_iraf(text: str, aps, ignoreaps: bool, say=None) -> list[Star]:
    """sf_stds: the std file IRAF's standard writes, the observations of
    the apertures in `aps`, then the Added and Composite stars. As in IRAF,
    a header with no bands after it is taken over by the next one."""
    say = say or (lambda line: None)
    stars: list[Star] = []
    rows: list[tuple] = []
    beam, wstart, wend = 0, f32(0), f32(0)

    def finish():
        if stars and rows:
            w, f, d, c = zip(*rows, strict=True)
            stars[-1].set_bands(w, f, d, c)

    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        if words[0].startswith("["):
            values = [None] * 6
            for i, word in enumerate(words[1:7]):
                values[i] = (int(word) if i < 2 and word.lstrip("-").isdigit()
                             else _real(word, _INDEF) if i >= 2 else None)
            if None in values:
                continue
            beam, npts = values[0], values[1]
            exptime, airmass, wstart, wend = (None if v is _INDEF else v
                                              for v in values[2:])
            if beam not in aps:
                continue
            if exptime is None or exptime <= 0:
                say(f"{words[0]}: Warning - exposure time missing or zero, using 1 second")
                exptime = f32(1)
            if stars and rows:
                finish()
                stars.append(Star(""))
            elif not stars:
                stars.append(Star(""))
            star = stars[-1]
            star.beam = 1 if ignoreaps else beam
            star.npts, star.exptime, star.airmass = npts, exptime, airmass
            star.wstart, star.wend, star.shift = wstart, wend, f32(0)
            star.title = line.split(None, 7)[7] if len(words) > 7 else ""
            name = words[0]
            image, _, rest = name[1:].partition("]")
            star.image = image
            star.sky = rest[2:].partition("]")[0] if rest.startswith("-[") else ""
            star.set_bands([], [], [], [])
            rows = []
            continue
        if not stars:
            continue
        values = [_real(word) for word in words[:4]]
        if len(values) < 3 or None in values[:3]:
            continue
        wavelength, flam, dwave = values[:3]
        count = values[3] if len(values) > 3 and values[3] is not None else f32(0)
        if wavelength < min(wstart, wend) or wavelength > max(wstart, wend) or \
                flam <= 0 or dwave <= 0 or count <= 0:
            continue
        if beam not in aps:
            continue
        rows.append((wavelength, flam, dwave, count))
    finish()
    return _with_added(stars)


def from_measurements(standard, aps, ignoreaps: bool) -> list[Star]:
    """The same from harwood's file: its rows grouped into observations by
    image and aperture, counts per second per Angstrom taken back to the
    counts in each band of an exposure of one second."""
    stars: list[Star] = []
    rows: list = []
    key = None

    def finish():
        if rows:
            first = rows[0]
            star = Star(first.image.strip("[]").split("]-[")[0] if first.image else
                        first.star or "standard",
                        sky=first.image.split("]-[")[1].rstrip("]")
                        if "]-[" in first.image else "",
                        title=first.star, beam=1 if ignoreaps else first.aperture,
                        npts=first.npts or 0, exptime=f32(1),
                        airmass=f32(first.airmass),
                        wstart=f32(first.wstart if first.wstart is not None
                                   else min(m.wavelength for m in rows)),
                        wend=f32(first.wend if first.wend is not None
                                 else max(m.wavelength for m in rows)))
            star.set_bands([m.wavelength for m in rows], [m.flux for m in rows],
                           [m.width for m in rows],
                           [m.counts * m.width for m in rows])
            stars.append(star)

    for m in standard.measurements:
        if m.aperture not in aps:
            continue
        this = (m.image, m.aperture, m.airmass, m.star, m.wstart)
        if this != key:
            finish()
            rows, key = [], this
        if m.counts > 0 and m.width > 0 and m.flux > 0:
            rows.append(m)
    finish()
    return _with_added(stars)


def _with_added(stars: list[Star]) -> list[Star]:
    if not stars:
        return stars
    first = stars[0]
    for name in ("Added", "Composite"):
        stars.append(Star(name, beam=first.beam, npts=first.npts, exptime=f32(1),
                          airmass=f32(1), wstart=first.wstart, wend=first.wend))
    return stars


def apertures(stars: list[Star]) -> list[int]:
    """sf_apertures: the apertures in order of first appearance."""
    out: list[int] = []
    for star in stars:
        if star.beam not in out:
            out.append(star.beam)
    return out


# --------------------------------------------------------- sorting


def qsort(keys) -> list[int]:
    """sys/etc/qsort.x on an index (0-based), as xt_sort2/3 call it:
    IRAF's quicksort, whose order for equal keys decides the order of
    sums."""
    x = list(range(len(keys)))
    n = len(x)
    if n < 2:
        return x

    def compare(a, b):
        return -1 if keys[a] < keys[b] else (1 if keys[a] > keys[b] else 0)

    lv, uv = [0] * 40, [0] * 40
    lv[1], uv[1], p = 1, n, 1
    xs = [None] + x                     # 1-based
    while p > 0:
        if lv[p] >= uv[p]:
            p -= 1
            continue
        i, j = lv[p] - 1, uv[p]
        k = (lv[p] + uv[p]) // 2
        xs[j], xs[k] = xs[k], xs[j]
        pivot = xs[j]
        while i < j:
            i += 1
            while compare(xs[i], pivot) < 0:
                i += 1
            j -= 1
            while j > i:
                if compare(xs[j], pivot) <= 0:
                    break
                j -= 1
            if i < j:
                xs[i], xs[j] = xs[j], xs[i]
        j = uv[p]
        xs[i], xs[j] = xs[j], xs[i]
        if i - lv[p] < uv[p] - i:
            lv[p + 1], uv[p + 1], lv[p] = lv[p], i - 1, i + 1
        else:
            lv[p + 1], uv[p + 1], uv[p] = i + 1, uv[p], i - 1
        p += 1
    return xs[1:]


# --------------------------------------------------------- the data


def extinction(curve, wave) -> f32:
    if curve is None:
        return f32(0)
    return f32(flux.extinction_at(float(f32(wave)), curve))


def reset(stars: list[Star], curve) -> None:
    """sf_reset: flags, sensitivities and weights back to what was read."""
    for star in stars[:-2]:
        if star.flag == EXCLUDE:
            continue
        star.flag, star.shift = INCLUDE, f32(0)
        for j in range(star.nwaves):
            ext = extinction(curve, star.waves[j])
            s = f32(star.counts[j] / f32(f32(star.fluxes[j] * star.dwaves[j])
                                          * star.exptime))
            star.sens[j] = f32(2.5 * math.log10(f64(s)) + f64(f32(star.airmass * ext))) \
                if s > 0 else f32(np.nan)
        star.wts = star.iwts.copy()
    added, composite = stars[-2], stars[-1]
    added.set_bands([], [], [], [])
    added.flag, composite.flag = DELETE, EXCLUDE


def fit(stars: list[Star], function: str, order: int, xmin, xmax) -> Curve:
    """sf_fit: one curve through the included stars, the order lowered
    until it can be solved, as IRAF lowers it."""
    from .curfit import CHEBYSHEV, FUNCTIONS, SPLINE1, SPLINE3

    kind = FUNCTIONS.get(function, SPLINE3)
    ordr = order
    ier = OK
    cv = None
    while ordr > 0:
        cv = Curve(kind, ordr, xmin, xmax)
        for star in stars:
            if star.flag == INCLUDE:
                cv.accumulate(star.waves, star.sens, star.wts)
        ier = cv.solve()
        if ier == OK:
            break
        ordr -= 1
        if kind in (SPLINE3, SPLINE1) and ordr == 0:
            kind, ordr = CHEBYSHEV, (2 if kind == SPLINE3 else 1)
    if ier == SINGULAR:
        advice.fail("Singular solution",
                    "too few bands for the order: lower --order, or measure more "
                    "bands with standard.")
    if ier == NO_DEG_FREEDOM:
        advice.fail("No degrees of freedom",
                    "there are fewer bands than coefficients: lower --order.")
    for star in stars:
        if star.flag != EXCLUDE and star.nwaves >= 1:
            star.fit = cv.vector_at(star.waves)
    return cv


def rms(stars: list[Star]) -> tuple[float | None, int]:
    """sf_rms: the rms about the fit of the included points of non-zero
    weight, and how many."""
    total, npts = f32(0), 0
    for star in stars:
        if star.flag != INCLUDE:
            continue
        for j in range(star.nwaves):
            if star.wts[j] != 0:
                d = f32(star.sens[j] - star.fit[j])
                total = f32(total + f32(d * d))
                npts += 1
    return (float(f32(math.sqrt(total / f32(npts - 1)))) if npts > 1 else None), npts


def data(stars: list[Star], graph: str) -> None:
    """sf_data: each star's x and y on a graph (s, a or r)."""
    for star in stars:
        if star.flag != INCLUDE:
            continue
        if graph == "s":
            star.x, star.y = star.waves.copy(), star.sens.copy()
        elif graph == "a":
            star.x = np.full(star.nwaves, star.airmass, f32)
            star.y = star.sens - star.fit
        elif graph == "r":
            star.x, star.y = star.waves.copy(), star.sens - star.fit


def shift(stars: list[Star], shifted: bool) -> tuple[bool, str]:
    """sf_shift: every star moved to a mean residual of zero (and all up
    by the largest downward shift), or back."""
    if shifted:
        for star in stars:
            if star.flag == EXCLUDE or not star.nwaves:
                continue
            star.sens = star.sens - star.shift
            star.shift = f32(0)
        return False, "Data unshifted"
    minshift = f32(0)
    for star in stars:
        if star.flag == EXCLUDE or not star.nwaves:
            continue
        nshift, s, s1 = 0, f32(0), f32(0)
        for j in range(star.nwaves):
            d = f32(star.fit[j] - star.sens[j])
            s1 = f32(s1 + d)
            if star.wts[j] > 0:
                s = f32(s + d)
                nshift += 1
        if nshift > 0:
            s = f32(star.shift + f32(s / f32(nshift)))
            minshift = min(minshift, s)
        else:
            s = f32(star.shift + f32(s1 / f32(star.nwaves)))
        star.shift = s
    for star in stars:
        if star.flag == EXCLUDE or not star.nwaves:
            continue
        s = f32(star.shift - minshift)
        star.sens = star.sens + s
        star.shift = s
    return True, "Data shifted"


def composite(stars: list[Star], cv: Curve) -> str:
    """sf_composite: the stars' points combined, weighted, per wavelength
    into the Composite star, the stars turned off -- or back."""
    comp = stars[-1]
    if comp.flag == INCLUDE:
        for star in stars[:-2]:
            if star.flag != EXCLUDE:
                star.flag = INCLUDE
        comp.flag = EXCLUDE
        for i in range(comp.nwaves):
            if comp.wts[i] == 0:
                for star in stars[:-2]:
                    if star.flag == EXCLUDE:
                        continue
                    star.wts[star.waves == comp.waves[i]] = f32(0)
        return "Individual star data"
    waves, sens, wts = [], [], []
    for star in stars[:-2]:
        if star.flag != INCLUDE:
            continue
        for j in range(star.nwaves):
            if star.wts[j] != 0:
                waves.append(star.waves[j])
                sens.append(star.sens[j])
                wts.append(star.wts[j])
        star.flag = DELETE
        comp.beam, comp.wstart, comp.wend = star.beam, star.wstart, star.wend
    order = qsort(waves)
    waves = [waves[i] for i in order]
    sens = [sens[i] for i in order]
    wts = [wts[i] for i in order]
    n = 0
    if waves:
        sens[0] = f32(wts[0] * sens[0])
    for i in range(1, len(waves)):
        if waves[i] == waves[n]:
            sens[n] = f32(sens[n] + f32(wts[i] * sens[i]))
            wts[n] = f32(wts[n] + wts[i])
        else:
            n += 1
            waves[n] = waves[i]
            sens[n] = f32(wts[i] * sens[i])
            wts[n] = wts[i]
    count = n + 1 if waves else 0
    comp.waves = np.array(waves[:count], f32)
    comp.sens = np.array([f32(sens[i] / wts[i]) for i in range(count)], f32)
    comp.wts = np.array(wts[:count], f32)
    comp.iwts = comp.wts.copy()
    comp.fit = cv.vector_at(comp.waves)
    comp.x = np.zeros(count, f32)
    comp.y = np.zeros(count, f32)
    comp.fluxes = comp.dwaves = comp.counts = np.zeros(count, f32)
    comp.flag = INCLUDE
    return "Composite star data"


def add(stars: list[Star], cv: Curve, graph: str, wx, wy) -> bool:
    """sf_add: a point at the cursor of graph s or r, into the Added star."""
    if graph == "s":
        wave, fitted, sen = f32(wx), cv(wx), f32(wy)
    elif graph == "r":
        wave, fitted = f32(wx), cv(wx)
        sen = f32(f32(wy) + fitted)
    else:
        return False
    star = stars[-2]
    star.flag = INCLUDE
    star.waves = np.append(star.waves, wave).astype(f32)
    star.sens = np.append(star.sens, sen).astype(f32)
    star.fit = np.append(star.fit, fitted).astype(f32)
    star.wts = np.append(star.wts, f32(1)).astype(f32)
    star.iwts = np.append(star.iwts, f32(1)).astype(f32)
    star.x = np.append(star.x, f32(0)).astype(f32)
    star.y = np.append(star.y, f32(0)).astype(f32)
    for name in ("fluxes", "dwaves", "counts"):
        setattr(star, name, np.append(getattr(star, name), f32(0)).astype(f32))
    return True


def delete(stars: list[Star], key: str, istd: int, ipt: int) -> None:
    """sf_delete: a point, a star or a wavelength given weight zero."""
    if key == "p":
        stars[istd].wts[ipt] = f32(0)
    elif key == "s":
        stars[istd].wts[:] = f32(0)
    elif key == "w":
        wave = stars[istd].waves[ipt]
        for star in stars:
            if star.flag == INCLUDE:
                star.wts[star.waves == wave] = f32(0)


def undelete(stars: list[Star], key: str, istd: int, ipt: int) -> None:
    """sf_undelete: the weights given back."""
    if key == "p":
        stars[istd].wts[ipt] = stars[istd].iwts[ipt]
    elif key == "s":
        stars[istd].wts = stars[istd].iwts.copy()
    elif key == "w":
        wave = stars[istd].waves[ipt]
        for star in stars:
            if star.flag == INCLUDE:
                same = star.waves == wave
                star.wts[same] = star.iwts[same]


def move(stars: list[Star], key: str, istd: int, ipt: int, delta) -> None:
    """sf_move: sensitivities moved by delta. For a wavelength IRAF never
    steps through a star's wavelengths, so each star whose first band is at
    that wavelength has all its points moved, and the others none."""
    delta = f32(delta)
    if key == "p":
        stars[istd].sens[ipt] = f32(stars[istd].sens[ipt] + delta)
    elif key == "s":
        star = stars[istd]
        star.sens = star.sens + delta
        star.shift = f32(star.shift + delta)
    elif key == "w":
        wave = stars[istd].waves[ipt]
        for star in stars:
            if star.flag == INCLUDE and star.nwaves and star.waves[0] == wave:
                star.sens = star.sens + delta


def reweight(stars: list[Star], key: str, istd: int, ipt: int, weight) -> None:
    """sf_weights: a new weight, the original one lost."""
    weight = f32(weight)
    if key == "p":
        stars[istd].wts[ipt] = stars[istd].iwts[ipt] = weight
    elif key == "s":
        stars[istd].wts[:] = weight
        stars[istd].iwts[:] = weight
    elif key == "w":
        wave = stars[istd].waves[ipt]
        for star in stars:
            if star.flag == INCLUDE:
                same = star.waves == wave
                star.wts[same] = weight
                star.iwts[same] = weight


# --------------------------------------------------------- residual extinction


@dataclass
class Residual:
    """sf_extinct's result: the residual extinction curve and how
    significant taking it out is."""

    curve: Curve | None
    significance: float = 0.0
    message: str = ""


def residual_extinction(stars: list[Star], cv_args, ecv_order: int = 1,
                        ecv_function: str = "chebyshev") -> Residual:
    """sf_extinct up to the question: the slope of residual sensitivity
    against airmass at each wavelength (or run of wavelengths) with enough
    airmass spread, a curve through those slopes, the sensitivity refitted
    with it applied, and the significance of the improvement. The stars
    are left with it applied; `cancel` takes it out."""
    included = [s for s in stars[:-2] if s.flag == INCLUDE]
    amin = min([f32(100)] + [s.airmass for s in included])
    amax = max([f32(0)] + [s.airmass for s in included])
    if amax - amin < RANGE_AIRMASS:
        return Residual(None, 0.0,
                        "Insufficient airmass coverage for extinction determination")
    waves, sens, airm = [], [], []
    for star in included:
        for j in range(star.nwaves):
            if star.wts[j] != 0:
                airm.append(star.airmass)
                waves.append(star.waves[j])
                sens.append(f32(star.sens[j] - star.fit[j]))
    nwaves = len(waves)
    order = qsort(waves)
    waves = [waves[i] for i in order]
    sens = [sens[i] for i in order]
    airm = [airm[i] for i in order]
    out_w, out_s = [], []
    acc = [0, 0.0, 0.0, 0.0, 0.0, 0.0]          # sum, x, y, z, x2, xy

    def flush():
        x = acc[4] - acc[1] * acc[1] / acc[0]
        if x > SIGMA_AIRMASS:
            out_w.append(f32(acc[3] / acc[0]))
            out_s.append(f32((acc[1] * acc[2] / acc[0] - acc[5]) / x))
            return True
        return False

    for i in range(nwaves - 1):
        x, y, z = f64(airm[i]), f64(sens[i]), f64(waves[i])
        acc[0] += 1
        acc[1] += x
        acc[2] += y
        acc[4] += x * x
        acc[5] += x * y
        acc[3] += z
        if z == waves[i + 1] or acc[0] < 2:
            continue
        if flush():
            acc[:] = [0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if acc[0] > 1:
        flush()
    if len(out_w) < 2:
        return Residual(None, 0.0, "Cannot determine residual extinction")
    # icg_fit: a polynomial is fitted over the range of its points (the
    # first star's range, which sf_extinct hands it, only bounds a spline).
    if ecv_function in ("chebyshev", "legendre"):
        low, high = min(out_w), max(out_w)
        ecv_order = min(ecv_order, len(out_w))
    else:
        first = stars[0]
        low = min(min(first.wstart, first.wend), min(out_w))
        high = max(max(first.wstart, first.wend), max(out_w))
    ecv = Curve(ecv_function, ecv_order, low, high)
    ecv.accumulate(out_w, out_s, np.ones(len(out_w), f32))
    ecv.solve()
    function, order_, xmin, xmax = cv_args
    fit(stars, function, order_, xmin, xmax)
    rms1, _ = rms(stars)
    for star in included:
        star.sens = star.sens + ecv.vector_at(star.waves) * star.airmass
    fit(stars, function, order_, xmin, xmax)
    rms2, _ = rms(stars)
    r2 = f32(1) - f32(rms2) ** 2 / f32(rms1) ** 2
    sig = f32(r2 * f32(nwaves - 2) / max(f32(0.01), f32(1) - r2))
    sig = 0.0 if sig <= 0 else float(f32(math.sqrt(sig)))
    return Residual(ecv, sig, "")


def cancel_extinction(stars: list[Star], ecv: Curve) -> None:
    """The cancel_ label of sf_extinct: the correction taken out again."""
    for star in stars[:-2]:
        if star.flag == EXCLUDE:
            continue
        for j in range(star.nwaves):
            star.sens[j] = f32(star.sens[j] - f32(star.airmass * ecv(star.waves[j])))


# --------------------------------------------------------- output


def sampling(stars: list[Star]) -> tuple[f32, f32, int, int]:
    """sf_output's grid: from the stars' first to last wavelength at their
    finest dispersion; and the aperture of the last one, which names it."""
    w1, w2, dw = f32(MAX_REAL), f32(-MAX_REAL), f32(MAX_REAL)
    beam = stars[0].beam
    for star in stars[:-2]:
        if star.flag == EXCLUDE:
            continue
        dw1 = f32(abs(f32(star.wend - star.wstart) / f32(star.npts - 1)))
        w1 = min(w1, star.wstart, star.wend)
        w2 = max(w2, star.wstart, star.wend)
        dw = min(dw, dw1)
        beam = star.beam
    nw = int(f32(f32(w2 - w1) / dw) + f32(1.5))
    return w1, dw, nw, beam


def sample(cv: Curve, w1, dw, nw) -> np.ndarray:
    return np.array([cv(f32(w1 + f32(f32(i) * dw))) for i in range(nw)], f32)


def revised_extinction(curve, ecvs: list[Curve | None]) -> list[str] | None:
    """sf_eout: the extinction table with the residual curves (averaged
    where they overlap) added, at 100 points; None without any."""
    ecvs = [e for e in ecvs if e is not None]
    if not ecvs:
        return None
    wmin = min(e.xmin for e in ecvs)
    wmax = max(e.xmax for e in ecvs)
    dw = f32(f32(wmax - wmin) / f32(NEPOINTS - 1))
    waves = [f32(wmin + f32(f32(i) * dw)) for i in range(NEPOINTS)]
    ext = [f32(0)] * NEPOINTS
    navg = [0] * NEPOINTS
    for e in ecvs:
        for i, w in enumerate(waves):
            if w < e.xmin or w > e.xmax:
                continue
            ext[i] = f32(ext[i] + e(w))
            navg[i] += 1
    lines = ["# Revised extinction table."]
    for i, w in enumerate(waves):
        if navg[i] > 0:
            ext[i] = f32(ext[i] / f32(navg[i]))
        ext[i] = f32(ext[i] + extinction(curve, w))
        lines.append(f"{float(w):7.2f}  {float(ext[i]):6.3f}")
    return lines


def _devs(star: Star):
    n = star.nwaves
    total, out, i = f32(0), [], 0
    count = 0
    for top in (n // 3, 2 * n // 3, n):
        dev, nd = f32(0), 0
        while i < top:
            if star.wts[i] != 0:
                d = f32(star.sens[i] - star.fit[i])
                dev = f32(dev + d)
                total = f32(total + f32(d * d))
                nd += 1
            i += 1
        out.append(f32(dev / f32(nd)) if nd else f32(0))
        count += nd
    r = f32(math.sqrt(total / f32(count))) if count else f32(0)
    return r, out


def stats(stars: list[Star], function: str, order: int, npts: int, fit_rms,
          system: str) -> list[str]:
    """sf_stats: the fit and each star's rms and mean deviation in thirds."""
    first = next((s for s in stars[:-1] if s.flag != EXCLUDE), stars[-1])
    lines = [system, f"Sensitivity function for aperture {first.beam}:",
             f"Fitting function is {function} of order {order} with {npts} points "
             f"and RMS of {_f(fit_rms, '6.4f')}.",
             f"{'Image':>12} {'Airmass':>7} {'Points':>7} {'Shift':>7} {'RMS Fit':>7} "
             f"{'Dev 1':>7} {'Dev 2':>7} {'Dev 3':>7}"]
    for i, star in enumerate(stars):
        if star.flag == EXCLUDE:
            continue
        n = int(np.count_nonzero(star.wts))
        if i == len(stars) - 2 and n == 0:
            continue
        r, (d1, d2, d3) = _devs(star)
        line = (f"{star.image[:63]:>12} {float(star.airmass):7.3f} {n:7d} "
                f"{float(star.shift):7.4f} {float(r):7.4f} {float(d1):7.4f} "
                f"{float(d2):7.4f} {float(d3):7.4f}")
        if n == 0:
            line += " <-- deleted"
        lines.append(line)
    lines.append("")
    return lines


def vstats(stars: list[Star], cv: Curve, curve, ecv: Curve | None) -> list[str]:
    """sf_vstats: per wavelength the fit, the mean sensitivity, the
    residual, the error of the mean, the extinction and its residual."""
    waves, sens = [], []
    for star in stars[:-1]:
        if star.flag == EXCLUDE:
            continue
        for j in range(star.nwaves):
            if star.wts[j] != 0:
                waves.append(star.waves[j])
                sens.append(star.sens[j])
    order = qsort(waves)
    waves = [waves[i] for i in order]
    sens = [sens[i] for i in order]
    lines = [f"{'Lambda':>8} {'Fit':>7} {'Avg':>7} {'Resid':>7} {'SD Avg':>7} "
             f"{'N':>5} {'Ext':>7} {'Dext':>7}"]
    n, total, total2 = 0, 0.0, 0.0
    for i, w in enumerate(waves):
        s = f64(sens[i])
        n += 1
        total += s
        total2 += s * s
        if i < len(waves) - 1 and w == waves[i + 1]:
            continue
        mean = total / n
        var = total2 / n - mean * mean
        sd = math.sqrt(var / n) if var > 0 else 0.0
        fitted = cv(w)
        ext = extinction(curve, w)
        dext = ecv(w) if ecv is not None else f32(0)
        lines.append(f"{float(w):8.2f} {float(fitted):7.3f} {mean:7.3f} "
                     f"{mean - float(fitted):7.4f} {sd:7.4f} {n:5d} "
                     f"{float(ext):7.4f} {float(dext):7.4f}")
        n, total, total2 = 0, 0.0, 0.0
    lines.append("")
    return lines


def _f(value, fmt: str) -> str:
    return "INDEF" if value is None else format(float(value), fmt)


# --------------------------------------------------------- the session


@dataclass
class Window:
    """One graph as sf_graph drew it: its type, viewport and window."""

    graph: str
    viewport: tuple
    window: tuple


@dataclass
class Graph:
    """What sf_graph and sf_fitgraph draw in one window. `marks`, in the order drawn,
    are (kind, x, y) with kind plus, cross or box; `lines` are (x, y, dashed); `bars`
    are the composite points' x, y and error, each drawn as a bar that tall in all,
    so half the error either way, as IRAF's gmark has it."""

    kind: str
    title: str
    window: tuple
    marks: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    bars: tuple | None = None


GRAPH_TITLES = {"s": "Sensitivity vs Wavelength", "r": "Sensitivity Residuals vs Wavelength",
                "a": "Sensitivity Residuals vs Airmass", "e": "Extinction vs Wavelength",
                "c": "Composite Points vs Wavelength",
                "i": "The spectrum, flux calibrated: not ported",
                "l": "The spectrum's log flux: not ported"}
NCURVE = 50


class Session:
    """sf_sensfunc's cursor loop on one aperture. Keys arrive with the
    cursor's position and the graph (wcs, 1-based) it was in; d, u, m and w
    take the next key as p, s or w; w then takes the new weight as the next
    ':' command's text, e takes keys for its fit until q and then the
    answer (y or n), where IRAF reads the terminal."""

    def __init__(self, stars: list[Star], curve, function: str, order: int,
                 graphs: str = "sr", system: str = "harwood") -> None:
        self.stars, self.curve = stars, curve
        self.function, self.order = function, order
        self.graphs = graphs
        self.system = system
        self.messages: list[str] = []
        self.files: dict[str, list[str]] = {}
        self.shifted = False
        self.ecv: Curve | None = None
        self.done = False
        self.pending: tuple | None = None
        reset(stars, curve)
        included = [s for s in stars[:-2] if s.flag != EXCLUDE]
        self.xmin = min(min(s.wstart, s.wend) for s in included)
        self.xmax = max(max(s.wstart, s.wend) for s in included)
        self.aperture = included[-1].beam
        self.cv = fit(stars, function, order, self.xmin, self.xmax)
        self.rms, self.npts = rms(stars)
        self.airmass = (min(s.airmass for s in included), max(s.airmass for s in included))
        delta = f32(self.airmass[1] - self.airmass[0])
        self.airmass = (f32(self.airmass[0] - f32(0.05) * delta),
                        f32(self.airmass[1] + f32(0.05) * delta))
        self.windows: list[Window] = []
        self.draw()

    # ------------------------------------------------------------ graphs

    def draw(self) -> None:
        """sf_graph's windows: x the wavelength (or airmass) range, y the
        included points' range and 5 per cent."""
        ports = VIEWPORTS[min(4, max(1, len(self.graphs)))]
        self.windows = []
        for graph, port in zip(self.graphs, ports, strict=False):
            self.windows.append(Window(graph, port, self._window(graph)))

    def _window(self, graph: str) -> tuple:
        if graph == "e":
            return self._extinction_curves()[0]
        if graph == "c":
            found = self._composite_points()
            return found[0] if found else (0, 1, 0, 1)
        if graph not in "asr":
            return (0, 1, 0, 1)
        data(self.stars, graph)
        ys = [s.y for s in self.stars if s.flag == INCLUDE and s.nwaves]
        ymin = min(f32(np.min(y)) for y in ys)
        ymax = max(f32(np.max(y)) for y in ys)
        d = f32(f32(0.05) * f32(ymax - ymin))
        x1, x2 = (self.airmass if graph == "a" else (self.xmin, self.xmax))
        return (x1, x2, f32(ymin - d), f32(ymax + d))

    def _extinction_curves(self) -> tuple:
        """sf_egraph: the window, and the extinction and, after e, what it became."""
        dx = f32(f32(self.xmax - self.xmin) / f32(NCURVE - 1))
        x = self.xmin
        grid, table, changed = [], [], []
        for _ in range(NCURVE):
            value = extinction(self.curve, x)
            grid.append(x)
            table.append(value)
            if self.ecv is not None:
                changed.append(f32(value + f32(self.ecv(x))))
            x = f32(x + dx)
        everything = table + changed
        window = (self.xmin, self.xmax, min(everything), max(everything))
        grid = np.linspace(float(self.xmin), float(self.xmax), NCURVE)
        return window, grid, np.array(table, dtype=f32), np.array(changed, dtype=f32)

    def _composite_points(self):
        """sf_cgraph: the window and the points, or None if none overlap."""
        waves, sens = [], []
        for star in self.stars[:-2]:
            if star.flag == EXCLUDE:
                continue
            kept = star.wts[:star.nwaves] != 0
            waves.append(star.waves[:star.nwaves][kept])
            sens.append(star.sens[:star.nwaves][kept])
        waves, sens = np.concatenate(waves), np.concatenate(sens)
        order = np.argsort(waves, kind="stable")
        waves, sens = waves[order], sens[order]
        xs, ys, errors = [], [], []
        ymin, ymax = 0.0, 0.0
        for wave in np.unique(waves):
            here = sens[waves == wave]
            if here.size <= 2:
                continue
            n = here.size
            total = float(np.sum(here.astype(np.float64)))
            squares = float(np.sum((here * here).astype(f32).astype(np.float64)))
            mean = total / n
            spread = squares / n - mean * mean
            spread = math.sqrt(spread / n) if spread > 0 else 0.0
            mean = mean - float(self.cv(wave))
            xs.append(wave)
            ys.append(f32(mean))
            errors.append(f32(spread))
            ymax = max(ymax, mean + spread)
            ymin = min(ymin, mean - spread)
        if not xs:
            return None
        window = (self.xmin, self.xmax, f32(ymin), f32(ymax))
        return (window, np.array(xs, dtype=f32), np.array(ys, dtype=f32),
                np.array(errors, dtype=f32))

    def drawn(self) -> list[Graph]:
        """Every graph as it stands, for the panel, in sf_graph's order."""
        return [self._graph(window) for window in self.windows]

    def _graph(self, window: Window) -> Graph:
        kind = window.graph
        graph = Graph(kind, GRAPH_TITLES[kind], window.window)
        if kind in "asr":
            data(self.stars, kind)
            for i, star in enumerate(self.stars):
                if star.flag != INCLUDE:
                    continue
                mark = "box" if i == len(self.stars) - 2 else "plus"
                for j in range(star.nwaves):
                    graph.marks.append(("cross" if star.wts[j] == 0 else mark,
                                        star.x[j], star.y[j]))
            if kind == "s":
                graph.lines.append((*self._fit_curve(window), False))
        elif kind == "e":
            _, grid, table, changed = self._extinction_curves()
            graph.lines.append((grid, table, False))
            if self.ecv is not None:
                graph.lines.append((grid, changed, True))
        elif kind == "c":
            found = self._composite_points()
            if found is not None:
                graph.bars = found[1:]
        return graph

    def _fit_curve(self, window: Window) -> tuple[np.ndarray, np.ndarray]:
        """sf_fitgraph."""
        x, right = window.window[:2]
        step = f32(f32(right - x) / f32(NCURVE))
        xs = [x]
        for _ in range(NCURVE):
            x = f32(x + step)
            xs.append(x)
        xs = np.array(xs, dtype=f32)
        return xs, np.asarray(self.cv.vector_at(xs), dtype=f32)

    def title(self) -> str:
        """sf_title."""
        rms_text = "INDEF" if self.rms is None else f"{self.rms:6.4f}"
        return (f"Aperture={self.aperture}  Function={self.function}  Order={self.order}  "
                f"Points={self.npts}  RMS={rms_text}")

    def _ndc(self, window: Window, x, y):
        vx1, vx2, vy1, vy2 = (f32(v) for v in window.viewport)
        wx1, wx2, wy1, wy2 = window.window
        nx = f32(vx1 + f32(f32(f32(x) - wx1) * f32(f32(vx2 - vx1) / f32(wx2 - wx1))))
        ny = f32(vy1 + f32(f32(f32(y) - wy1) * f32(f32(vy2 - vy1) / f32(wy2 - wy1))))
        return nx, ny

    def nearest(self, wx, wy, wcs: int, kind: int):
        """sf_nearest: the point nearest the cursor in NDC, of the deleted
        ones (kind 1), the others (0) or any (2); (star, point) or None."""
        if wcs < 1 or wcs > len(self.windows) or self.windows[wcs - 1].graph not in "ars":
            return None
        window = self.windows[wcs - 1]
        data(self.stars, window.graph)
        cx, cy = self._ndc(window, wx, wy)
        best, found = MAX_REAL, None
        for i, star in enumerate(self.stars):
            if star.flag != INCLUDE:
                continue
            for j in range(star.nwaves):
                if kind == 0 and star.wts[j] == 0:
                    continue
                if kind == 1 and star.wts[j] != 0:
                    continue
                x0, y0 = self._ndc(window, star.x[j], star.y[j])
                r2 = f32(f32(f32(x0 - cx) ** 2) + f32(f32(y0 - cy) ** 2))
                if r2 < best:
                    best, found = r2, (i, j)
        return found

    # ------------------------------------------------------------ keys

    def refit(self) -> None:
        self.cv = fit(self.stars, self.function, self.order, self.xmin, self.xmax)
        self.rms, self.npts = rms(self.stars)

    def key(self, wx, wy, wcs: int, key: str, command: str = "") -> None:
        if self.pending is not None:
            self._second(wx, wy, wcs, key, command)
            return
        graph = self.windows[wcs - 1].graph if 1 <= wcs <= len(self.windows) else ""
        newfit = newgraph = False
        if key == "?":
            self.messages.append(KEYS)
        elif key == ":":
            newfit, newgraph = self._colon(command)
        elif key == "a":
            add(self.stars, self.cv, graph, wx, wy)
        elif key == "c":
            self.messages.append(composite(self.stars, self.cv))
            newfit = newgraph = True
        elif key in "dumw" and key:
            kind = {"d": 0, "u": 1, "m": 2, "w": 0}[key]
            found = self.nearest(wx, wy, wcs, kind)
            if found is not None:
                self.pending = (key, found, wcs)
                verb = {"d": "Delete", "u": "Undelete", "m": "Move", "w": "Reweight"}[key]
                self.messages.append(f"{self.stars[found[0]].image} - {verb} p(oint), "
                                     "s(tar), or w(avelength)" +
                                     (" to cursor:" if key == "m" else ":"))
        elif key == "e":
            if self.ecv is not None:
                cancel_extinction(self.stars, self.ecv)
                self.ecv = None
                self.messages.append("Residual extinction correction canceled")
                newfit = newgraph = True
            else:
                result = residual_extinction(self.stars, (self.function, self.order,
                                                          self.xmin, self.xmax))
                if result.curve is None:
                    self.messages.append(result.message)
                    newfit = newgraph = True
                else:
                    self.pending = ("e", result, None)
                    self.messages.append(f"Significance = {result.significance:4.1f} "
                                         "sigma:  Apply residual extinction correction? ")
                    return
        elif key == "f":
            newfit = True
        elif key == "g":
            newfit = newgraph = True
        elif key == "i":
            found = self.nearest(wx, wy, wcs, 2)
            if found is not None:
                star, j = self.stars[found[0]], found[1]
                self.messages.append(
                    f"{star.image}: airmass={float(star.airmass):6.3f} "
                    f"wavelen={float(star.waves[j]):6.3f} sens={float(star.sens[j]):6.3f} "
                    f"fit={float(star.fit[j]):6.3f} weight={float(star.wts[j]):3f}")
        elif key == "o":
            reset(self.stars, self.curve)
            self.shifted = False
            self.ecv = None
            newfit = newgraph = True
        elif key == "q":
            self.done = True
            return
        elif key == "I":
            advice.fail("Interrupt", "sensfunc was interrupted from the cursor.")
        elif key == "r":
            newgraph = True
        elif key == "s":
            self.shifted, message = shift(self.stars, self.shifted)
            self.messages.append(message)
            newfit = newgraph = True
        else:
            self.messages.append("\a")
        self._after(newfit, newgraph)

    def _after(self, newfit: bool, newgraph: bool) -> None:
        if newfit:
            self.refit()
        if newgraph:
            self.draw()

    def _second(self, wx, wy, wcs, key, command) -> None:
        what, found, extra = self.pending
        if what == "e":
            if key == "q" or (key == ":" and command):
                pass
            if key in ("y", "n") or (key == ":" and command.strip().lower() in
                                     ("yes", "no", "y", "n")):
                self.pending = None
                answer = (key if key != ":" else command.strip().lower())[0] == "y"
                result = found
                if answer:
                    self.ecv = result.curve
                    self.messages.append("Residual extinction correction applied")
                else:
                    cancel_extinction(self.stars, result.curve)
                    self.messages.append("Residual extinction correction canceled")
                self._after(True, True)
            return
        istd, ipt = found
        if what == "w":
            if not isinstance(extra, tuple):
                self.pending = ("w", found, (key,))
                self.messages.append(f"New weight ({float(self.stars[istd].iwts[ipt]):g}):")
                return
            self.pending = None
            sub = extra[0]
            try:
                weight = float(command if key == ":" else key)
            except ValueError:
                return
            if sub in "psw":
                reweight(self.stars, sub, istd, ipt, weight)
            return
        self.pending = None
        if key not in "psw" or not key:
            return
        if what == "d":
            delete(self.stars, key, istd, ipt)
        elif what == "u":
            undelete(self.stars, key, istd, ipt)
        elif what == "m":
            # y as sf_data left it for the graph the m was typed in.
            data(self.stars, self.windows[extra - 1].graph)
            delta = f32(f32(wy) - self.stars[istd].y[ipt])
            move(self.stars, key, istd, ipt, delta)

    def _colon(self, command: str) -> tuple[bool, bool]:
        words = command.split()
        if not words:
            return False, False
        name, rest = words[0], words[1:]
        names = ("stats", "vstats", "function", "order", "graphs", "images", "skys",
                 "marks", "fluxlimits", "colors")
        matches = [n for n in names if n.startswith(name)]
        name = name if name in names else (matches[0] if len(matches) == 1 else "")
        if name in ("stats", "vstats"):
            lines = stats(self.stars, self.function, self.order, self.npts, self.rms,
                          self.system)
            if name == "vstats":
                lines += vstats(self.stars, self.cv, self.curve, self.ecv)
            if rest:
                self.files.setdefault(rest[0], []).extend(lines)
            else:
                self.messages.append("\n".join(lines))
        elif name == "function":
            if rest:
                self.function = rest[0]
            else:
                self.messages.append(f"function {self.function}")
        elif name == "order":
            if rest:
                self.order = int(rest[0])
            else:
                self.messages.append(f"order {self.order}")
        elif name == "graphs":
            kept = "".join(c for c in "".join(rest) if c in "aceilrs")
            if kept:
                self.graphs = kept[:4]
                return False, True
            self.messages.append(f"graphs {self.graphs}")
        elif name in ("images", "skys", "marks", "fluxlimits", "colors"):
            pass
        else:
            self.messages.append("Unrecognized or ambiguous command\a")
        return False, False


class CursorFile:
    """IRAF's cursor file, 'x y graph key' or 'x y graph :command' per line,
    read across the apertures as clgcur reads it; the weight w asks for and
    the answer e asks for come from the next line's ':' text."""

    def __init__(self, path) -> None:
        from pathlib import Path

        self.events = []
        for line in Path(path).read_text().splitlines():
            words = line.split(None, 3)
            if len(words) < 4 or line.lstrip().startswith("#"):
                continue
            key = {"\\040": " ", "\\40": " "}.get(words[3].strip(), words[3].strip())
            wcs = int(float(words[2]))
            if key.startswith(":"):
                self.events.append((float(words[0]), float(words[1]), wcs, ":", key[1:]))
            else:
                self.events.append((float(words[0]), float(words[1]), wcs, key[:1], ""))
        self.position = 0

    def edit(self, session: Session) -> None:
        while not session.done and self.position < len(self.events):
            x, y, wcs, key, command = self.events[self.position]
            self.position += 1
            session.key(x, y, wcs, key, command)
