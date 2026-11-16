"""telluric and skytweak: a calibration spectrum shifted and scaled to remove telluric
bands (divided, raised to the power of the airmass ratio) or sky lines (subtracted)
from each spectrum, the shift and scale found by cross-correlation and by the least
RMS.

From noao/onedspec/t_tweak.x (tweak, twk_gcal, twk_tweak, twk_prep, twk_xcorr,
twk_rmsmin, twk_ascale, twk_ashift, twk_spec), in real as f2c compiles it. The
calibration is interpolated by onedspec's interp (core/asi.py), the continuum of the
correlation by icfit in real (core/icfitbits.py).

IRAF's ways kept: the shift and scale found for one spectrum are where the search for
the next starts, image after image; the cross-correlation is of the spectra with their
continuum (a chebyshev of three terms, rejecting 3 sigma low and 1 high) taken out and
their ends tapered, over the sample, and its peak is looked for in steps of 0.01 of a
pixel in real; the RMS is of each pixel less the mean of those three either side; a
spectrum's pixel is taken from the calibration's nearest end where it does not reach,
and a pixel half a pixel beyond the calibration's last is counted outside it -- the
first end never is, the calibration's world coordinates stopping half a pixel before
it; the telluric output is normalized by the mean of the correction.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import advice, icfitbits, iraffmt
from .asi import Interpolant
from .gtools import GTools, Scan, pagefile, strdic

f32 = np.float32
INDEFR = None
INDEF = np.float32(1.6e38)      #: a real INDEF a colon command was given


@dataclass
class Tweak:
    """The TWK structure: what carries from spectrum to spectrum."""
    telluric: bool
    shift: np.float32
    scale: np.float32
    dshift: np.float32
    dscale: np.float32
    thresh: np.float32
    sample: str
    rms: np.float32 | None = None
    # the current spectrum
    wave: np.ndarray | None = None      #: its world coordinates, double
    y: np.ndarray | None = None         #: its pixels
    cal: object = None                  #: the calibration's shdr
    asi: Interpolant | None = None      #: and its interpolant
    am: np.float32 = f32(1)             #: AM(sh)
    am_cal: np.float32 = f32(1)         #: AM(cal)
    ranges: list | None = None
    z: np.ndarray | None = None         #: the corrected spectrum
    ycal: np.ndarray | None = None      #: SY(cal): the calibration where the spectrum is
    offset: np.float32 = f32(1)         #: the graphs' offset
    box: int = 1                        #: and their boxcar
    sh: object = None                   #: the spectrum's shdr


def in_range(ranges, i: int) -> bool:
    """rg_inrange, for a 1-based pixel."""
    return any(a <= i - 1 <= b for a, b in ranges)


def prep(data: np.ndarray, nout: int) -> np.ndarray:
    """twk_prep: the continuum (a chebyshev of three terms, rejecting 3 sigma low and
    1 high five times, grow 1) taken out, the ends tapered over lag pixels, and lag
    zeros either side."""
    nin = data.size
    x = np.arange(1, nin + 1, dtype=f32)
    ic = icfitbits.Ic(function="chebyshev", order=3, low_reject=3.0, high_reject=1.0,
                      niterate=5, grow=1.0)
    cv = icfitbits.fit(ic, x, np.asarray(data, f32), np.ones(nin, f32))
    lag = (nout - nin) // 2
    out = np.zeros(nout, f32)
    pi = f32(3.1415926535897932385)

    def resid(i):
        return f32(data[i - 1] - cv(f32(i)))

    for i in range(1, lag):
        taper = (1 - math.cos(float(f32(f32(f32(i) * pi) / f32(lag))))) / 2
        out[i + lag - 1] = f32(taper * float(resid(i)))
    for i in range(lag, nin - lag + 2):
        out[i + lag - 1] = resid(i)
    for i in range(nin - lag + 2, nin + 1):
        taper = (1 - math.cos(float(f32(f32(f32(nin + 1 - i) * pi) / f32(lag))))) / 2
        out[i + lag - 1] = f32(taper * float(resid(i)))
    return out


def xcorr(spec1, spec2, npix: int, ranges, lag: int, center):
    """twk_xcorr: the shift where the correlation of the two peaks, from the centre
    given; the centre unchanged when too few pixels take part."""
    nprof = 2 * lag + 1
    ishift = _nint(float(center))
    prof = np.zeros(nprof, f32)
    n = 0
    for j in range(-lag, lag + 1):
        p = f32(0)
        for i in range(1 + lag, npix - lag + 1):
            if not in_range(ranges, i - lag):
                continue
            k = i - j - ishift
            if k < 1 or k > npix:
                continue
            p = f32(p + f32(spec1[i - 1] * spec2[k - 1]))
            n += 1
        prof[j + lag] = p
    if n < 10 * nprof:
        return center
    asi = Interpolant("spline3", prof)
    center = f32(1)
    pmin = f32(asi(1.0))
    pmax = pmin
    x = f32(1)
    while x <= f32(nprof):
        p = f32(asi(float(x)))
        if p < pmin:
            pmin = p
        if p > pmax:
            pmax = p
            center = x
        x = f32(x + f32(.01))
    # IRAF goes on to normalize the profile and measure its width, which it does not
    # use: the centre is what comes back.
    return f32(f32(f32(center - f32(lag)) - f32(1)) + f32(ishift))


def _pow(base: float, power: float) -> float:
    """pow_dd: C's pow, NaN for a negative base and a fractional power."""
    with np.errstate(all="ignore"):
        return float(np.power(np.float64(base), np.float64(power)))


def _nint(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def spec(twk: Tweak, shift, scale) -> None:
    """twk_spec: the spectrum corrected with this shift and scale, and the RMS of it
    over the sample; twk.shift and twk.scale become them."""
    shift, scale = f32(shift), f32(scale)
    y = twk.y
    n = y.size
    cal = twk.cal
    ncal = cal.n
    amratio = f32(twk.am / twk.am_cal)
    z = np.zeros(n, f32)
    kept = np.zeros(n, f32)
    twk.ycal = kept
    norm = f32(0)
    for i in range(n):
        xcal = f32(cal.wl(float(twk.wave[i])) + float(shift))
        xcal1 = max(f32(1), min(f32(ncal), xcal))
        ycal = f32(twk.asi(float(xcal1)))
        kept[i] = ycal
        if twk.telluric:
            ycal = max(twk.thresh, ycal)
            kept[i] = ycal
            if ycal <= 0:
                advice.fail("Calibration spectrum negative or zero (set threshold "
                            "parameter)", "give --threshold, the least value the "
                            "calibration is taken to have.")
            z[i] = f32(float(y[i]) / _pow(float(ycal), float(f32(amratio * scale))))
        else:
            z[i] = f32(y[i] - f32(ycal * scale))
        norm = f32(norm + z[i])
    sum1 = sum2 = f32(0)
    nstat = 0
    for i in range(3, n - 3):
        if not in_range(twk.ranges, i + 1):
            continue
        zval = f32(z[i] - f32(f32(z[i - 3] + z[i + 3]) / f32(2)))
        sum1 = f32(sum1 + zval)
        sum2 = f32(sum2 + f32(zval * zval))
        nstat += 1
    if twk.telluric:
        norm = f32(norm / f32(n))
        if norm > 0:
            z = (z / norm).astype(f32)
            sum1 = f32(sum1 / norm)
            sum2 = f32(f32(sum2 / norm) / norm)
    if nstat == 0:
        twk.rms = None
    else:
        inside = f32(f32(f32(nstat) * sum2) - f32(sum1 * sum1))
        with np.errstate(invalid="ignore"):
            twk.rms = f32(math.sqrt(float(inside)) / nstat) if inside >= 0 else f32("nan")
    twk.shift, twk.scale, twk.z = shift, scale, z
    if twk.box > 1:
        # the graphs' boxcar, the ends taking the end pixels again
        box = min(n, twk.box)
        i = int((1 - box) / 2)
        j = i + box
        total = f32(0)
        for k in range(i, j):
            total = f32(total + z[max(0, k)])
        temp = np.zeros(n, f32)
        for k in range(n):
            temp[k] = total
            total = f32(f32(total - z[max(0, i)]) + z[min(n - 1, j)])
            i += 1
            j += 1
        twk.z = (temp / f32(box)).astype(f32)


def _less(a, b) -> bool:
    """a < b for the RMS, which may be INDEF or NaN, as IRAF compares reals."""
    if a is None or b is None:
        a = f32(1.6e38) if a is None else a
        b = f32(1.6e38) if b is None else b
    return bool(a < b)


def ascale(twk: Tweak) -> None:
    """twk_ascale: the scale of least RMS, by halving steps from dscale."""
    dscale = twk.dscale
    if dscale == 0:
        return
    oscale, shift = twk.scale, twk.shift
    scale = [f32(0)] * 3
    rms = [None] * 3
    lastscale = None
    for i in range(3):
        scale[i] = f32(f32(f32(1) - f32(f32(i - 1) * dscale)) * oscale)
        spec(twk, shift, scale[i])
        rms[i] = twk.rms
        lastscale = twk.scale
    while dscale > f32(.01):
        if f32(scale[0] / oscale) < f32(.5) or f32(scale[2] / oscale) > f32(2):
            twk.scale = oscale
            break
        if _less(rms[0], rms[1]):
            scale[2], scale[1] = scale[1], scale[0]
            scale[0] = f32(f32(f32(1) - dscale) * scale[1])
            rms[2], rms[1] = rms[1], rms[0]
            spec(twk, shift, scale[0])
            rms[0] = twk.rms
            lastscale = twk.scale
        elif _less(rms[2], rms[1]):
            scale[0], scale[1] = scale[1], scale[2]
            scale[2] = f32(f32(dscale + f32(1)) * scale[1])
            rms[0], rms[1] = rms[1], rms[2]
            spec(twk, shift, scale[2])
            rms[2] = twk.rms
            lastscale = twk.scale
        else:
            dscale = f32(dscale / f32(2))
            scale[0] = f32(f32(f32(1) - dscale) * scale[1])
            scale[2] = f32(f32(dscale + f32(1)) * scale[1])
            spec(twk, shift, scale[0])
            rms[0] = twk.rms
            spec(twk, shift, scale[2])
            rms[2] = twk.rms
            lastscale = twk.scale
        if _less(rms[0], rms[1]):
            twk.scale = scale[0]
        elif _less(rms[2], rms[1]):
            twk.scale = scale[2]
        else:
            twk.scale = scale[1]
    if twk.scale != lastscale:
        spec(twk, shift, twk.scale)


def ashift(twk: Tweak) -> None:
    """twk_ashift: the shift of least RMS, by halving steps from dshift."""
    dshift = twk.dshift
    if dshift == 0:
        return
    oshift, scale = twk.shift, twk.scale
    shift = [f32(0)] * 3
    rms = [None] * 3
    lastshift = None
    for i in range(3):
        shift[i] = f32(oshift + f32(dshift * f32(i - 1)))
        spec(twk, shift[i], scale)
        rms[i] = twk.rms
        lastshift = twk.shift
    while dshift > f32(.01):
        if f32(abs(f32(oshift - shift[1]))) > f32(2):
            twk.shift = oshift
            break
        if _less(rms[0], rms[1]):
            shift[2], shift[1] = shift[1], shift[0]
            shift[0] = f32(shift[1] - dshift)
            rms[2], rms[1] = rms[1], rms[0]
            spec(twk, shift[0], scale)
            rms[0] = twk.rms
            lastshift = twk.shift
        elif _less(rms[2], rms[1]):
            shift[0], shift[1] = shift[1], shift[2]
            shift[2] = f32(shift[1] + dshift)
            rms[0], rms[1] = rms[1], rms[2]
            spec(twk, shift[2], scale)
            rms[2] = twk.rms
            lastshift = twk.shift
        else:
            dshift = f32(dshift / f32(2))
            shift[0] = f32(shift[1] - dshift)
            spec(twk, shift[0], scale)
            rms[0] = twk.rms
            shift[2] = f32(shift[1] + dshift)
            spec(twk, shift[2], scale)
            rms[2] = twk.rms
            lastshift = twk.shift
        if _less(rms[0], rms[1]):
            twk.shift = shift[0]
        elif _less(rms[2], rms[1]):
            twk.shift = shift[2]
        else:
            twk.shift = shift[1]
    if twk.shift != lastshift:
        spec(twk, twk.shift, scale)


def rmsmin(twk: Tweak) -> None:
    """twk_rmsmin: the shift, then the scale, twice, unless nothing changes."""
    lastshift = lastscale = None
    for _ in range(2):
        if twk.shift == lastshift and twk.scale == lastscale:
            break
        lastshift = twk.shift
        ashift(twk)
        if twk.shift == lastshift and twk.scale == lastscale:
            break
        lastscale = twk.scale
        ascale(twk)


def tweak(twk: Tweak, xcorr_: bool, tweakrms: bool, lag: int, by_hand=None) -> None:
    """twk_tweak: the sample, the cross-correlation and the least RMS; then, by hand,
    `by_hand(twk)` -- the graphs and their keys (FitSession)."""
    twk.ranges = icfitbits._xranges(twk.sample, np.asarray(twk.wave), [])
    if xcorr_ and lag > 0:
        n = twk.y.size
        nlag = n + 2 * lag
        y = np.zeros(n, f32)
        for i in range(n):
            ical = f32(max(1.0, min(float(twk.cal.n), twk.cal.wl(float(twk.wave[i])))))
            y[i] = f32(twk.asi(float(ical)))
        x = prep(y, nlag)
        y = prep(twk.y, nlag)
        twk.shift = f32(xcorr(x, y, n, twk.ranges, lag, twk.shift))
    if tweakrms:
        rmsmin(twk)
    if by_hand is not None:
        by_hand(twk)


def correct(twk: Tweak):
    """The loop of tweak over one spectrum: (the corrected spectrum, the mean of the
    correction for a telluric one, how many pixels fall outside the calibration)."""
    shift = twk.shift
    scale = f32(f32(twk.scale * twk.am) / twk.am_cal) if twk.telluric else twk.scale
    n = twk.y.size
    ncal = twk.cal.n
    out = np.zeros(n, f32)
    nout = 0
    mean = f32(0)
    for k in range(n):
        ical = f32(twk.cal.wl(float(twk.wave[k])) + float(shift))
        if ical < 1 or ical > f32(ncal):
            if ical < f32(.5) or ical > f32(f32(ncal) + f32(.5)):
                nout += 1
            ical = max(f32(1), min(f32(ncal), ical))
        if twk.telluric:
            base = max(twk.thresh, f32(twk.asi(float(ical))))
            fcor = f32(_pow(float(base), float(scale)))
            with np.errstate(all="ignore"):
                out[k] = f32(twk.y[k] / fcor)
            mean = f32(mean + fcor)
        else:
            fcor = f32(f32(twk.asi(float(ical))) * scale)
            out[k] = f32(twk.y[k] - fcor)
    mean = f32(mean / f32(n))
    if twk.telluric:
        out = (out * mean).astype(f32)
    return out, mean, nout


# ------------------------------------------------------------ by hand: twk_fit

MAX_REAL = float(np.finfo(np.float32).max)
CMDS = ("help", "shift", "scale", "dshift", "dscale", "offset", "smooth", "sample")
BELL = "\a\n"
SAMPLE_PROMPT = "s to add sample region or n for new regions:\n"
#: The keys, for the panel's guide.
KEYS = [
    ("x", "the three candidates by shift; over one of them, choose it"),
    ("y", "the three candidates by scale; over one of them, choose it"),
    ("a", "the shift and scale of least RMS over the sample"),
    ("e", "double the step of the candidates shown"),
    ("s", "a sample region from here to the next key: s adds it, n starts anew"),
    ("c", "show or hide the calibration below"),
    ("d", "show or hide the spectrum below"),
    ("r", "redraw"),
    ("w", "window, then a key: a b c d e f g j k l m n p r t u x y z ?"),
    ("q", "quit: the middle candidate's shift and scale are taken"),
    (":", "shift scale dshift dscale offset smooth sample, and /gtools"),
]


@dataclass
class FitFrame:
    """What twk_graph draws: the windows as drawn (x1 x2 y1 y2 on the screen) and the
    viewports (x1 x2 y1 y2 in NDC) of the candidates' graph and of the one below, if
    there is one; the title block, the label under the lowest graph; the three
    candidates, lowest first, and the curve below (the calibration where the spectrum
    is, or the spectrum), against x; the sample's regions as (x1, x2, tick at x1, tick
    at x2) with the heights y1 y2 y3 of the ticks and of the bar."""
    windows: list[tuple]
    viewports: list[tuple]
    title: list[str]
    xlabel: str
    x: np.ndarray
    curves: list[np.ndarray]
    below: np.ndarray | None
    regions: list[tuple]
    heights: tuple
    transposed: tuple[bool, bool] = (False, False)
    logs: tuple = ((False, False), (False, False))


def _inside(gt: GTools, x: np.ndarray, y: np.ndarray, window: tuple) -> tuple:
    """gt_ascale: the window set to the range of the points within gtools' limits --
    under transpose the limits of each axis tried on the other's values, as the code
    has it; with none inside, the window as it was."""
    limits = (gt.xmin, gt.xmax, gt.ymin, gt.ymax)
    if gt.transpose:
        limits = limits[2:] + limits[:2]
    xmin, xmax, ymin, ymax = ((MAX_REAL if at % 2 else -MAX_REAL) if v is None else v
                              for at, v in enumerate(limits))
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)
    with np.errstate(invalid="ignore"):
        inside = ~((x < xmin) | (x > xmax) | (y < ymin) | (y > ymax))
    if not inside.any():
        return window
    xs, ys = x[inside], y[inside]
    x1, x2, y1, y2 = window
    with np.errstate(invalid="ignore"):
        a, b = float(np.fmin.reduce(xs)), float(np.fmax.reduce(xs))
        c, d = float(np.fmin.reduce(ys)), float(np.fmax.reduce(ys))
    y_new = _clipped(gt, np.sort(ys))
    if y_new is not None:
        c, d = y_new
    if a <= b:
        x1, x2 = a, b
    if c <= d:
        y1, y2 = c, d
    return x1, x2, y1, y2


def _clipped(gt: GTools, ys: np.ndarray):
    """gt_ascale's :/clip: the values at the clips only when the low one is past the
    high one -- which gswind then refuses, being the wrong way round."""
    if not (gt.lclip > 0 or gt.hclip > 0):
        return None
    n = ys.size
    j = int(gt.lclip / 100 * n) if gt.lclip > 1 else int(max(0.0, gt.lclip * n))
    k = int(gt.hclip / 100 * n) if gt.hclip > 1 else int(max(0.0, gt.hclip * n))
    k = n - 1 - k
    if j <= k:
        return None
    return float(ys[min(j, n - 1)]), float(ys[max(k, 0)])


class FitSession:
    """twk_fit on one spectrum: the candidates drawn at once, then the keys, one read
    at a time (`key`). What IRAF prints goes to `out`, its error stream to `err`; each
    graph drawn is a FitFrame in `frames`. `done` once q is read: the shift and scale
    left in the Tweak are the middle candidate's."""

    def __init__(self, twk: Tweak, label: str = "", units: str = "") -> None:
        self.twk = twk
        sh, cal = twk.sh, twk.cal
        kind = "TELLURIC" if twk.telluric else "SKYTWEAK"
        self.help = "telluric.key" if twk.telluric else "skytweak.key"
        self.gt = [GTools(), GTools()]
        self.gt[0].title = (f"{kind}: spectrum = {sh.image}{sh.section}, calibration = "
                            f"{cal.image}{cal.section}")
        for gt in self.gt:
            gt.xlabel, gt.xunits = (label, units) if label else ("Pixels", "")
            gt.set_type("line")
        self.x = np.asarray(twk.wave, f32)
        self.graph1 = "y" if twk.dscale > 0 else "x"
        self.graph2 = "cal"
        self.shifts = [f32(0)] * 3
        self.scales = [f32(0)] * 3
        self.z: list[np.ndarray] = []
        self.out = ""
        self.err = ""
        self.frames: list[FitFrame] = []
        self.windows = [(0.0, 1.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)]
        self.current = 0                # the WCS left set: the last graph drawn
        self.state = ""
        self.first = (0.0, 0.0)
        self.wcs = 1
        self.done = False
        self._step(newdata=True, newgraph=True)

    # ------------------------------------------------------------ the data

    def _candidates(self) -> None:
        twk = self.twk
        one = f32(1)
        if self.graph1 == "x":
            self.shifts = [f32(twk.shift - twk.dshift), twk.shift,
                           f32(twk.shift + twk.dshift)]
            self.scales = [twk.scale] * 3
        else:
            self.shifts = [twk.shift] * 3
            self.scales = [f32(twk.scale * f32(one - twk.dscale)), twk.scale,
                           f32(twk.scale * f32(one + twk.dscale))]
        spec(twk, self.shifts[0], self.scales[0])
        low = (twk.z - twk.offset).astype(f32)
        spec(twk, self.shifts[2], self.scales[2])
        high = (twk.z + twk.offset).astype(f32)
        spec(twk, self.shifts[1], self.scales[1])
        self.z = [low, twk.z, high]
        p = iraffmt.printf
        text = f"scale = {p(float(twk.scale), 'g', width=5):>5}"
        if self.graph1 == "y":
            text += f" +/- {p(float(twk.dscale), 'g', width=6):>6}"
        text += f", shift = {p(float(twk.shift), 'f', 2)}"
        if self.graph1 == "x":
            text += f" +/- {p(float(twk.dshift), 'f', 2)}"
        text += f", offset = {p(float(twk.offset), 'g', width=3):>3}"
        text += ", rms = " + ("INDEF" if twk.rms is None
                              else p(float(twk.rms), "g", 3))
        self.gt[0].comments = text

    # ------------------------------------------------------------ the graph

    def graph(self) -> FitFrame:
        """twk_graph."""
        twk, gt1, gt2 = self.twk, *self.gt
        two = self.graph2 != "none"
        if two:
            gt1.drawxlabels = False
        window = (0.0, 1.0, 0.0, 1.0)
        spans = []
        for y in self.z:
            window = _inside(gt1, self.x, y, window)
            spans.append(window)
        xmin = float(f32(min(s[0] for s in spans)))
        xmax = float(f32(max(s[1] for s in spans)))
        ymin = float(f32(min(s[2] for s in spans)))
        ymax = float(f32(max(s[3] for s in spans)))
        self.windows[0] = gt1.swind((xmin, xmax), (ymin, ymax))
        self.current = 0
        title = gt1.title_block()
        label = gt1.label("x")
        x1, x2, y1, y2 = (f32(v) for v in self.windows[0])
        dy = f32(y2 - y1)
        h1 = f32(y1 + f32(dy / f32(100)))
        h2 = f32(h1 + f32(dy / f32(20)))
        heights = (float(h1), float(h2), float(f32(f32(h1 + h2) / f32(2))))
        regions = []
        if "*" not in twk.sample:
            for a, b in icfitbits._xranges(twk.sample, self.x, [], merge=False):
                xa, xb = self.x[a], self.x[b]
                regions.append((float(xa), float(xb), bool(xa > x1 and xa < x2),
                                bool(xb > x1 and xb < x2)))
        below = None
        viewports = [(0.1, 0.9, 0.4, 0.9)] if two else [self._viewport(len(title))]
        if two:
            below = twk.ycal if self.graph2 == "cal" else np.asarray(twk.y, f32)
            gt2.drawxlabels = True
            gt2.drawtitle = False
            inner = _inside(gt2, self.x, below, (0.0, 1.0, 0.0, 1.0))
            self.windows[1] = gt2.swind((xmin, xmax), (inner[2], inner[3]))
            self.current = 1
            label = gt2.label("x")
            viewports.append((0.1, 0.9, 0.1, 0.4))
        frame = FitFrame(list(self.windows[:2 if two else 1]), viewports, title, label,
                         self.x, list(self.z), below, regions, heights,
                         (gt1.transpose, gt2.transpose), (gt1.logs(), gt2.logs()))
        self.frames.append(frame)
        return frame

    def _viewport(self, lines: int) -> tuple:
        """glb_set_viewport for the candidates alone, on stdvdm: the title block and
        the system's line above; below, room for the x axis' numbers and label --
        which, once the graph below has been drawn, the candidates' graph no longer
        has (twk_graph turns them off and never on again)."""
        gt = self.gt[0]
        top = 1.0 - (int(gt.sysid and gt.drawtitle) + lines + 1) * 0.0294
        bottom = 4 * 0.0294 if gt.drawxlabels else 0.0
        return 9 * 0.0125, 1.0 - 9 * 0.0125, bottom, top

    # ---------------------------------------------------------- the colon

    def colon(self, text: str) -> tuple[bool, bool]:
        """twk_colon: (new data, new graph)."""
        twk = self.twk
        gt = self.gt[self.wcs - 1]
        if text.startswith("/"):
            out, err, newgraph = gt.colon(text, self.windows[self.current])
            self.out += out
            self.err += err
            self._share(self.wcs - 1)
            return False, newgraph
        scan = Scan(text, gt.last_real)
        found = strdic(scan.wrd(), CMDS)
        name = CMDS[found - 1] if found else ""
        g = iraffmt.g
        if name == "help":
            self.out += pagefile(self.help)
        elif name in ("shift", "scale", "dshift", "dscale", "offset"):
            value = scan.r()
            if scan.n == 1:
                self.out += f"{name} {g(float(getattr(twk, name)))}\n"
            elif name == "dscale" and (value is None or value < 0 or value >= 1):
                self.out += "dscale must be between zero and one\a\n"
            elif name == "offset":
                value = INDEF if value is None else f32(value)
                if value != twk.offset:
                    twk.offset = value
                    self.gt[0].setr("ymin", None)
                    self.gt[0].setr("ymax", None)
                    return True, False
            else:
                setattr(twk, name, INDEF if value is None else f32(value))
                return True, False
        elif name == "smooth":
            value = scan.i()
            if scan.n == 1:
                self.out += f"smooth {twk.box}\n"
            else:
                value = max(1, value)
                if value != twk.box:
                    twk.box = value
                    return True, False
        elif name == "sample":
            rest = scan.line[scan.pos:].split("\n")[0]
            if not rest:
                self.out += f"sample {twk.sample}\n"
            else:
                said: list[str] = []
                ranges = icfitbits._xranges(rest[1:], np.asarray(twk.wave), said)
                if said:
                    # IRAF warns and takes it all the same, and has then no sample
                    self.err += (f"Warning: {said[0]}: the sample is kept; write ranges "
                                 "as 6850:6900,7000:7040\n")
                else:
                    twk.sample, twk.ranges = rest[1:], ranges
                    return True, False
        else:
            self.out += BELL
        return False, False

    def _share(self, source: int) -> None:
        """The x limits and flip of the graph a w or a :/ was over, given to the other
        through gt_setr and gt_getr."""
        a, b = self.gt[source], self.gt[1 - source]
        low, high = a.getr("xmin"), a.getr("xmax")
        b.setr("xmin", low)
        b.setr("xmax", high)
        b.xflip = a.xflip

    # ------------------------------------------------------------ the keys

    def _sample(self, text: str) -> None:
        twk = self.twk
        twk.sample = text
        twk.ranges = icfitbits._xranges(text, np.asarray(twk.wave), [])

    def key(self, key: str, wx: float = 0.0, wy: float = 0.0, wcs: int = 1,
            rest: str = "") -> None:
        """One read of twk_fit's cursor loop; `wcs` the graph the cursor is over (1 the
        candidates, 2 the one below), `rest` a colon command's text."""
        twk = self.twk
        state, self.state = self.state, ""
        newdata = newgraph = False
        if state == "s":
            a, b = iraffmt.g(float(f32(self.first[0]))), iraffmt.g(float(f32(wx)))
            if key == "n":
                self._sample(f"{a}:{b}")
                newdata = True
            elif key == "s":
                self._sample(f"{a}:{b}" if twk.sample.startswith("*")
                             else f"{twk.sample},{a}:{b}")
                newdata = True
        elif state == "window":
            if key == "e":
                self.first, self.state = (wx, wy), "window2"
                self.out += "again:"
                return
            said, newgraph = self.gt[self.wcs - 1].window1(key, wx, wy,
                                                           self.windows[self.current])
            self.out += said
            self._share(self.wcs - 1)
        elif state == "window2":
            newgraph = self.gt[self.wcs - 1].window2(self.first, (wx, wy),
                                                     self.windows[self.current])
            self._share(self.wcs - 1)
        elif key == ":":
            self.wcs = wcs if wcs in (1, 2) else 1
            newdata, newgraph = self.colon(rest)
        elif key == "?":
            self.out += pagefile(self.help)
        elif key == "a":
            rmsmin(twk)
            newdata = True
        elif key in ("c", "d"):
            kind = "cal" if key == "c" else "data"
            self.graph2 = "none" if self.graph2 == kind else kind
            self.gt[1].setr("ymin", None)
            self.gt[1].setr("ymax", None)
            newgraph = True
        elif key == "e":
            two = f32(2)
            if self.graph1 == "x":
                twk.dshift = f32(.1) if twk.dshift == 0 else f32(two * twk.dshift)
            else:
                twk.dscale = f32(.1) if twk.dscale == 0 else \
                    min(f32(.99), f32(two * twk.dscale))
            newdata = True
        elif key == "q":
            self.done = True
            return
        elif key == "r":
            newgraph = True
        elif key == "s":
            self.first, self.state = (wx, wy), "s"
            self.out += SAMPLE_PROMPT
            return
        elif key == "w":
            self.wcs = wcs if wcs in (1, 2) else 1
            self.state = "window"
            self.out += "window:"
            return
        elif key in ("x", "y"):
            n = twk.y.size
            pix = max(1, min(n, _nint(twk.sh.wl(float(f32(wx)))))) - 1
            wy = f32(wy)
            j = 0
            dy = abs(f32(wy - self.z[0][pix]))
            for i in (1, 2):
                if abs(f32(wy - self.z[i][pix])) < dy:
                    j = i
                    dy = abs(f32(wy - self.z[i][pix]))
            twk.shift, twk.scale = self.shifts[j], self.scales[j]
            if j == 1 and self.graph1 == key:
                if key == "x":
                    twk.dshift = f32(twk.dshift / f32(2))
                else:
                    twk.dscale = f32(twk.dscale / f32(2))
            # the code asks twice about the shift's step, never about the scale's
            self.graph1 = "y" if twk.dshift == 0 else key
            newdata = True
        else:
            self.out += BELL
        self._step(newdata, newgraph)

    def _step(self, newdata: bool, newgraph: bool) -> None:
        if newdata:
            self._candidates()
            newgraph = True
        if newgraph:
            self.graph()


def run_reads(session: FitSession, cursor) -> None:
    """twk_fit's loop over reads (x, y, wcs, key, rest), until q or the end of them.
    The next spectrum's session reads on from where this one stopped."""
    for x, y, wcs, key, rest in cursor:
        session.key(key, x, y, wcs, rest)
        if session.done:
            return
