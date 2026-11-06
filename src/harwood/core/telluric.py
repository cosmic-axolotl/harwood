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

from . import advice, icfitbits
from .asi import Interpolant

f32 = np.float32
INDEFR = None


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
    norm = f32(0)
    for i in range(n):
        xcal = f32(cal.wl(float(twk.wave[i])) + float(shift))
        xcal1 = max(f32(1), min(f32(ncal), xcal))
        ycal = f32(twk.asi(float(xcal1)))
        if twk.telluric:
            ycal = max(twk.thresh, ycal)
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


def tweak(twk: Tweak, xcorr_: bool, tweakrms: bool, lag: int) -> None:
    """twk_tweak without the interactive step: the sample, the cross-correlation and
    the least RMS."""
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
