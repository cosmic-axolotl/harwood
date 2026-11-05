"""sarith and scopy: spectra copied, cut to a wavelength region, put on an even grid,
operated on alone (log, sqrt, mag, flam...) or with a constant or another spectrum
resampled onto theirs.

From noao/onedspec/t_sarith.x (sa_ms, sa_1d, sa_sextract, sa_arith, sa_verbose) and
smw/shdr.x (shdr_lw, shdr_wl, shdr_extract, shdr_linear, shdr_rebin). An aperture is
held as IRAF's shdr holds it: its wavelengths, its bands (flux, raw, sky and error,
IRAF's spectrum, raw, background and sigma), and while its world coordinates are its
own, a wavelength's pixel by their inverse, half a pixel past the ends at most; once
cut or resampled, by its first and last wavelength and its step, in real.

IRAF's ways kept: a region is the nearest whole pixels to its ends, or with rebin
the ends themselves and as many pixels between them; the resampling integrates the
interpolant (onedspec's interp) over each new pixel and takes the mean, and over
exactly whole pixels sums them -- dividing by one fewer than it summed, as IRAF's
does; the sigma band is only ever copied; an operand that has no such aperture
makes the operation a copy; what cannot be computed (a division by zero, the log of
a negative) is errval.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import advice, iraffmt, spectrum
from .asi import Interpolant

f32 = np.float32
UNARY = ("abs", "copy", "dex", "exp", "flam", "fnu", "inv", "ln", "log", "lum", "mag",
         "sqrt")
BINARY = ("replace", "+", "-", "*", "/", "^")
OPS = UNARY + BINARY
#: harwood's columns as IRAF's bands, in apall's order.
BANDS = ("flux", "raw", "sky", "error")


def _nint(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _equal(a: float, b: float) -> bool:
    """fp_equald."""
    return a == b or abs(a - b) <= 2.220e-16 * 32 * max(abs(a), abs(b))


@dataclass
class Sh:
    """shdr on one band of one aperture."""
    image: str                      #: as it is said: IMNAME
    section: str                    #: IMSEC, [*,line] or [*,line,band]
    ap: int
    beam: int
    title: str
    band: str                       #: harwood's column
    data: np.ndarray                #: SY, in real
    wave: np.ndarray                #: the world coordinates, in double
    z: float = 0.0
    dc: int = 0
    wcs: bool = True                #: CTLW and CTWL: the world coordinates are its own
    w0: float = 0.0                 #: W0, W1, WP: in real
    w1: float = 0.0
    wp: float = 0.0
    lin: tuple = field(default=(0.0, 0.0))   #: a linear or log WCS's start and step

    @property
    def n(self) -> int:
        return self.data.size

    def lw(self, x: float) -> float:
        """shdr_lw: the world coordinate of a logical pixel."""
        if self.wcs:
            if self.dc in (0, 1):
                w = self.lin[0] + self.lin[1] * (x - 1)
                w = 10 ** w if self.dc == 1 else w
                return w / (1 + self.z)
            pixel = np.arange(1.0, self.n + 1.0)
            return float(np.interp(x, pixel, self.wave))
        if self.dc == 1:
            return float(self.w0) * 10.0 ** (math.log10(float(self.w1) / float(self.w0))
                                             * (x - 1) / (self.n - 1))
        return float(self.w0) + (x - 1) * float(self.wp)

    def wl(self, w: float) -> float:
        """shdr_wl: a world coordinate's logical pixel."""
        if self.wcs:
            if self.dc in (0, 1):
                din = w * (1 + self.z)
                if self.dc == 1:
                    din = math.log10(din)
                x = (din - self.lin[0]) / self.lin[1] + 1
                return max(0.5, min(self.n + 0.5, x))
            pixel = np.arange(1.0, self.n + 1.0)
            wave, px = (self.wave, pixel) if self.wave[-1] >= self.wave[0] else \
                (self.wave[::-1], pixel[::-1])
            return max(0.5, min(self.n + 0.5, float(np.interp(w, wave, px))))
        if self.dc == 1:
            span = math.log10(float(self.w1) / float(self.w0))
            return math.log10(w / float(self.w0)) / span * (self.n - 1) + 1
        return (w - float(self.w0)) / float(self.wp) + 1


def open_aperture(aperture, image: str, section: str, band: str) -> Sh:
    """shdr_open on a band of a harwood aperture."""
    wave = np.asarray(aperture.wavelength, dtype=float) if aperture.wavelength is not None \
        else np.arange(1.0, len(aperture.flux) + 1.0)
    dc = spectrum.dispersion_type(aperture.wavelength)
    z = float(aperture.redshift or 0.0)
    sh = Sh(image, section, aperture.number,
            aperture.number if aperture.beam is None else aperture.beam,
            aperture.title, band, np.asarray(getattr(aperture, band), dtype=f32), wave, z,
            max(dc, 0))
    d = wave * (1 + z)
    n = wave.size
    if sh.dc == 0:
        sh.lin = (float(f"{d[0]:.15g}"), float(f"{(d[-1] - d[0]) / (n - 1):.15g}"))
    elif sh.dc == 1:
        a = iraffmt.ctod(f"{math.log10(d[0]):.15g}")
        b = iraffmt.ctod(f"{(math.log10(d[-1]) - math.log10(d[0])) / (n - 1):.15g}")
        sh.lin = (a, b)
    sh.w0, sh.w1 = f32(sh.lw(1.0)), f32(sh.lw(float(n)))
    sh.wp = f32((sh.w1 - sh.w0) / f32(n - 1)) if n > 1 else f32(0)
    return sh


def _interpolant(data: np.ndarray, interp: str) -> Interpolant:
    """asifit on the data extended by a pixel at each end."""
    extended = np.concatenate(([data[0]], data, [data[-1]])).astype(f32)
    return Interpolant(interp, extended)


def _mean(asi: Interpolant, a, b) -> np.float32:
    """The mean of the interpolant from a to b, shdr_linear's and shdr_rebin's."""
    if a <= b:
        ia, ib = _nint(float(a + f32(0.5))), _nint(float(b - f32(0.5)))
        if abs(float(a) + 0.5 - ia) < 0.00001 and abs(float(b) - 0.5 - ib) < 0.00001:
            total = f32(0)
            for j in range(ia, ib + 1):
                total = f32(total + f32(asi(float(j))))
            if ib - ia > 0:
                total = f32(total / f32(ib - ia))
            return total
        total = f32(asi.integral(float(a), float(b)))
        return f32(total / f32(b - a)) if b - a > 0 else total
    ib, ia = _nint(float(b + f32(0.5))), _nint(float(a - f32(0.5)))
    if abs(float(a) - 0.5 - ia) < 0.00001 and abs(float(b) + 0.5 - ib) < 0.00001:
        total = f32(0)
        for j in range(ib, ia + 1):
            total = f32(total + f32(asi(float(j))))
        if ia - ib > 0:
            total = f32(total / f32(ia - ib))
        return total
    total = f32(asi.integral(float(b), float(a)))
    return f32(total / f32(a - b)) if a - b > 0 else total


def linear(sh: Sh, w0, w1, n: int, dc: int, interp: str) -> None:
    """shdr_linear: the band on n pixels evenly from w0 to w1 (in log10 for dc 1)."""
    w0, w1 = f32(w0), f32(w1)
    if sh.dc == dc and sh.w0 == w0 and sh.w1 == w1 and sh.n == n:
        return
    asi = _interpolant(sh.data, interp)
    xmin, xmax = 0.5, sh.n + 0.5
    if dc == 1:
        w0l = math.log10(float(w0))
        wp = (math.log10(float(w1)) - math.log10(float(w0))) / (n - 1)

        def world(x):
            return 10.0 ** (w0l + (x - 1) * wp)
    else:
        wp = (float(w1) - float(w0)) / (n - 1)

        def world(x):
            return float(w0) + (x - 1) * wp
    b = f32(max(xmin, min(xmax, sh.wl(world(0.5)))) + 1)
    out = np.empty(n, dtype=f32)
    for i in range(1, n + 1):
        a = b
        b = f32(max(xmin, min(xmax, sh.wl(world(i + 0.5)))) + 1)
        out[i - 1] = _mean(asi, a, b)
    sh.data = out
    sh.wave = np.array([world(i + 1.0) for i in range(n)], dtype=f32).astype(float)
    sh.w0, sh.w1 = w0, w1
    sh.wp = f32((w1 - w0) / f32(n - 1))
    sh.dc, sh.wcs = dc, False


def extract(sh: Sh, w1, w2, rebin: bool, interp: str) -> None:
    """sa_sextract and shdr_extract: the region from w1 to w2 (None, the end)."""
    if w1 is None and w2 is None:
        return
    a = sh.lw(1.0) if w1 is None else w1
    b = sh.lw(float(sh.n)) if w2 is None else w2
    a, b = float(f32(a)), float(f32(b))       # shdr_extract takes them in real
    l1, l2 = sh.wl(a), sh.wl(b)
    if _equal(l1, l2) or max(l1, l2) < 1 or min(l1, l2) > sh.n:
        advice.fail("No pixels to extract",
                    "give w1 and w2 that are inside the spectrum.")
    l1, l2 = max(1.0, min(float(sh.n), l1)), max(1.0, min(float(sh.n), l2))
    i1, i2 = _nint(l1), _nint(l2)
    n = abs(i2 - i1) + 1
    if rebin:
        start, end = sh.lw(l1), sh.lw(l2)
        linear(sh, start, end, n, 0 if sh.dc == 2 else sh.dc, interp)
        return
    if i1 == 1 and i2 == sh.n:
        return
    if i1 <= i2:
        sh.data, sh.wave = sh.data[i1 - 1:i2].copy(), sh.wave[i1 - 1:i2].copy()
    else:
        sh.data = sh.data[i2 - 1:i1][::-1].copy()
        sh.wave = sh.wave[i2 - 1:i1][::-1].copy()
    sh.w0, sh.w1 = f32(sh.wave[0]), f32(sh.wave[-1])
    if sh.n > 1:
        sh.wp = f32((sh.w1 - sh.w0) / f32(sh.n - 1))
    sh.wcs = False


def rebin(sh: Sh, ref: Sh, interp: str) -> None:
    """shdr_rebin: sh onto ref's pixels, the mean over each."""
    if sh.dc == ref.dc and sh.dc != 2 and sh.w0 == ref.w0 and sh.wp == ref.wp and \
            sh.n == ref.n:
        return
    asi = _interpolant(sh.data, interp)
    xmin, xmax = 0.5, sh.n + 0.5
    n = ref.n
    out = np.empty(n, dtype=f32)
    b = f32(max(xmin, min(xmax, sh.wl(ref.lw(0.5)))) + 1)
    for i in range(1, n + 1):
        a = b
        b = f32(max(xmin, min(xmax, sh.wl(ref.lw(i + 0.5)))) + 1)
        out[i - 1] = _mean(asi, a, b)
    sh.data, sh.wave = out, ref.wave.copy()
    sh.dc, sh.w0, sh.w1, sh.wp, sh.wcs = ref.dc, ref.w0, ref.w1, ref.wp, False


def arith(op: str, sh: Sh, other: Sh | None, const: float | None, reverse: bool,
          errval: float) -> np.ndarray:
    """sa_arith, in real, as IRAF's: what cannot be computed is errval."""
    x = np.asarray(sh.data, dtype=f32)
    err = f32(errval)
    if op in BINARY:
        operand = other.data if other is not None else \
            np.full(x.size, f32(const), dtype=f32)
    out = np.empty_like(x)
    for i, v in enumerate(x):
        v = f32(v)
        if op == "abs":
            r = abs(v)
        elif op == "copy":
            r = v
        elif op == "dex":
            r = f32(10.0 ** float(v))
        elif op == "exp":
            r = f32(math.exp(float(v)))
        elif op == "flam":
            s = f32(sh.wave[i])
            r = f32(v / f32(f32(s * s) / f32(2.997925e18)))
        elif op == "fnu":
            s = f32(sh.wave[i])
            r = f32(v * f32(f32(s * s) / f32(2.997925e18)))
        elif op == "inv":
            r = err if v == 0 else f32(f32(1) / v)
        elif op == "ln":
            r = err if v <= 0 else f32(math.log(float(v)))
        elif op == "log":
            r = err if v <= 0 else f32(math.log10(float(v)))
        elif op == "lum":
            r = f32(10.0 ** float(f32(f32(-0.4) * v)))
        elif op == "mag":
            r = err if v <= 0 else f32(f32(-2.5) * f32(math.log10(float(v))))
        elif op == "sqrt":
            r = err if v < 0 else f32(math.sqrt(float(v)))
        else:
            o = f32(operand[i])
            a, b = (o, v) if reverse else (v, o)
            if op == "replace":
                r = o
            elif op == "+":
                r = f32(v + o)
            elif op == "-":
                r = f32(a - b)
            elif op == "*":
                r = f32(v * o)
            elif op == "/":
                r = err if b == 0 else f32(a / b)
            else:
                r = f32(float(a) ** float(b)) if a > 0 or float(b).is_integer() else err
        out[i] = r
    return out


def said(sh: Sh, other: Sh | None, out_name: str, out_section: str, out_ap: int, op: str,
         const: float | None, reverse: bool) -> str:
    """sa_verbose."""
    first = f"{sh.image}{sh.section}({sh.ap})"
    target = f"{out_name}{out_section}({out_ap})"
    if op in UNARY:
        if op == "copy":
            return f"{first}  -->  {target}"
        # clgwrd gives the word as OPS spells it, and OPS breaks its line after sqrt.
        word = "sqrt\t\t " if op == "sqrt" else op
        return f"{first} -- {word}  -->  {target}"
    if const is None:
        second = f"{other.image}{other.section}({other.ap})"
        left, right = (second, first) if reverse else (first, second)
        return f"{left}  {op}  {right}  -->  {target}"
    number = iraffmt.g(float(f32(const)))
    if reverse:
        return f"{number}  {op}  {first}  -->  {target}"
    return f"{first}  {op}  {number}  -->  {target}"
