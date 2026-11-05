"""sbands: fluxes in bandpasses, and with one or two bands of continuum beside them
an index and an equivalent width.

From noao/onedspec/t_sbands.x. A band is an id, a centre, a width and a filter (a
file of wavelength and response, or none for a flat one); a line of the band file
holds one band, or one and two continua. The flux is the sum of the pixels the band
covers, the end pixels by the fraction of them inside, each weighed by the filter's
response at the pixel's wavelength; normalized, it is divided by the sum of those
weights. The continuum is the one band beside, or the two interpolated to the
centre; the index is flux over continuum, the equivalent width (1 - index) times the
band's width.

The pixels and their wavelengths are in real, the sums in double, as IRAF's are. A
wavelength's pixel is the world coordinate's inverse, which stops half a pixel past
either end: a band that starts before the spectrum is measured from its edge. The
band file is read word after word, as IRAF reads it, so a continuum written without
its id shifts the rest of the line along.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import iraffmt

f32 = np.float32


@dataclass
class Band:
    id: str
    filter: str
    centre: float
    width: float
    fw: list[float] = field(default_factory=list)       #: the filter's wavelengths
    fr: list[float] = field(default_factory=list)       #: and responses


_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?")


class _Line:
    """A line read as sb_bands reads it, ctowrd and ctod in turn from where the last
    left off: a number ctod does not find costs only the blanks before it."""

    def __init__(self, text: str):
        self.text, self.ip = text, 0

    def _blanks(self):
        while self.ip < len(self.text) and self.text[self.ip] in " \t":
            self.ip += 1

    def word(self) -> str | None:
        self._blanks()
        start = self.ip
        while self.ip < len(self.text) and self.text[self.ip] not in " \t\n":
            self.ip += 1
        return self.text[start:self.ip] or None

    def number(self) -> float | None:
        self._blanks()
        found = _NUMBER.match(self.text, self.ip)
        if found is None:
            return None
        self.ip = found.end()
        return float(found.group(0).replace("d", "e").replace("D", "e"))

    def band(self):
        """id, centre, width and filter, or None at the first that is not there."""
        band_id = self.word()
        if band_id is None:
            return None
        centre = self.number()
        if centre is None:
            return None
        width = self.number()
        if width is None:
            return None
        name = self.word()
        if name is None:
            return None
        return band_id, centre, width, name


def _filter(path: str, where: Path) -> tuple[list[float], list[float]]:
    fw, fr = [], []
    for line in (where / path).read_text(encoding="utf-8").splitlines():
        words = line.split()
        if len(words) < 2:
            continue
        w, r = iraffmt.ctod(words[0]), iraffmt.ctod(words[1])
        if w is None or r is None:
            continue
        fw.append(w)
        fr.append(r)
    return fw, fr


def read_bands(text: str, where: Path = Path(".")) -> list[list[Band | None]]:
    """sb_bands: each line's band and its continua, None where there is none; a line
    that does not start with a whole band, or whose id starts with #, is skipped."""
    bands = []
    for line in text.splitlines():
        reader = _Line(line)
        first = reader.band()
        if first is None or first[0].startswith("#"):
            continue
        row = [first, reader.band(), reader.band()]
        out = []
        for found in row:
            if found is None:
                out.append(None)
                continue
            band = Band(found[0], found[3], found[1], found[2])
            if band.filter != "none":
                band.fw, band.fr = _filter(band.filter, where)
            out.append(band)
        bands.append(out)
    return bands


def response(w: float, band: Band) -> float:
    """sb_filter: the filter's response at w, a straight line between its points and
    its end values beyond them."""
    n = len(band.fw)
    if n == 0:
        return 1.0
    x, y = band.fw, band.fr
    if w <= x[0]:
        return y[0]
    if w >= x[-1]:
        return y[-1]
    if (w - x[0]) < (x[-1] - w):
        i = 1
        while w > x[i]:
            i += 1
    else:
        i = n - 1
        while w < x[i - 1]:
            i -= 1
    return (w - x[i - 1]) / (x[i] - x[i - 1]) * (y[i] - y[i - 1]) + y[i - 1]


def flux(wave, data, band: Band | None, to_pixel) -> tuple[float | None, float]:
    """sb_flux: the band's flux and the sum of its weights; None outside the spectrum.
    `to_pixel` is shdr_wl, a wavelength to a pixel (1-based, fractional)."""
    if band is None:
        return None, 1.0
    a, b = band.centre - band.width / 2.0, band.centre + band.width / 2.0
    w1, w2 = min(a, b), max(a, b)
    a, b = to_pixel(w1), to_pixel(w2)
    x1, x2 = min(a, b), max(a, b)
    i1, i2 = _nint(x1), _nint(x2)
    if x1 == x2 or i1 < 1 or i2 > len(data):
        return None, 1.0
    sx, sy = np.asarray(wave, dtype=f32), np.asarray(data, dtype=f32)
    if i1 == i2:
        wt = response(float(sx[i1 - 1]), band) * (x2 - x1)
        return wt * float(sy[i1 - 1]), wt
    wt = response(float(sx[i1 - 1]), band) * (i1 + 0.5 - x1)
    total, norm = wt * float(sy[i1 - 1]), wt
    for i in range(i1 + 1, i2):
        wt = response(float(sx[i - 1]), band)
        total += wt * float(sy[i - 1])
        norm += wt
    wt = response(float(sx[i2 - 1]), band) * (x2 - i2 + 0.5)
    return total + wt * float(sy[i2 - 1]), norm + wt


def _nint(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _g(value, fmt: str, width: int) -> str:
    """%W.6g of a double, INDEF for none; fmt is IRAF's, kept for reading."""
    if value is None:
        return "INDEF".rjust(width)
    return iraffmt.printf(float(value), "g", 6, width=width, double=True).rjust(width)


def measure(name: str, wave, data, bands, to_pixel, norm: bool, mag: bool,
            magzero: float) -> list[str]:
    """sb_proc: a line for each band the spectrum covers."""
    lines = []
    for band1, band2, band3 in bands:
        flux1, norm1 = flux(wave, data, band1, to_pixel)
        if flux1 is None:
            continue
        if norm:
            flux1, norm1 = flux1 / norm1, 1.0
        shown = magzero - 2.5 * math.log10(flux1) if mag and flux1 > 0 else flux1
        line = f"{name:>26} {band1.id[:7]:>7} {_g(shown, 'g', 11)}"
        flux2, norm2 = flux(wave, data, band2, to_pixel)
        flux3, norm3 = flux(wave, data, band3, to_pixel)
        if flux2 is None and flux3 is None:
            lines.append(line)
            continue
        if norm:
            if flux2 is not None:
                flux2, norm2 = flux2 / norm2, 1.0
            if flux3 is not None:
                flux3, norm3 = flux3 / norm3, 1.0
        contval = index = eqwidth = None
        if flux2 is not None and flux3 is not None:
            a = (flux2 / norm2 - flux3 / norm3) / (band2.centre - band3.centre)
            b = flux2 / norm2 - a * band2.centre
            contval = (a * band1.centre + b) * norm1
            line += f" {'cont':>7}"
        elif flux2 is not None:
            contval = flux2
            line += f" {band2.id[:7]:>7}"
        else:
            contval = flux3
            line += f" {band3.id[:7]:>7}"
        shown = magzero - 2.5 * math.log10(contval) if mag and contval > 0 else contval
        line += f" {_g(shown, 'g', 11)}"
        if flux1 > 0 and contval > 0:
            index = flux1 / contval
            eqwidth = (1 - index) * band1.width
        if mag and index is not None:
            index = -2.5 * math.log10(index)
        lines.append(line + f" {_g(index, 'g', 9)} {_g(eqwidth, 'g', 9)}")
    return lines


def header(sysid: str, bandfile: str, bands, norm: bool, mag: bool,
           magzero: float) -> list[str]:
    """sb_header: the banner, the parameters, the bands and the column headings."""
    yes = lambda b: "yes" if b else "no"  # noqa: E731
    params = "#  "
    if bandfile:
        params += f" bands = {bandfile},"
    params += f" norm = {yes(norm)}, mag = {yes(mag)}"
    if mag:
        params += f", magzero = {magzero:.2f}"
    lines = ["", f"# SBANDS: {sysid}", params,
             f"# {'band':>10} {'filter':>10} {'wavelength':>10} {'width':>10}"]
    for row in bands:
        for band in row:
            if band is not None:
                ten = lambda v: iraffmt.printf(  # noqa: E731
                    float(v), "g", width=10, double=True).rjust(10)
                lines.append(f"# {band.id:>10} {band.filter:>10} {ten(band.centre)} "
                             f"{ten(band.width)}")
    what = "mag" if mag else "flux"
    heading = f"#\n# {'spectrum':>24} {'band':>7.7} {what:>11.11}"
    if sum(b is not None for row in bands for b in row) > len(bands):
        heading += f" {'band':>7.7} {what:>11.11} {'index':>9.9} {'eqwidth':>9.9}"
    return lines + heading.split("\n")


def inverse(wave):
    """shdr_wl on a wavelength column: the fractional pixel of a wavelength, a straight
    line between pixels and past the ends, kept between 0.5 and N + 0.5 as
    wf_msp_inv keeps it."""
    wave = np.asarray(wave, dtype=float)
    n = wave.size
    pixel = np.arange(1.0, n + 1.0)
    rising = wave[-1] >= wave[0]
    w, p = (wave, pixel) if rising else (wave[::-1], pixel[::-1])

    def to_pixel(value: float) -> float:
        if value <= w[0]:
            step = (p[1] - p[0]) / (w[1] - w[0])
            x = p[0] + (value - w[0]) * step
        elif value >= w[-1]:
            step = (p[-1] - p[-2]) / (w[-1] - w[-2])
            x = p[-1] + (value - w[-1]) * step
        else:
            x = float(np.interp(value, w, p))
        return max(0.5, min(n + 0.5, x))

    return to_pixel
