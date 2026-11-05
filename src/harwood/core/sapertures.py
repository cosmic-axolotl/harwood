"""sapertures: an aperture's attributes changed -- its beam, its dispersion (type,
first wavelength, step, redshift), its limits across the slit, its title -- from a
table or from defaults, as IRAF's sapertures changes them in the world coordinates.

From noao/onedspec/t_sapertures.x. Its ways kept:

* w1 and dw are always wavelengths, as smw_gwattrs gives them and smw_swattrs takes
  them, a logarithmic dispersion's too: given dtype 1 with w1 3.6 and dw 0.0001,
  the spectrum runs from 3.6 A in steps that are even in log10.
* The limits are where the aperture is across the slit, not about its centre.
* A table line is aperture, beam, dtype, w1, dw, z, aplow, aphigh and title; with
  fewer than those nine it is aperture, beam and title. A negative beam is ignored,
  and said when verbose. An aperture not in the table takes the defaults.
* wcsreset puts every aperture back in pixels and ignores the table and the defaults.

harwood's spectra keep their wavelengths in a column, the dispersion's over 1+z:
an aperture given dtype -1 loses them, where IRAF keeps w1 and dw in the world
coordinates for another sapertures to bring back.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import iraffmt, spectrum

f32 = np.float32


@dataclass
class Values:
    """One aperture's new values; None is INDEF, keep what it has."""
    ap: int | None = None
    beam: int | None = None
    dtype: int | None = None
    w1: float | None = None
    dw: float | None = None
    z: float | None = None
    aplow: float | None = None
    aphigh: float | None = None
    title: str = "INDEF"


class _Scan:
    """fscan's garg*, a word at a time, INDEF read as nothing given; nscan counts
    what was read."""

    def __init__(self, line: str):
        self.line, self.ip, self.nscan, self.failed = line, 0, 0, False

    def _word(self):
        while self.ip < len(self.line) and self.line[self.ip] in " \t":
            self.ip += 1
        start = self.ip
        while self.ip < len(self.line) and self.line[self.ip] not in " \t":
            self.ip += 1
        return self.line[start:self.ip]

    def number(self, whole: bool):
        if self.failed:
            return None
        save = self.ip
        word = self._word()
        if word.startswith("INDEF"):
            self.nscan += 1
            return None
        value = iraffmt.ctod(word) if word else None
        if value is None or (whole and not word.lstrip("+-").isdigit()):
            self.ip, self.failed = save, True
            return None
        self.nscan += 1
        return int(value) if whole else value

    def rest(self) -> str:
        if self.failed:
            return ""
        self.nscan += 1
        return self.line[self.ip:].strip()


def read_table(text: str, verbose: bool) -> tuple[list[Values], list[str]]:
    """sap_gids from a file: the apertures' values, and what it says on its error
    stream."""
    rows, said = [], []
    for line in text.splitlines():
        scan = _Scan(line)
        ap, beam = scan.number(True), scan.number(True)
        if scan.nscan < 2:
            continue
        if beam is not None and beam < 0 and verbose:
            said.append(f"Negative beam number for aperture {ap} ignored.")
            beam = None
        row = Values(ap=ap, beam=beam)
        row.dtype = scan.number(True)
        row.w1, row.dw, row.z = scan.number(False), scan.number(False), scan.number(False)
        row.aplow, row.aphigh = scan.number(False), scan.number(False)
        row.title = scan.rest()
        if scan.nscan < 9:
            again = _Scan(line)
            ap, beam = again.number(True), again.number(True)
            row = Values(ap=ap, beam=None if beam is not None and beam < 0 else beam,
                         title=again.rest())
        if row.aplow is not None:
            row.aplow = float(f32(row.aplow))
        if row.aphigh is not None:
            row.aphigh = float(f32(row.aphigh))
        rows.append(row)
    return rows, said


def attributes(aperture) -> tuple[int, int, float, float, float]:
    """smw_gwattrs: beam, dtype, w1, dw and z, w1 and dw in wavelength."""
    beam = aperture.number if aperture.beam is None else aperture.beam
    dtype = spectrum.dispersion_type(aperture.wavelength)
    z = float(aperture.redshift or 0.0)
    if dtype == -1:
        return beam, dtype, 1.0, 1.0, z
    d = np.asarray(aperture.wavelength, dtype=float) * (1 + z)
    w1, dw = float(d[0]), float((d[-1] - d[0]) / (d.size - 1))
    if dtype == 0:
        # The world coordinates keep the w1 and dw they were given; from the column
        # their last figure wanders, so a linear one's are taken to 15 figures.
        w1, dw = float(f"{w1:.15g}"), float(f"{dw:.15g}")
    elif dtype == 1:
        # smw_gwattrs: the log10 start and step as the world coordinates' text holds
        # them, read back with IRAF's ctod, and turned into wavelengths.
        n = d.size
        a = iraffmt.ctod(f"{math.log10(d[0]):.15g}")
        b = iraffmt.ctod(f"{(math.log10(d[-1]) - math.log10(d[0])) / (n - 1):.15g}")
        w1 = 10.0 ** a
        dw = w1 * (10.0 ** ((n - 1) * b) - 1) / (n - 1)
    return beam, dtype, w1, dw, z


def change(aperture, row: Values, title: str) -> list[str] | None:
    """sap_ms on one aperture: the change made and what verbose says of it, or None
    when nothing changes. `title` is the aperture's as it is."""
    g = lambda v: iraffmt.g(float(v), double=True)  # noqa: E731
    r = lambda v: iraffmt.g(float(f32(v)))  # noqa: E731
    beam, dtype, w1, dw, z = attributes(aperture)
    low, high = float(f32(aperture.centre + aperture.low)), \
        float(f32(aperture.centre + aperture.high))
    obeam = beam if row.beam is None else row.beam
    odtype = dtype if row.dtype is None else max(-1, min(1, row.dtype))
    ow1 = w1 if row.w1 is None else row.w1
    odw = dw if row.dw is None else row.dw
    oz = z if row.z is None else row.z
    olow = low if row.aplow is None else row.aplow
    ohigh = high if row.aphigh is None else row.aphigh
    newtitle = row.title not in ("INDEF", "") and row.title != title
    if not (obeam != beam or odtype != dtype or ow1 != w1 or odw != dw or oz != z
            or olow != low or ohigh != high or newtitle):
        return None
    lines = [f"  Aperture {aperture.number}:"]
    for name, old, new, show in (("beam", beam, obeam, str), ("dtype", dtype, odtype, str),
                                 ("w1", w1, ow1, g), ("dw", dw, odw, g), ("z", z, oz, g),
                                 ("aplow", low, olow, r), ("aphigh", high, ohigh, r)):
        if new != old:
            lines.append(f"    {name} {show(old)} --> {show(new)}")
    if newtitle:
        lines.append(f"    apid {title} --> {row.title}")
    aperture.beam = obeam
    n = len(aperture.flux)
    pixel = np.asarray(aperture.pixel if aperture.pixel is not None
                       else np.arange(1.0, n + 1.0), dtype=float)
    if odtype == -1:
        aperture.wavelength = None
        aperture.redshift = 0.0
    else:
        if odtype == dtype and ow1 == w1 and odw == dw or odtype == 2:
            d = np.asarray(aperture.wavelength, dtype=float) * (1 + z)
        elif odtype == 0:
            d = ow1 + odw * (pixel - 1)
        else:
            l1, l2 = math.log10(ow1), math.log10(ow1 + odw * (n - 1))
            d = 10 ** (l1 + (l2 - l1) / (n - 1) * (pixel - 1))
        aperture.wavelength = d / (1 + oz)
        aperture.redshift = oz
    aperture.low, aperture.high = olow - aperture.centre, ohigh - aperture.centre
    if newtitle:
        aperture.title = row.title
    return lines
