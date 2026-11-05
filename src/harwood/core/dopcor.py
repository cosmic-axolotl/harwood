"""dopcor: a Doppler correction of spectra -- their wavelengths to the rest frame of
a redshift or a velocity, and their fluxes by a power of 1+z.

From noao/onedspec/t_dopcor.x. IRAF keeps the correction in the spectrum's world
coordinates, as a z each aperture carries beside its dispersion function; harwood's
wavelengths are in the table, so each aperture keeps the z it has had (`APZ` in its
header) and the wavelengths are the dispersion's over 1+z. As in IRAF, a new z
takes the place of the old one unless it is added to it: (1+z)(1+zold) - 1.

The redshift is a number or a header keyword; a + or - before either flips it, a
velocity by its sign and a redshift to 1/(1+z) - 1 -- but only when it is read from
the header: a number after a - is only made negative. A velocity is relativistic,
sqrt((1+v/c)/(1-v/c)) - 1, c 299792.5 km/s. The flux factor is computed and applied
in real, as IRAF's is.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
import re

import numpy as np

from . import advice, iraffmt

VLIGHT = 2.997925e5         #: km/s, t_dopcor.x's
f32 = np.float32

#: ctod: what of a string is a number, from its start.
_NUMBER = re.compile(r"\s*[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?")


def _ctod(text: str) -> float | None:
    found = _NUMBER.match(text)
    if found is None:
        return None
    return float(found.group(0).replace("d", "e").replace("D", "e"))


def redshift(text: str, header, isvelocity: bool) -> tuple[float, float]:
    """The z to apply, and what was given (the velocity, when it is one), from the
    redshift parameter as t_dopcor reads it: a number or a keyword of `header`."""
    text = str(text)
    sign = text[:1] if text[:1] in "+-" else ""
    word = text[1:] if sign else text
    z = _ctod(word)
    if z is None:
        key = word.strip()
        if key not in header:
            advice.fail(f"Image header parameter not found ({key})",
                        "give the redshift or velocity as a number, or the name of a "
                        "keyword the spectrum's header has.", KeyError)
        z = float(header[key])
        if sign == "-":
            z = -z if isvelocity else 1 / (1 + z) - 1
    elif sign == "-":
        z = -z
    given = z
    if isvelocity:
        z = z / VLIGHT
        if abs(z) >= 1.0:
            advice.fail("Impossible velocity", "a velocity is in km/s, and less than "
                        "that of light.")
        z = math.sqrt((1 + z) / (1 - z)) - 1
    if z <= -1.0:
        advice.fail("Impossible redshift", "a redshift is more than -1.")
    return z, given


def flux_factor(z: float, factor: float) -> np.float32:
    """fcval = (1+z)**ffac, kept in real."""
    return f32((1 + z) ** float(f32(factor)))


def correct(aperture, z: float, add: bool, dispersion: bool, fcval) -> tuple[float, float]:
    """One aperture: its wavelengths to the new z, its bands by the flux factor (None
    for no flux correction). Gives the old z and the new."""
    zold = float(getattr(aperture, "redshift", 0.0) or 0.0)
    znew = zold
    if dispersion:
        znew = (1 + z) * (1 + zold) - 1 if add else z
        wave = np.asarray(aperture.wavelength, dtype=float)
        aperture.wavelength = wave * (1 + zold) / (1 + znew)
        aperture.redshift = znew
    if fcval is not None:
        for band in ("flux", "error", "sky", "raw"):
            values = getattr(aperture, band, None)
            if values is not None:
                setattr(aperture, band,
                        (np.asarray(values, dtype=f32) * f32(fcval)).astype(float))
    return zold, znew


def log_card(z: float, given: float, isvelocity: bool, factor: float | None,
             apertures: str) -> str:
    """The DOPCORnn card: "%8g %g %s" with a flux correction, "%8g %s" without."""
    value = _g8(given if isvelocity else z)
    if factor is None:
        return f"{value} {apertures}"
    return f"{value} {iraffmt.g(float(f32(factor)))} {apertures}"


def card_name(header) -> str:
    """The first DOPCORnn the header does not have; past 98, DOPCOR98 again."""
    for i in range(1, 99):
        if f"DOPCOR{i:02d}" not in header:
            return f"DOPCOR{i:02d}"
    return "DOPCOR98"


def verbose_lines(output: str, apertures: str, z: float, given: float, isvelocity: bool,
                  factor: float, add: bool, zold: float, znew: float) -> list[str]:
    """What verbose prints for an image, as t_dopcor does: the old and the new z are
    those of the last aperture it corrected."""
    line = f"{output}: Doppler correction - apertures={apertures},"
    if isvelocity:
        line += f" velocity={_g8(given)},"
    line += f" redshift={_g8(z)}, flux factor={iraffmt.g(float(f32(factor)))}"
    lines = [line]
    if add and zold != 0.0:
        lines.append(f"    Correction added: {iraffmt.g(zold, double=True)} + "
                     f"{iraffmt.g(z, double=True)} = {iraffmt.g(znew, double=True)}")
    return lines


def _g8(value: float) -> str:
    return iraffmt.printf(float(value), "g", width=8, double=True).rjust(8)
