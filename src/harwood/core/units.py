"""IRAF's dispersion units: Angstroms, nanometers... meters, hertz to gigahertz, m/s
and km/s from a reference wavelength, ev, kev, mev, z; log and inverse of any.

From noao/onedspec/smw/units.x (un_open, un_decode, un_ctrand) and units.h. A
units string is up to three words, the first that is a unit ending it: "inverse
centimeters", "log angstroms", "km/s 6563 angstroms" (a velocity's reference
wavelength, in any units but a velocity). An abbreviation of noao$lib/units.dat
(nm, Hz, wn...) is the units it stands for; any word may be shortened as far as
it stays one. The scales and the reference are kept in real, as IRAF keeps them,
and the conversions go through Angstroms in double; a velocity is relativistic.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import advice, iraffmt

f32 = np.float32
ABBREVIATIONS = Path(__file__).resolve().parent.parent / "data" / "iraf" / "lib" / "units.dat"
WAVE, FREQ, VEL, ENERGY, DOP = 1, 2, 3, 4, 5
#: units.h: each unit, its class and its scale from Angstroms.
UNITS = [("angstroms", WAVE, 1.0), ("nanometers", WAVE, 0.1), ("millimicrons", WAVE, 0.1),
         ("microns", WAVE, 1e-4), ("millimeters", WAVE, 1e-7), ("centimeters", WAVE, 1e-8),
         ("meters", WAVE, 1e-10), ("hertz", FREQ, 2.9979e18), ("kilohertz", FREQ, 2.9979e15),
         ("megahertz", FREQ, 2.9979e12), ("gigahertz", FREQ, 2.9979e9),
         ("m/s", VEL, 2.9979e8), ("km/s", VEL, 2.9979e5), ("ev", ENERGY, 1.2396e4),
         ("kev", ENERGY, 1.2396e1), ("mev", ENERGY, 1.2396e-2), ("z", DOP, 1.0)]
LABELS = {WAVE: "Wavelength", FREQ: "Frequency", VEL: "Velocity", ENERGY: "Energy",
          DOP: "Redshift"}


@dataclass
class Units:
    """un_open's structure: the class (0 unknown), the scale, log and inverse, the
    reference wavelength of a velocity (in Angstroms), and the label and units it
    is shown with."""
    text: str
    kind: int = 0
    scale: float = 1.0
    log: bool = False
    inverse: bool = False
    vref: float = 0.0
    label: str = ""
    units: str = ""


def _abbreviations() -> dict[str, str]:
    out = {}
    if ABBREVIATIONS.is_file():
        for line in ABBREVIATIONS.read_text(encoding="utf-8").splitlines():
            words = line.split(None, 1)
            if len(words) == 2:
                out[words[0]] = words[1].strip().strip('"')
    return out


def _dic(word: str) -> int:
    """strdic on UN_DIC: 1.. the units, then log and inverse; 0 for none."""
    names = [name for name, _, _ in UNITS] + ["log", "inverse"]
    if word in names:
        return names.index(word) + 1
    found = [i for i, name in enumerate(names, 1) if name.startswith(word)]
    return found[0] if len(found) == 1 else 0


def _decode1(text: str) -> tuple[Units, str]:
    """un_decode1: the units, and the reference's own units string after a velocity."""
    un = Units(text)
    words = _abbreviations().get(text, text).lower().split()
    log = inverse = False
    for i, word in enumerate(words[:3]):
        j = _dic(word)
        if j > len(UNITS):
            if j - len(UNITS) == 1:
                if log:
                    break
                log = True
            else:
                if inverse:
                    break
                inverse = True
            continue
        if j == 0:
            break
        name, kind, scale = UNITS[j - 1]
        rest = ""
        if kind in (VEL, DOP):
            if len(words) < i + 3:
                advice.fail("Error in velocity reference wavelength",
                            "a velocity's units are km/s and the reference wavelength "
                            "with its units: km/s 6563 angstroms.")
            un.vref = float(f32(iraffmt.ctod(words[i + 1])))
            rest = " ".join(words[i + 2:])
        un.kind, un.scale, un.log, un.inverse = kind, float(f32(scale)), log, inverse
        un.label = ("Log " if log else "") + (
            "Wavenumber" if kind == WAVE and inverse else LABELS[kind])
        un.units = ("inverse " if inverse else "") + name
        return un, rest
    un.units = text
    return un, ""


def decode(text: str) -> Units:
    """un_decode: the units, a velocity's reference turned into Angstroms."""
    un, rest = _decode1(text or "angstroms")
    if un.kind in (VEL, DOP):
        reference, _ = _decode1(rest)
        if reference.kind in (VEL, DOP):
            advice.fail("Velocity reference units may not be velocity",
                        "give the reference in a wavelength's units: km/s 6563 angstroms.")
        un.vref = float(f32(convert([un.vref], reference, decode("angstroms"),
                                    real=True)[0]))
    return un


def _same(a: Units, b: Units) -> bool:
    return (a.kind, a.scale, a.log, a.inverse, a.vref) == \
        (b.kind, b.scale, b.log, b.inverse, b.vref) and a.kind != 0


def convert(values, a: Units, b: Units, real: bool = False) -> np.ndarray:
    """un_ctrand (un_ctranr with `real`): values in a's units in b's."""
    values = np.array(values, dtype=f32 if real else float)
    if _same(a, b):
        return values
    if a.kind == 0 or b.kind == 0:
        advice.fail("Cannot convert between selected units",
                    "give units IRAF knows: angstroms, nm, microns, hertz, ev, km/s...")
    t = f32 if real else float
    out = []
    for v in values:
        v = t(v)
        s = t(a.scale)
        if a.log:
            v = t(10.0 ** v)
        if a.inverse:
            v = t(1.0 / v)
        if a.kind == WAVE:
            v = t(v / s)
        elif a.kind in (FREQ, ENERGY):
            v = t(s / v)
        elif a.kind == VEL:
            z = t(v / s)
            v = t(math.sqrt((1 + z) / (1 - z)) * t(a.vref))
        elif a.kind == DOP:
            v = t((v / s + 1) * t(a.vref))
        s = t(b.scale)
        if b.kind == WAVE:
            v = t(v * s)
        elif b.kind in (FREQ, ENERGY):
            v = t(s / v)
        elif b.kind == VEL:
            z = t((v / t(b.vref)) ** 2)
            v = t((z - 1) / (z + 1) * s)
        elif b.kind == DOP:
            v = t((v / t(b.vref) - 1) * s)
        if b.inverse:
            v = t(1.0 / v)
        if b.log:
            v = t(math.log10(v))
        out.append(v)
    return np.array(out, dtype=f32 if real else float)
