"""What is that line? -- naming the atom, ion or molecule behind a feature.

A calibrated spectrum gives wavelengths. A wavelength is not yet a fact about
the object: 6707.8 Angstroms is a number, and "neutral lithium, the resonance
doublet that burns away when a star mixes its surface below two and a half
million degrees" is a fact. This module does the second step.

It is deliberately a *curated* catalogue rather than a complete one, and that
is the whole design. A complete atomic line list has millions of entries, and
against a million lines every wavelength is within a hundredth of an Angstrom
of something -- so a complete list answers every query confidently and is
useless. What is here are the lines that name themselves: the ones a
spectroscopist would point at. A feature this catalogue cannot name is better
reported as unidentified than assigned to the nearest of a thousand
possibilities.

Three things it refuses to hide:

**Wavelength alone is not an identification.** Where two species sit within the
tolerance, both are returned, and the result says it is ambiguous. Deciding
between them takes something the spectrum does not contain -- what you pointed
the telescope at, how reddened it is, what else is present.

**Radial velocity moves everything.** A star at 100 km/s shifts H-alpha by 2.2
Angstroms, which is more than enough to land it on the wrong identification.
The shift is applied explicitly, and the telluric lines -- which do not move,
because they are in the air above the telescope -- are the check on whether it
was applied correctly.

**Some features are not in the object at all.** The O2 A band at 7594 is
oxygen above the telescope; the sodium D absorption may be a cloud in the way
rather than the star's photosphere. The catalogue says which, because that is
usually the more useful half of the answer.

Wavelengths are air wavelengths in Angstroms, as every optical catalogue and
every IRAF line list is.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from pathlib import Path

#: Speed of light in km/s, for Doppler shifts.
LIGHT = 299792.458

#: Where a catalogue entry comes from, and what that means for the reduction.
ORIGINS = {
    "stellar": "in the star's own atmosphere",
    "nebular": "in ionised gas around or between stars",
    "interstellar": "in a cloud between here and the object",
    "telluric": "in the air above the telescope -- not in the object at all",
    "lamp": "in the calibration lamp",
}

#: What kind of thing produces the line.
KINDS = {
    "atomic": "a neutral atom",
    "ionic": "an ion",
    "forbidden": "a forbidden transition, possible only at very low density",
    "molecular": "a molecule, usually a band rather than a single line",
    "telluric": "a band in the Earth's atmosphere",
    "interstellar": "an interstellar feature",
}


@dataclass(frozen=True)
class Line:
    """One catalogue entry: a wavelength and what makes it."""

    species: str
    wavelength: float
    label: str = ""
    kind: str = "atomic"
    origin: tuple[str, ...] = ()
    prominence: int = 1
    note: str = ""
    about: str = ""          #: the note attached to the species as a whole
    common: str = ""         #: the species in words -- "neutral lithium"

    @property
    def name(self) -> str:
        """How to print it: the label if it has one, else species and value."""
        return self.label or f"{self.species} {self.wavelength:.2f}"

    def describe(self) -> str:
        parts = [f"{self.species} at {self.wavelength:.2f} A"]
        if self.common:
            parts.append(f"({self.common})")
        where = [ORIGINS[origin] for origin in self.origin if origin in ORIGINS]
        if where:
            parts.append("-- " + "; ".join(where))
        return " ".join(parts)


@dataclass
class Candidate:
    """One possible identification of a measured feature."""

    line: Line
    offset: float            #: measured minus rest, in Angstroms
    score: float

    @property
    def velocity(self) -> float:
        """The radial velocity this identification would imply, in km/s."""
        return LIGHT * self.offset / self.line.wavelength


@dataclass
class Match:
    """What a measured wavelength might be, with how sure that is."""

    wavelength: float
    candidates: list[Candidate] = field(default_factory=list)
    velocity: float = 0.0    #: the shift that was assumed, in km/s

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def confident(self) -> bool:
        """True when one candidate clearly beats the rest.

        The rule is the one `search_dispersion` uses for the same reason: a
        winner that barely beats its rival has not won. Twice the score is the
        threshold, and it is a judgement, not a probability.
        """
        if not self.candidates:
            return False
        if len(self.candidates) == 1:
            return True
        return self.candidates[0].score > 2.0 * self.candidates[1].score

    def describe(self) -> str:
        if not self.candidates:
            return f"{self.wavelength:.2f} A: not in the catalogue"
        best = self.candidates[0]
        text = f"{self.wavelength:.2f} A: {best.line.name}"
        if not self.confident:
            others = ", ".join(item.line.name for item in self.candidates[1:3])
            text += f" -- or {others}; the wavelength alone cannot choose"
        return text


def air_to_vacuum(wavelength: float) -> float:
    """Air wavelength to vacuum, by the IAU standard (Morton 1991).

    Catalogues below 2000 A are in vacuum and above it in air, because air
    does not transmit in the ultraviolet. Mixing the two is a 2 Angstrom error
    in the red -- thirteen pixels of the spectrograph in the exercises, and
    entirely invisible as anything but a wrong identification.
    """
    if wavelength < 2000.0:
        return wavelength
    sigma = 1e4 / wavelength
    factor = (1.0 + 6.4328e-5 + 2.94981e-2 / (146.0 - sigma ** 2)
              + 2.5540e-4 / (41.0 - sigma ** 2))
    return wavelength * factor


def vacuum_to_air(wavelength: float) -> float:
    """Vacuum wavelength to air, the inverse of `air_to_vacuum`."""
    if wavelength < 2000.0:
        return wavelength
    sigma = 1e4 / wavelength
    factor = (1.0 + 8.34254e-5 + 2.406147e-2 / (130.0 - sigma ** 2)
              + 1.5998e-4 / (38.9 - sigma ** 2))
    return wavelength / factor


def shifted(wavelength: float, velocity: float) -> float:
    """Where a rest wavelength lands at a given radial velocity, in km/s."""
    return wavelength * (1.0 + velocity / LIGHT)


def rest(wavelength: float, velocity: float) -> float:
    """The rest wavelength of a feature measured at this one."""
    return wavelength / (1.0 + velocity / LIGHT)


@functools.cache
def catalogue() -> tuple[Line, ...]:
    """Every line harwood can name, sorted by wavelength."""
    import yaml

    path = Path(__file__).resolve().parent.parent / "data" / "spectral_lines.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))

    lines = []
    for group in document["groups"]:
        origin = tuple(group.get("origin", ()))
        for entry in group["lines"]:
            lines.append(Line(
                species=group["species"],
                wavelength=float(entry["wavelength"]),
                label=entry.get("label", ""),
                kind=group.get("kind", "atomic"),
                origin=origin,
                prominence=int(entry.get("prominence", 1)),
                note=_tidy(entry.get("note", "")),
                about=_tidy(group.get("note", "")),
                common=group.get("common", ""),
            ))
    return tuple(sorted(lines, key=lambda line: line.wavelength))


def _tidy(text: str) -> str:
    """YAML block scalars keep their line breaks; prose does not want them."""
    return " ".join(str(text).split())


def at(wavelength: float, tolerance: float = 1.0, velocity: float = 0.0,
       origin: str | None = None, limit: int = 4) -> Match:
    """What might be at this measured wavelength.

    `velocity` is the object's radial velocity in km/s: the measurement is
    moved back to rest before the catalogue is consulted. `origin` narrows the
    search to one kind of source -- 'stellar', 'telluric' and so on -- which
    is how outside knowledge enters, since the spectrum itself cannot say.
    """
    centre = rest(wavelength, velocity)
    sigma = max(tolerance, 1e-9) / 2.0

    candidates = []
    for line in catalogue():
        offset = centre - line.wavelength
        if abs(offset) > tolerance:
            continue
        if origin and origin not in line.origin:
            continue
        # Closeness and prominence both count. A faint line half a tolerance
        # away should not beat H-alpha sitting on the measurement, and
        # H-alpha a whole tolerance away should not beat a faint line on it.
        weight = line.prominence * math.exp(-0.5 * (offset / sigma) ** 2)
        candidates.append(Candidate(line, wavelength - shifted(line.wavelength, velocity),
                                    weight))

    candidates.sort(key=lambda item: -item.score)
    return Match(wavelength, candidates[:limit], velocity)


def identify(wavelengths, tolerance: float = 1.0, velocity: float = 0.0,
             origin: str | None = None) -> list[Match]:
    """Name a whole list of measured wavelengths."""
    return [at(float(value), tolerance, velocity, origin) for value in wavelengths]


def within(low: float, high: float, minimum: int = 1,
           origin: str | None = None) -> list[Line]:
    """Every catalogued line in a wavelength range.

    This is what a plot overlay wants: not a search around a measurement, but
    everything worth marking between the ends of the spectrum.
    """
    return [line for line in catalogue()
            if low <= line.wavelength <= high
            and line.prominence >= minimum
            and (origin is None or origin in line.origin)]


@dataclass
class Mark:
    """One line to draw on a spectrum, at the wavelength it actually lands."""

    line: Line
    at: float                #: where it falls, after any Doppler shift
    detected: bool = False   #: a measured feature sits within tolerance of it

    @property
    def colour_key(self) -> str:
        """Which origin decides how it is drawn."""
        return self.line.origin[0] if self.line.origin else "lamp"


def marks(low: float, high: float, velocity: float = 0.0,
          origins=None, minimum: int = 2, measured=None,
          tolerance: float = 1.0) -> list[Mark]:
    """Which catalogue lines to draw over a spectrum, and where.

    Separate from the panel that draws them, so it can be checked without a
    screen. Two subtleties live here rather than in the drawing:

    A radial velocity moves the object's lines but not the telluric bands,
    because those are in the air above the telescope. Drawing them shifted is
    the mistake that makes a wrong velocity look right.

    The range has to be widened before the shift is applied, or a line whose
    rest wavelength sits just outside the spectrum but whose shifted position
    falls inside it is never drawn.
    """
    margin = abs(velocity) / LIGHT * max(abs(high), 1.0) + 1.0
    found = [line for line in catalogue()
             if low - margin <= line.wavelength <= high + margin
             and line.prominence >= minimum
             and (origins is None
                  or any(origin in origins for origin in line.origin))]

    positions = list(measured) if measured is not None else None
    out = []
    for line in found:
        shift = 0.0 if "telluric" in line.origin else velocity
        where = shifted(line.wavelength, shift)
        if not low <= where <= high:
            continue
        seen = bool(positions is not None
                    and any(abs(value - where) <= tolerance for value in positions))
        out.append(Mark(line, where, seen))
    return out


def named(text: str) -> list[Line]:
    """Every line of one species, found by name or by common name."""
    key = text.strip().lower()
    exact = [line for line in catalogue() if line.species.lower() == key]
    if exact:
        return exact
    return [line for line in catalogue()
            if key in line.species.lower() or key in line.common.lower()
            or key in line.label.lower()]


def species() -> list[str]:
    """Every species in the catalogue, in the order they first appear."""
    return list(dict.fromkeys(line.species for line in catalogue()))


def velocity_from(matches: list[Match]) -> tuple[float, float] | None:
    """The radial velocity a set of confident identifications implies.

    Returns the median and the scatter, in km/s, or None when too few lines
    were identified confidently for the number to mean anything. The scatter
    is the useful half: velocities that agree to a few km/s are a measurement,
    and velocities scattered over hundreds are a set of misidentifications.
    """
    import numpy as np

    values = [match.best.velocity for match in matches
              if match.confident and match.best is not None
              and "telluric" not in match.best.line.origin]
    if len(values) < 3:
        return None
    array = np.asarray(values, dtype=float)
    return float(np.median(array)), float(np.std(array))
