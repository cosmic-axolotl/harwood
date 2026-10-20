"""Instrument translation: teaching harwood to read one observatory's headers.

Every observatory writes its headers differently. KPNO's direct camera puts the
filter in `FILTERS` and calls a flat field `DOME FLAT`; the next instrument
calls the keyword `FILTER` and the value `flat`. A reduction package that
hard-codes one of them works at one telescope.

IRAF solved this with small translation files under `ccddb$`, and the solution
is good enough that harwood reads IRAF's own files rather than inventing a
format. Two kinds of line, exactly as IRAF writes them::

    subset          filters          # this parameter lives in that keyword
    'DOME FLAT'     flat             # this header value means that image type

Everything downstream then speaks one vocabulary -- `object`, `zero`, `dark`,
`flat`, `comp` -- and `ccdproc` can pick the right bias and the right flat for
the right filter out of a directory without being told.

The translation in force is named in the project file, so it is part of the
reduction's record rather than a setting on the machine.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

#: Where `ccddb$` is looked for: harwood's own copy of IRAF's translations
#: first, then an installed IRAF, for a file someone has added to theirs.
IRAF_CCDDB = (
    Path(__file__).resolve().parent.parent / "data" / "iraf" / "ccddb",
    # Lower case on purpose: `iraf` is the name IRAF itself sets.
    Path(os.environ.get("iraf", "/usr/lib/iraf")) / "noao" / "imred" / "ccdred" / "ccddb",  # noqa: SIM112
    Path("/usr/share/iraf/noao/imred/ccdred/ccddb"),
)

#: The image types the reduction speaks. Everything a header says is mapped
#: onto one of these, or onto `unknown`.
TYPES = ("object", "zero", "dark", "flat", "illum", "fringe", "comp", "other",
         "unknown")

#: Parameters that may be redirected to a different keyword. A line whose first
#: word is one of these is a keyword mapping; anything else is an image type.
PARAMETERS = ("imagetyp", "subset", "exptime", "darktime", "biassec", "trimsec",
              "ccdsec", "datasec", "ccdmean", "fixfile", "gain", "rdnoise",
              "observat", "date-obs", "ut", "airmass", "filter")

#: What harwood looks for when the translation says nothing.
DEFAULTS = {
    "imagetyp": "IMAGETYP",
    "subset": "FILTERS",
    "exptime": "EXPTIME",
    "darktime": "DARKTIME",
    "biassec": "BIASSEC",
    "trimsec": "TRIMSEC",
    "ccdsec": "CCDSEC",
    "datasec": "DATASEC",
    "gain": "GAIN",
    "rdnoise": "RDNOISE",
}


@dataclass
class Instrument:
    """One instrument's header conventions."""

    name: str = "none"
    source: str = ""
    #: parameter -> header keyword
    keywords: dict[str, str] = field(default_factory=dict)
    #: header value (upper case) -> image type
    types: dict[str, str] = field(default_factory=dict)

    def keyword(self, parameter: str) -> str:
        """Which header keyword holds this parameter, for this instrument."""
        parameter = parameter.lower()
        return self.keywords.get(parameter, DEFAULTS.get(parameter,
                                                         parameter.upper()))

    def value(self, header, parameter: str, default=None):
        """Read a parameter out of a header, through the translation."""
        return header.get(self.keyword(parameter), default)

    def image_type(self, header) -> str:
        """What kind of exposure this is, in harwood's vocabulary.

        An untranslated value is reported as `unknown` rather than guessed at.
        A frame whose type harwood cannot read is one it must not process by
        accident, and saying so is how the translation file gets fixed.
        """
        raw = self.value(header, "imagetyp")
        if raw in (None, ""):
            return "unknown"
        text = str(raw).strip()
        if text.upper() in self.types:
            return self.types[text.upper()]
        lowered = text.lower()
        return lowered if lowered in TYPES else "unknown"

    def subset(self, header) -> str:
        """The filter, or whatever else separates one set of flats from another.

        IRAF calls this the subset because it need not be a filter -- for a
        spectrograph it is the grating and its angle. The word is worth keeping
        for that reason.
        """
        raw = self.value(header, "subset")
        return "" if raw in (None, "") else str(raw).strip()

    def describe(self) -> str:
        return (f"{self.name}: {len(self.keywords)} keyword translations, "
                f"{len(self.types)} image types")


def resolve(name: str | Path) -> Path:
    """Find a translation file, accepting IRAF's `ccddb$kpno/direct.dat`."""
    text = str(name)
    if "$" in text:
        prefix, _, rest = text.partition("$")
        if prefix != "ccddb":
            raise ValueError(f"unknown IRAF directory prefix '{prefix}$'")
        for directory in IRAF_CCDDB:
            candidate = directory / rest
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"there is no translation '{rest}' under ccddb$. "
            "`harwood setinstrument --list-available` shows them all; or give "
            "the path of a file of your own."
        )
    path = Path(text)
    if path.is_file():
        return path
    # A bare name: try IRAF's tree, which is how people refer to these.
    for directory in IRAF_CCDDB:
        for candidate in (directory / text, directory / f"{text}.dat"):
            if candidate.is_file():
                return candidate
        for found in sorted(directory.glob(f"*/{text}.dat")):
            return found
    raise FileNotFoundError(
        f"no instrument translation called '{text}'. "
        f"`harwood setinstrument --list` shows what is installed."
    )


def load(name: str | Path) -> Instrument:
    """Read a translation file, IRAF's format."""
    path = resolve(name)
    instrument = Instrument(name=path.stem, source=str(path))

    for number, raw in enumerate(path.read_text(encoding="utf-8",
                                                errors="replace").splitlines(),
                                 start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            parts = shlex.split(line)
        except ValueError:
            parts = line.split()
        if len(parts) < 2:
            raise ValueError(
                f"{path}:{number}: expected two words, got {line!r}. A line is "
                "either 'parameter keyword' or 'HEADER-VALUE imagetype'."
            )
        left, right = parts[0], parts[1]

        if left.lower() in PARAMETERS:
            instrument.keywords[left.lower()] = right
            continue
        if right.lower() not in TYPES:
            raise ValueError(
                f"{path}:{number}: '{right}' is not an image type. "
                f"Use one of {', '.join(TYPES)}."
            )
        instrument.types[left.upper()] = right.lower()

    return instrument


def installed() -> dict[str, list[str]]:
    """Every translation file available, grouped by observatory."""
    found: dict[str, list[str]] = {}
    for directory in IRAF_CCDDB:
        if not directory.is_dir():
            continue
        for child in sorted(directory.iterdir()):
            if child.is_dir():
                names = sorted(p.stem for p in child.glob("*.dat"))
                if names:
                    # The same file in harwood's copy and in an IRAF: once.
                    have = found.setdefault(child.name, [])
                    have.extend(n for n in names if n not in have)
    return {place: sorted(names) for place, names in found.items()}


def from_project(start: Path | None = None) -> Instrument:
    """The translation this reduction uses, from `harwood.yaml`.

    With none set, a neutral instrument that reads the usual keywords. That
    works on well-behaved modern data and fails visibly on anything else,
    which is the right way round.
    """
    from . import params as store

    document, _ = store.load_project(start)
    name = document.get("instrument")
    if not name:
        return Instrument()
    return load(name)
