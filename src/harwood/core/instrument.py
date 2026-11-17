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
    #: parameter -> the value it has when the header does not say (the third column)
    defaults: dict[str, str] = field(default_factory=dict)

    def keyword(self, parameter: str) -> str:
        """Which header keyword holds this parameter, for this instrument."""
        parameter = parameter.lower()
        return self.keywords.get(parameter, DEFAULTS.get(parameter,
                                                         parameter.upper()))

    def value(self, header, parameter: str, default=None):
        """Read a parameter out of a header, through the translation; its default in
        the translation when the header does not have it (hdmgstr)."""
        key = self.keyword(parameter)
        if key in header:
            return header[key]
        fallback = self.defaults.get(parameter.lower(), "")
        return fallback if fallback != "" else default

    def flag(self, header, parameter: str) -> bool:
        """ccdflag: a processing step is done when its keyword says something other
        than its default -- present at all, when it has none. The camera format writes
        its flags as 0, the default, until a step is done."""
        from ..tasks.imutil import image_parameter

        key = self.keyword(parameter)
        default = self.defaults.get(parameter.lower(), "")
        value = image_parameter(header, "", key)
        text = default if value is None else str(value)
        return text != default

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
    """Read a translation file, IRAF's format (hdrmap): a word, the header keyword or
    value it stands for, and a default -- quoted where they have blanks. A line whose
    second word is an image type says what a header's IMAGETYP value means; any other
    is a parameter's keyword, and default, which IRAF takes for any parameter at all:
    the camera format names its processing flags (fixpix bp-flag 0)."""
    from .ccdinstsession import words

    path = resolve(name)
    instrument = Instrument(name=path.stem, source=str(path))

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = words(raw)
        if not parts or parts[0].startswith("#"):
            continue
        left = parts[0]
        right = parts[1] if len(parts) > 1 else ""
        default = parts[2] if len(parts) > 2 else ""
        if right.lower() in TYPES and left.lower() not in PARAMETERS:
            instrument.types[left.upper()] = right.lower()
            continue
        if left != left.lower() or any(c.isspace() for c in left):
            # a header's value (upper case, or with blanks), not a parameter's name:
            # what it means can only be an image type
            raise ValueError(
                f"{path}: '{right}' is not an image type, for the header value "
                f"'{left}'. Use one of {', '.join(TYPES)}.")
        instrument.keywords[left.lower()] = right
        if default:
            instrument.defaults[left.lower()] = default

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


def settings(name: str | Path) -> list[tuple[str, str, str]]:
    """The parameter settings of an instrument's .cl file, as
    (task, parameter, value text), with `cl < "ccddb$..."` includes read in
    place -- what IRAF's setinstrument runs after choosing the .dat."""
    path = Path(str(name))
    if path.suffix != ".cl":
        try:
            path = resolve(name).with_suffix(".cl")
        except FileNotFoundError:
            return []
    if not path.is_file():
        return []
    found: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="latin-1").splitlines():
        text = line.split("#", 1)[0].strip() if not line.lstrip().startswith("#") else ""
        if not text:
            continue
        if text.startswith("cl") and "<" in text:
            target = text.split("<", 1)[1].strip().strip('"\'')
            if "$" in target:
                prefix, _, rest = target.partition("$")
                for directory in IRAF_CCDDB:
                    if (directory / rest).is_file():
                        found += settings(directory / rest)
                        break
            else:
                found += settings(path.parent / target)
            continue
        left, sep, right = text.partition("=")
        if not sep or "." not in left:
            continue
        task, _, parameter = left.strip().partition(".")
        found.append((task.strip(), parameter.strip(), right.strip().strip('"')))
    return found
