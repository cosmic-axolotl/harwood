"""Reading and writing catalogues, including the file names IRAF taught people.

A user coming from IRAF writes `m92010.coo` and `m92010.mag`, because those are
the extensions IRAF used. Astropy cannot guess a format from them and fails
with a wall of text listing every format it knows, which is a bad answer to a
reasonable habit.

So harwood owns this: **one writer, one reader, any extension.** The content is
ECSV -- self-describing, with units and metadata, readable by anything in the
Python ecosystem -- and the name is whatever the user wants to call it.

The metadata carries `harwood_convention`, which is how a downstream task knows
the coordinates are 1-based without having to assume (section 2.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Extensions IRAF used, mapped to nothing in particular -- they are just names.
IRAF_SUFFIXES = {".coo", ".mag", ".ctr", ".sky", ".als", ".pst", ".psg", ".nst",
                 ".arv", ".obs", ".ctg", ".dat", ".lis", ".txt"}

CONVENTION = "1-based (x,y), as IRAF and DS9"


def write(table: Any, path: Path | str, **meta: Any) -> Path:
    """Write a catalogue as ECSV, whatever the file is called."""
    path = Path(path)
    table.meta.setdefault("harwood_convention", CONVENTION)
    table.meta.update(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(str(path), format="ascii.ecsv", overwrite=True)
    return path


def read(path: Path | str) -> Any:
    """Read a catalogue, trying the sensible things before giving up.

    Order: ECSV (what harwood writes), then whatever astropy guesses from the
    name, then plain whitespace-separated text. An IRAF `.coo` file written by
    IRAF itself lands in the last of those.
    """
    from astropy.table import Table

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"catalogue '{path}' does not exist")

    attempts: list[tuple[str, dict]] = [("ascii.ecsv", {})]
    if _iraf_database(path):
        # An apphot/daophot text database names its columns in #N lines;
        # read as plain text it would be col1, col2 ... and lose them.
        attempts.append(("ascii.daophot", {}))
    if _suffix(path) not in IRAF_SUFFIXES:
        attempts.append(("", {}))                      # let astropy guess
    attempts.append(("ascii", {}))
    attempts.append(("ascii.daophot", {}))

    errors = []
    for fmt, options in attempts:
        try:
            table = Table.read(str(path), **({"format": fmt} if fmt else {}), **options)
        except Exception as error:
            errors.append(f"{fmt or 'auto'}: {type(error).__name__}")
            continue
        if fmt == "ascii" and _iraf_database(path):
            # Written by hand or by a script, the columns no longer sit
            # where #F says; split on blanks and keep #N's names.
            names = _iraf_names(path)
            if len(names) == len(table.colnames):
                table.rename_columns(table.colnames, names)
        if _iraf_database(path, signed=True) and "harwood_convention" not in table.meta:
            # IRAF's own tasks wrote it (#K IRAF, #N names): 1-based.
            table.meta["harwood_convention"] = CONVENTION
        return table
    raise ValueError(
        f"cannot read '{path}' as a catalogue. Tried {', '.join(errors)}. "
        "harwood writes ECSV; if this came from IRAF, check it is a text table."
    )


def _suffix(path: Path) -> str:
    """The suffix, past IRAF's version number: obj.coo.1 is a .coo."""
    suffixes = [x.lower() for x in path.suffixes]
    while suffixes and suffixes[-1][1:].isdigit():
        suffixes.pop()
    return suffixes[-1] if suffixes else ""


def _iraf_database(path: Path, signed: bool = False) -> bool:
    """An IRAF text database: #N lines naming the columns; `signed`, also
    the #K IRAF line IRAF's tasks put at the top."""
    names = iraf = False
    try:
        with open(path, errors="replace") as handle:
            for _, line in zip(range(400), handle, strict=False):
                names = names or line.startswith("#N ")
                iraf = iraf or line.startswith("#K IRAF ")
                if line.strip() and not line.startswith("#"):
                    break
    except OSError:
        return False
    return names and (iraf or not signed)


def _iraf_names(path: Path) -> list[str]:
    """The column names of a single-line-record IRAF text database."""
    names = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("#N "):
            names += line[3:].replace("\\", " ").split()
    return names


def coordinates(table: Any, path: Path | str) -> tuple[Any, Any]:
    """Get 1-based (x, y) out of a catalogue, refusing to guess the origin.

    A catalogue harwood wrote says which convention it uses. One from elsewhere
    does not, and quietly assuming 1-based would put a systematic offset into
    every magnitude -- so that assumption is named in an error rather than made
    in silence.
    """
    import numpy as np

    names = {name.lower(): name for name in table.colnames}
    for xcol, ycol in (("x", "y"), ("xcenter", "ycenter"), ("xcentroid", "ycentroid"),
                       ("xc", "yc"), ("col1", "col2")):
        if xcol in names and ycol in names:
            if xcol != "x" and "harwood_convention" not in table.meta:
                raise ValueError(
                    f"'{path}' has columns '{names[xcol]}'/'{names[ycol]}' and no "
                    "harwood convention in its metadata, so its pixel origin is "
                    "unknown. Rename them to x/y once you have confirmed they are "
                    "1-based, as IRAF and DS9 write them."
                )
            return (np.asarray(table[names[xcol]], dtype=float),
                    np.asarray(table[names[ycol]], dtype=float))
    raise ValueError(
        f"'{path}' has no coordinate columns. Looked for x/y, xcenter/ycenter, "
        f"xcentroid/ycentroid; found {', '.join(table.colnames)}."
    )
