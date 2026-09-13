"""The subsets file: short names for whatever separates one set of flats
from another.

A header says the filter was `6 0`, because that is what the filter wheel
reported. Nothing downstream wants to be called `Flat6 0`, and nobody reading a
directory listing six months later will remember which wheel position was V.

So the first time harwood meets a subset it writes it down, with a short name
derived from it, in a file called `subsets` in the reduction directory:

    # subset            name
    '5 0'               s5
    '6 0'               s6

and you edit the right-hand column to say `B` and `V`. Everything that names a
file after a subset uses that column. IRAF does exactly this, and it is a good
idea for the same reason the instrument translation is: the convention belongs
to the reduction, written down, rather than to whoever happened to run it.

The word is IRAF's and worth keeping: a subset need not be a filter. For a
spectrograph it is the grating and its angle, and for an echelle the order --
anything that means two flats are not interchangeable.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

FILE = "subsets"

HEADER = """\
# What separates one set of flats from another, and what to call it.
#
# The left column is what the header says. The right column is the short name
# harwood uses when it has to name a file after a subset -- edit it to
# something you will recognise, such as B or V, and nothing else needs to
# change.
#
# Lines added automatically the first time each subset is seen.

"""


def path_for(workdir: Path | str = ".") -> Path:
    return Path(workdir) / FILE


def read(workdir: Path | str = ".") -> dict[str, str]:
    """The subsets known to this reduction: header value -> short name."""
    path = path_for(workdir)
    if not path.is_file():
        return {}

    known: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(),
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
                f"{path}:{number}: expected 'subset name', got {line!r}. The left "
                "column is what the header says, the right is what to call it."
            )
        known[parts[0]] = parts[1]
    return known


def suggest(subset: str, taken: set[str]) -> str:
    """A short name for a subset nobody has named yet.

    Derived from the header value rather than a counter, so `6 0` becomes `s6`
    and not `subset3`: a name with something of the original in it is one you
    can check, and the point is that you then replace it with `V`.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", subset) or "x"
    candidate = f"s{cleaned}"[:8]
    if candidate not in taken:
        return candidate
    index = 2
    while f"{candidate}{index}" in taken:
        index += 1
    return f"{candidate}{index}"


def name_for(subset: str, workdir: Path | str = ".", create: bool = True) -> str:
    """The short name for one subset, adding it to the file if it is new."""
    if not subset:
        return ""
    known = read(workdir)
    if subset in known:
        return known[subset]
    if not create:
        return suggest(subset, set(known.values()))

    name = suggest(subset, set(known.values()))
    known[subset] = name
    write(known, workdir)
    return name


def write(known: dict[str, str], workdir: Path | str = ".") -> Path:
    """Write the subsets file, keeping it readable and in a stable order."""
    path = path_for(workdir)
    lines = [HEADER.rstrip("\n"), "", f"# {'subset':<20}name"]
    for subset, name in sorted(known.items()):
        quoted = shlex.quote(subset)
        lines.append(f"{quoted:<22}{name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def record(subsets: list[str], workdir: Path | str = ".") -> dict[str, str]:
    """Make sure every subset given has a name, and return the whole mapping."""
    known = read(workdir)
    added = False
    for subset in subsets:
        if subset and subset not in known:
            known[subset] = suggest(subset, set(known.values()))
            added = True
    if added:
        write(known, workdir)
    return known
