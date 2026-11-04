"""The photcal tables typed at the terminal, as IRAF asks for them.

obsfile, mknobsfile and mkobsfile take their image sets, shifts, aperture
corrections and observing parameters from a file -- or, given STDIN, ask for
them image by image (phimtable.x's ph_esetimtable, ph_eshimtable,
ph_eapimtable, ph_eobsimtable), and mkapfile its observing parameters
(phaimtable.x's ph_eaobsimtable). Each prompt is IRAF's, each line is read as
its scan() reads it, and each answer that a verify prompt follows can be
changed there.

A Terminal is where the prompts go and the lines come from: a file of answers,
the keyboard, or a question in the window; what it was asked is kept, so that it
can be read back as IRAF's standard output.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

import numpy as np

from .. import iraffmt
from ..iraffmt import ctod

f32 = np.float32
NAN = f32(np.nan)
INDEFR = 1.6e38
_NUMBER = re.compile(r"\s*(INDEF|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?"
                     r"(?::\d+(?:\.\d*)?)*)")


class Terminal:
    """IRAF's STDIN and STDOUT for a task that asks. `lines` are what is typed, in
    order, or `read(prompt)` asks for each; None from either is the end of the input,
    which stays the end, as a file's does."""

    def __init__(self, lines: Iterator[str] | None = None,
                 read: Callable[[str], str | None] | None = None, echo=None):
        self.lines, self.read, self.echo = lines, read, echo
        self.out: list[str] = []
        self.ended = False
        self._prompt = ""

    def printf(self, text: str) -> None:
        """What IRAF prints: kept; shown at once on a keyboard's screen (`echo`); and,
        what came since the last line read, handed to `read` with the next question."""
        self.out.append(text)
        self._prompt += text
        if self.echo is not None:
            self.echo(text)

    def scan(self) -> Scan | None:
        """scan(): the next line, or None at the end of the input."""
        if self.ended:
            return None
        if self.lines is not None:
            line = next(self.lines, None)
        elif self.read is not None:
            line = self.read(self._prompt)
        else:
            line = None
        self._prompt = ""
        if line is None:
            self.ended = True
            return None
        return Scan(line)

    @property
    def text(self) -> str:
        return "".join(self.out)


class Scan:
    """sscan and the garg calls: one field after another, nscan() counting those
    read; the first that cannot be read stops the count there."""

    def __init__(self, line: str):
        self.line = line.rstrip("\n")
        self.reset()

    def reset(self) -> None:
        """reset_scan."""
        self.pos = 0
        self.n = 0
        self.failed = False

    def wrd(self) -> str | None:
        if self.failed:
            return None
        rest = self.line[self.pos:]
        stripped = rest.lstrip()
        if not stripped:
            self.failed = True
            return None
        word = stripped.split()[0]
        self.pos += len(rest) - len(stripped) + len(word)
        self.n += 1
        return word

    def r(self) -> float | None:
        """gargr: a real, INDEF as None."""
        if self.failed:
            return None
        found = _NUMBER.match(self.line, self.pos)
        if found is None:
            self.failed = True
            return None
        self.pos = found.end()
        self.n += 1
        value = ctod(found.group(1))
        return None if value is None or abs(value) >= 1e30 else float(f32(value))

    def c(self) -> str | None:
        """gargc: the next character."""
        if self.failed or self.pos >= len(self.line):
            self.failed = True
            return None
        char = self.line[self.pos]
        self.pos += 1
        self.n += 1
        return char


def _g(value) -> str:
    """%g of a real, INDEF for one not given."""
    if value is None or (isinstance(value, float | np.floating) and np.isnan(value)):
        return "INDEF"
    return iraffmt.g(float(value))


def _h(value) -> str:
    """%0.1h of a real: hours as h:mm:ss.s."""
    if value is None or np.isnan(value):
        return "INDEF"
    from .apfile import _hours

    return _hours(float(value)).strip()


def _indef(value) -> bool:
    return value is None or (isinstance(value, float | np.floating) and np.isnan(value))


def enter_sets(term: Terminal, nfilters: int, verify: bool):
    """ph_esetimtable: the image sets, typed: a name, then an image per filter."""
    from .obsfile import ImEntry, ImTable

    table = ImTable()
    setno = nsymbols = 0
    field = 1
    name = ""
    term.printf("\n")
    while True:
        if field == 1:
            term.printf(f"Enter name of image set {setno + 1} (name, <EOF>=quit entry): ")
            got = term.scan()
            if got is None:
                term.printf("\n")
                break
            word = got.wrd()
            if got.n != 1:
                continue
            name = word
            if verify:
                term.printf(f"    Verify name (name, <CR>={name}): ")
                got = term.scan()
                if got is None:
                    term.printf("\n")
                else:
                    word = got.wrd()
                    if got.n == 1:
                        name = word
            field += 1
            setno += 1
            continue
        if field <= nfilters + 1:
            term.printf(f"    Enter image name {field - 1} (name, <CR>=INDEF): ")
            got = term.scan()
            if got is None:
                term.printf("\n")
                continue
            image = got.wrd()
            if got.n != 1:
                image = "INDEF"
            if verify:
                term.printf(f"        Verify name (name,  <CR>={image}): ")
                got = term.scan()
                if got is None:
                    term.printf("\n")
                else:
                    word = got.wrd()
                    if got.n == 1:
                        image = word
            nsymbols += 1
            field += 1
            if image == "INDEF":
                image = f"INDEF{nsymbols}"
            table.enter(ImEntry(setno, nsymbols, image, name))
            continue
        field = 1
    term.printf("\n")
    return table


def _by_set(term: Terminal, table, what: str, separator: str = "  "):
    """The images in order, each set's prompt said before its first image."""
    osetno = 0
    for entry in table.entries:
        if entry.setno != osetno:
            term.printf(f"Image set {entry.setno} ({entry.label}):{separator}{what}\n")
        osetno = entry.setno
        yield entry


def _pair(got: Scan) -> tuple[float, float, int]:
    """x and y, or x,y."""
    x, y = got.r(), got.r()
    if got.n == 2:
        return x, y, 2
    if got.n == 1:
        got.reset()
        x = got.r()
        comma = got.c()
        y = got.r()
        if got.n == 3 and comma == ",":
            return x, y, 3
        return x, None, 1
    return None, None, 0


def enter_shifts(term: Terminal, table, verify: bool) -> None:
    """ph_eshimtable: each image's shift in x and y, typed."""
    term.printf("\n")
    for entry in _by_set(term, table, "Enter the shift in x and y"):
        term.printf(f"    Image {entry.name} (xshift  yshift, <CR>=0.0 0.0, "
                    "<EOF>=quit entry): ")
        got = term.scan()
        if got is None:
            term.printf("\n")
            break
        x, y, n = _pair(got)
        if n == 0:
            x = y = 0.0
        elif n == 1:
            y = 0.0
        entry.xshift, entry.yshift = f32(x), f32(y)
        if not verify:
            continue
        term.printf(f"        Verify (xshift yshift, <CR>={_g(entry.xshift)} "
                    f"{_g(entry.yshift)}): ")
        got = term.scan()
        if got is None:
            term.printf("\n")
            continue
        x, y, n = _pair(got)
        if n >= 2:
            entry.xshift, entry.yshift = f32(x), f32(y)
        elif n == 1:
            entry.xshift = f32(x)
    term.printf("\n")


def enter_apercors(term: Terminal, table, verify: bool) -> None:
    """ph_eapimtable: each image's aperture correction, typed."""
    term.printf("\n")
    for entry in _by_set(term, table, "Enter the aperture correction"):
        term.printf(f"    Image {entry.name} (magnitude, <CR>=0.0, <EOF>=quit entry): ")
        got = term.scan()
        if got is None:
            term.printf("\n")
            break
        value = got.r()
        entry.apercor = f32(0.0 if got.n < 1 or value is None else value)
        if not verify:
            continue
        term.printf(f"        Verify (magnitude, <CR>={_g(entry.apercor)}): ")
        got = term.scan()
        if got is None:
            term.printf("\n")
            continue
        value = got.r()
        if got.n == 1 and value is not None:
            entry.apercor = f32(value)
    term.printf("\n")


def _four(got: Scan):
    """filter id, exposure time, airmass, time: what is not given, INDEF."""
    fields = [got.wrd(), got.r(), got.r(), got.r()]
    n = got.n
    return fields, n


def enter_obsparams(term: Terminal, table, verify: bool) -> None:
    """ph_eobsimtable: each image's filter, exposure time, airmass and time, typed."""
    term.printf("\n")
    for entry in _by_set(term, table,
                         "Enter filter id, exposure time, airmass, observation time",
                         separator=" "):
        term.printf(f"    Image {entry.name} (f t X T, <CR>=4 X (INDEF), "
                    "<EOF>=quit entry): ")
        got = term.scan()
        if got is None:
            term.printf("\n")
            break
        (ifilter, itime, xairmass, otime), n = _four(got)
        if n < 1:
            ifilter = "INDEF"
        if n < 2:
            itime = None
        if n < 3:
            xairmass = None
        if n < 4:
            otime = None
        if entry.ifilter.startswith("INDEF"):
            entry.ifilter = ifilter
        if itime is not None:
            entry.itime = f32(itime)
        if xairmass is not None:
            entry.xairmass = f32(xairmass)
        if otime is not None:
            entry.otime = f32(otime)
        if not verify:
            continue
        term.printf(f"        Verify (f t X T, <CR>={entry.ifilter} {_g(entry.itime)} "
                    f"{_g(entry.xairmass)} {_h(entry.otime)}): ")
        got = term.scan()
        if got is None:
            term.printf("\n")
            continue
        (ifilter, itime, xairmass, otime), n = _four(got)
        if n >= 1:
            entry.ifilter = ifilter
        if n >= 2:
            entry.itime = NAN if itime is None else f32(itime)
        if n >= 3:
            entry.xairmass = NAN if xairmass is None else f32(xairmass)
        if n >= 4:
            entry.otime = NAN if otime is None else f32(otime)
    term.printf("\n")


def enter_apfile_obsparams(term: Terminal, images, verify: bool,
                           notes: list[str] | None = None) -> None:
    """ph_eaobsimtable: mkapfile's images' filter, exposure time, airmass and time,
    typed; an airmass not given is 1.25, and said."""
    from .apfile import DEF_AIROFFSET

    term.printf("\n")
    term.printf("Enter filterid, exposure time and airmass\n")
    for image in images:
        term.printf(f"    Image {image.name} (f t X T, <CR>=4 X INDEF, <EOF>=quit entry): ")
        got = term.scan()
        if got is None:
            term.printf("\n")
            break
        (ifilter, itime, xairmass, otime), n = _four(got)
        if n < 1:
            ifilter = "INDEF"
        if n < 2:
            itime = None
        if n < 3:
            xairmass = None
        if n < 4:
            otime = None
        if not ifilter.startswith("INDEF"):
            image.ifilter = ifilter
        if itime is not None:
            image.itime = float(itime)
        if xairmass is not None:
            image.xairmass = f32(xairmass)
            image.nxairmass = f32(f32(xairmass) - DEF_AIROFFSET)
        else:
            image.xairmass = DEF_AIROFFSET
            image.nxairmass = f32(0.0)
            term.printf(f"    Warning: Setting airmass for image {image.name} to "
                        f"{iraffmt.g(float(DEF_AIROFFSET))}\n")
        if otime is not None:
            image.otime = f32(otime)
        if not verify:
            continue
        term.printf(f"        Verify (f t X T, <CR>={image.ifilter} {_g(image.itime)} "
                    f"{_g(image.xairmass)} {_h(image.otime)}): ")
        got = term.scan()
        if got is None:
            term.printf("\n")
            continue
        # IRAF reads three fields here: the fourth is pargr'd, not gargr'd, so a time
        # typed at the verify prompt is never taken.
        ifilter, itime, xairmass = got.wrd(), got.r(), got.r()
        n = got.n
        if n >= 1:
            image.ifilter = ifilter
        if n >= 2:
            image.itime = float("nan") if itime is None else float(itime)
        if n >= 3:
            if xairmass is not None:
                image.xairmass = f32(xairmass)
                image.nxairmass = f32(f32(xairmass) - DEF_AIROFFSET)
            else:
                image.xairmass = DEF_AIROFFSET
                image.nxairmass = f32(0.0)
                term.printf(f"    Warning: Setting airmass for image {image.name} to "
                            f"{iraffmt.g(float(DEF_AIROFFSET))}\n")
    term.printf("\n")
