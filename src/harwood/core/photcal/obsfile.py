"""obsfile: the stars of an image set, matched across its filters, as one
line of the observations file each.

From photcal/mkobsfile (t_obsfile.x, phimtable.x, phsort.x, phmatch.x). An
image set is a field observed once in each filter; the sets file names
them, one per line (`NAME : image1 image2 ...`, in the order of idfilters).
The magnitudes are read from text files by column number -- mkapfile's
best magnitudes, or phot's output dumped to text -- corrected for the
exposure time and an aperture correction, and each star of the first
filter's image is matched to the nearest star within the tolerance in the
others'. Each matched star is written as a line per filter, the second and
later starting with "*", which is how fitparams and invertfit read them.

What IRAF does is kept where it decides what is written: the matching
walks each image's stars in y order, by IRAF's quicksort, and takes the
nearest star not yet taken; a line with fewer columns than the image
column ends the search for the next image, so a file's comment lines give
one warning each, as IRAF gives them; and the correction is
2.5 log10(itime) + apercor in double, added in single precision.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ctor, strdic
from .io import words

f32 = np.float32
NAN = f32(np.nan)
INDEFR = f32(1.6e38)               # how IRAF sorts an undefined y: last
LOG10E = 0.43429448190325182765
DEF_LENLABEL = 15
DEF_LENINDEX = 6
FIRST_COLUMN = 3
DELTA_COLUMN = 2
CAT_NFIELDS = 10
(CAT_IMAGE, CAT_XCENTER, CAT_YCENTER, CAT_IFILTER, CAT_ITIME, CAT_XAIRMASS,
 CAT_OTIME, CAT_MAG, CAT_MERR, CAT_ID) = range(10)
OBS_NFIELDS = 5


@dataclass
class ImEntry:
    """IMT_*: an image of an image set."""
    setno: int
    imno: int
    name: str
    label: str
    offset: int = -1               # where its stars start; -1 none
    nentries: int = 0
    xshift: np.float32 = f32(0.0)
    yshift: np.float32 = f32(0.0)
    apercor: np.float32 = f32(0.0)
    itime: np.float32 = NAN
    xairmass: np.float32 = NAN
    otime: np.float32 = NAN
    ifilter: str = "INDEF"


class ImTable:
    """IRAF's symbol table: the entries in order, a name finding its latest."""

    def __init__(self):
        self.entries: list[ImEntry] = []
        self.byname: dict[str, ImEntry] = {}

    def enter(self, entry: ImEntry) -> None:
        self.entries.append(entry)
        self.byname[entry.name] = entry

    def find(self, name: str) -> ImEntry | None:
        return self.byname.get(name)


@dataclass
class Stars:
    objid: list[str] = field(default_factory=list)
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    mag: list = field(default_factory=list)
    merr: list = field(default_factory=list)
    imid: list = field(default_factory=list)
    id: list = field(default_factory=list)


def getlabels(text: str) -> list[str]:
    """ph_getlabels, over and over: the labels of a list separated by
    commas or blanks."""
    return [w for w in text.replace(",", " ").split() if w]


def columns(text: str) -> list[int]:
    """The incolumns string: a column number per field, 0 when not one."""
    out = []
    for label in getlabels(text):
        try:
            out.append(int(label))
        except ValueError:
            out.append(0)
    return out


def _real(token: str | None):
    """gargr: the value, or None when the token is not a number."""
    if token is None:
        return None
    value = ctor(token)
    if value is None:
        return None
    return NAN if value > 1e38 else f32(value)


def _fields(line: str, wanted: dict[int, str]) -> tuple[dict[int, object], bool]:
    """sscan of a line and a garg per column up to the last one wanted,
    by kind ('w' a word, 'r' a real): the values, and whether every garg
    succeeded -- the scan stops at the first that does not."""
    tokens = words(line)
    out: dict[int, object] = {}
    ok = True
    for col in range(1, max(wanted, default=0) + 1):
        token = tokens[col - 1] if col - 1 < len(tokens) else None
        kind = wanted.get(col)
        if not ok or token is None:
            ok = False
            continue
        if kind == "r":
            value = _real(token)
            if value is None:
                ok = False
                continue
            out[col] = value
        elif kind == "w":
            out[col] = token
    return out, ok


class _Reader:
    """A text file read line by line, as ph_nxtimage and ph_getimage do."""

    def __init__(self, path: str):
        self.lines = Path(path).read_text(errors="replace").split("\n")
        if self.lines and self.lines[-1] == "":
            self.lines.pop()
        self.i = 0

    def _word(self, line: str, col: int) -> tuple[str, bool]:
        tokens = words(line)
        if len(tokens) >= col:
            return tokens[col - 1], True
        return "", False

    def next_image(self, image: str, col: int):
        """ph_nxtimage: the next line whose image differs from `image`. A
        line with too few columns reads as image "" -- so after a named
        image, a blank line ends the search."""
        while self.i < len(self.lines):
            line = self.lines[self.i]
            self.i += 1
            word, _ = self._word(line, col)
            if word != image:
                return line, word
        return None

    def get_image(self, col: int):
        """ph_getimage: the next line with an image column."""
        while self.i < len(self.lines):
            line = self.lines[self.i]
            self.i += 1
            word, ok = self._word(line, col)
            if ok:
                return line, word
        return None


# ------------------------------------------------------------- the table


def read_sets(path: str, nfilters: int, notes: list[str], verbose: bool = True) -> ImTable:
    """ph_setimtable: `name : image image ...` per line."""
    table = ImTable()
    setno = nsymbols = 0
    for line in Path(path).read_text(errors="replace").split("\n"):
        tokens = words(line)
        if not tokens or tokens[0].startswith("#"):
            continue
        name = tokens[0]
        rest = line.lstrip()[len(name):] if not line.lstrip().startswith('"') else ""
        if name == ":":
            # IRAF means `: image ...` as a set without a name, numbered,
            # but its scan count falls one short there, and the line is
            # dropped without a word.
            continue
        else:
            stripped = rest.lstrip()
            colon = stripped[:1]
            images = words(stripped[1:]) if colon else []
        if colon != ":":
            if verbose:
                notes.append(f"Warning: Error decoding image set {name}")
            continue
        setno += 1
        if not name:
            name = str(setno)
        for image in images:
            nsymbols += 1
            if image == "INDEF":
                image = f"INDEF{nsymbols}"
            table.enter(ImEntry(setno, nsymbols, image, name))
        if not verbose:
            continue
        if not images:
            notes.append(f"Warning: Image set {setno} name {name} is empty")
        elif len(images) < nfilters:
            notes.append(f"Warning: Image set {setno} name  {name} is incomplete")
        elif len(images) > nfilters:
            notes.append(f"Warning: Image set {setno} name  {name} has more than "
                         f"{len(images)} images")
    return table


def read_shifts(table: ImTable, path: str, notes: list[str], verbose: bool = True) -> None:
    """ph_shimtable: `image xshift yshift` per line."""
    for line in Path(path).read_text(errors="replace").split("\n"):
        values, ok = _fields(line, {1: "w", 2: "r", 3: "r"})
        if not ok:
            continue
        entry = table.find(values[1])
        if entry is None:
            if verbose:
                notes.append(f"Warning: File: {path} image: {values[1]} is not in the "
                             "image sets file")
            continue
        entry.xshift, entry.yshift = values[2], values[3]


def read_apercors(table: ImTable, path: str, notes: list[str], verbose: bool = True) -> None:
    """ph_apimtable: `image apercor` per line, as mkapfile writes them."""
    for line in Path(path).read_text(errors="replace").split("\n"):
        values, ok = _fields(line, {1: "w", 2: "r"})
        if not ok:
            continue
        entry = table.find(values[1])
        if entry is None:
            if verbose:
                notes.append(f"Warning: File: {path} image: {values[1]} is not in the "
                             "image sets file")
            continue
        entry.apercor = values[2]


def obs_columns(text: str) -> list[int]:
    """The obscolumns string: the image is column 1, then filter, exposure
    time, airmass and time; a 1 among them is skipped."""
    cols = [1] + [0] * (OBS_NFIELDS - 1)
    n = 1
    for label in getlabels(text):
        if n >= OBS_NFIELDS:
            break
        try:
            cols[n] = int(label)
        except ValueError:
            cols[n] = 0
        else:
            if cols[n] == 1:
                continue
        n += 1
    return cols


def _imdata(line: str, fcol: int, icol: int, acol: int, ocol: int):
    """ph_imdata / ph_obsdata: filter, exposure time, airmass and time."""
    wanted = {c: k for c, k in ((icol, "r"), (acol, "r"), (ocol, "r"), (fcol, "w")) if c > 0}
    tokens = words(line)
    out = {"f": "INDEF", "i": NAN, "a": NAN, "o": NAN}
    ok = True
    for col in range(1, max(wanted, default=0) + 1):
        token = tokens[col - 1] if col - 1 < len(tokens) else None
        if token is None:
            ok = False
        if not ok:
            continue
        if col == fcol:
            out["f"] = token
        elif col in (icol, acol, ocol):
            value = _real(token)
            if value is None:
                ok = False
                continue
            out["i" if col == icol else "a" if col == acol else "o"] = value
    return out["f"], out["i"], out["a"], out["o"]


def read_obsparams(table: ImTable, path: str, cols: list[int], notes: list[str],
                   verbose: bool = True) -> None:
    """ph_obsimtable: an image's filter, exposure time, airmass and time
    from a file of their own, replacing what the photometry says."""
    reader = _Reader(path)
    image = ""
    while (found := reader.next_image(image, cols[0])) is not None:
        line, image = found
        entry = table.find(image)
        if entry is None:
            if verbose:
                notes.append(f"Warning: File: {path} image: {image} is not in the "
                             "image sets file")
            continue
        ifilter, itime, airmass, otime = _imdata(line, cols[1], cols[2], cols[3], cols[4])
        if not np.isnan(itime):
            entry.itime = itime
        if not np.isnan(airmass):
            entry.xairmass = airmass
        if not np.isnan(otime):
            entry.otime = otime
        if ifilter != "INDEF":
            entry.ifilter = ifilter


def read_photometry(table: ImTable, path: str, cols: list[int], stars: Stars,
                    normtime: bool, notes: list[str], verbose: bool = True) -> bool:
    """ph_mkimtable: each image's stars from one file; True when an image's
    stars are not all together, and have to be sorted."""
    sortimid = False
    reader = _Reader(path)
    imcol = cols[CAT_IMAGE]
    found = reader.next_image("", imcol)
    while found is not None:
        line, image = found
        entry = table.find(image)
        if entry is None:
            if image and verbose:
                notes.append(f"Warning: File: {path} Image: {image} is not in the image "
                             "sets file")
            found = reader.next_image(image, imcol)
            continue
        ifilter, itime, airmass, otime = _imdata(
            line, cols[CAT_IFILTER], cols[CAT_ITIME], cols[CAT_XAIRMASS], cols[CAT_OTIME])
        if np.isnan(entry.itime):
            entry.itime = f32(1.0) if np.isnan(itime) or not normtime else itime
        elif entry.nentries > 0:
            pass
        elif np.isnan(itime):
            if not normtime:
                entry.itime = f32(1.0)
        else:
            entry.itime = f32(entry.itime / itime)
        if np.isnan(entry.xairmass):
            entry.xairmass = airmass
        if np.isnan(entry.otime):
            entry.otime = otime
        if entry.ifilter == "INDEF":
            entry.ifilter = ifilter
        if entry.nentries > 0:
            sortimid = True
        else:
            entry.offset = len(stars.x)
        while True:
            star = _stardata(line, cols)
            if star is not None:
                objid, x, y, mag, merr = star
                stars.objid.append(objid)
                stars.x.append(x)
                stars.y.append(y)
                stars.mag.append(mag)
                stars.merr.append(merr)
                stars.imid.append(entry.imno)
                entry.nentries += 1
                stars.id.append(entry.nentries)
            nxt = reader.get_image(imcol)
            if nxt is None:
                found = None
                break
            line, imname = nxt
            if imname != image:
                found = nxt
                break
    return sortimid


def _stardata(line: str, cols: list[int]):
    """ph_stardata: id, x, y, mag and merr; None when a column is missing."""
    icol, xcol, ycol = cols[CAT_ID], cols[CAT_XCENTER], cols[CAT_YCENTER]
    mcol, ecol = cols[CAT_MAG], cols[CAT_MERR]
    tokens = words(line)
    values = {"x": NAN, "y": NAN, "m": NAN, "e": NAN}
    objid = ""
    for col in range(1, max(icol, xcol, ycol, mcol, ecol) + 1):
        if col - 1 >= len(tokens):
            return None
        token = tokens[col - 1]
        if col == icol:
            objid = token[:DEF_LENLABEL]
        elif col in (xcol, ycol, mcol, ecol):
            value = _real(token)
            if value is None:
                return None
            key = "x" if col == xcol else "y" if col == ycol else "m" if col == mcol else "e"
            values[key] = value
    return objid, values["x"], values["y"], values["m"], values["e"]


# ---------------------------------------------------------------- sorting


def qsort(data: list, n: int, offset: int) -> list[int]:
    """ph_qsort: the indices of data[offset:offset+n] in increasing order,
    by IRAF's quicksort -- the last element the pivot, so equal values come
    out in its order, not the file's."""
    def key(i):
        v = data[offset + i]
        return INDEFR if np.isnan(v) else v

    index = list(range(n))
    lv, uv = [0] * 40, [0] * 40
    p = 0
    lv[0], uv[0] = 0, n - 1
    while p >= 0:
        if lv[p] >= uv[p]:
            p -= 1
            continue
        i = lv[p] - 1
        j = uv[p]
        pivot = key(index[j])
        while i < j:
            i += 1
            while key(index[i]) < pivot:
                i += 1
            j -= 1
            while j > i:
                if key(index[j]) <= pivot:
                    break
                j -= 1
            if i < j:
                index[i], index[j] = index[j], index[i]
        j = uv[p]
        index[i], index[j] = index[j], index[i]
        if i - lv[p] < uv[p] - i:
            lv[p + 1], uv[p + 1] = lv[p], i - 1
            lv[p] = i + 1
        else:
            lv[p + 1], uv[p + 1] = i + 1, uv[p]
            uv[p] = i - 1
        p += 1
    return [offset + i for i in index]


def sort_stars(table: ImTable, stars: Stars, sortimid: bool, tolerance: float) -> list[int]:
    """ph_sort: the stars grouped by image when they were not, and each
    image's stars in y order for the matching."""
    npts = len(stars.x)
    if sortimid:
        order = sorted(range(npts), key=lambda k: (stars.imid[k], stars.id[k]))
        for name in ("objid", "x", "y", "mag", "merr", "imid", "id"):
            values = getattr(stars, name)
            setattr(stars, name, [values[k] for k in order])
        nptr = npts
        for entry in reversed(table.entries):
            if entry.nentries <= 0:
                continue
            nptr -= entry.nentries
            entry.offset = nptr
    ysort = list(stars.id)
    if tolerance > 0:
        for entry in table.entries:
            if entry.nentries <= 0:
                continue
            ysort[entry.offset:entry.offset + entry.nentries] = qsort(
                stars.y, entry.nentries, entry.offset)
    return ysort


def correct(table: ImTable, stars: Stars, minmagerr: float) -> None:
    """ph_correct: the shifts added, the magnitudes corrected for the
    exposure time and the aperture, the errors given a floor."""
    floor = f32(minmagerr)
    for entry in table.entries:
        if entry.nentries <= 0 or table.find(entry.name) is not entry:
            continue
        itime = float(entry.itime)
        magshift = (f32(2.5 * (LOG10E * math.log(itime)) + float(entry.apercor))
                    if itime > 0 else NAN)
        for i in range(entry.offset, entry.offset + entry.nentries):
            stars.x[i] = f32(stars.x[i] + entry.xshift)
            stars.y[i] = f32(stars.y[i] + entry.yshift)
            if np.isnan(stars.mag[i]):
                continue
            stars.mag[i] = f32(stars.mag[i] + magshift)
            if np.isnan(stars.merr[i]):
                continue
            if stars.merr[i] < floor:
                stars.merr[i] = floor


# ---------------------------------------------------------------- writing


def _f(value, width: int, decpl: int) -> str:
    from .apfile import _f as f
    return f(value, width, decpl)


def _otime(value) -> str:
    from .apfile import _hours

    if np.isnan(value):
        return "INDEF".rjust(11)
    if 0.0 <= value <= 24.0:
        return _hours(value).rjust(11)
    from .. import iraffmt
    return iraffmt.g(float(value)).rjust(11)


def _star_line(ifilter: str, otime, airmass, x, y, mag, merr) -> str:
    return (f"{ifilter[:10]:<10} {_otime(otime)} {_f(airmass, 6, 3)} {_f(x, 9, 3)} "
            f"{_f(y, 9, 3)} {_f(mag, 7, 3)} {_f(merr, 6, 3)}")


def _label(text: str, width: int) -> str:
    return f"{text[:width]:<{width}} "


def _tab(text: str, column: int) -> str:
    """%Nt: on to column N (1-based)."""
    return text.ljust(column - 1)


def header(nfilters: int, wrap: bool) -> str:
    first = ""
    for col, word in ((1, "# FIELD"), (17, "FILTER"), (34, "OTIME"), (40, "AIRMASS"),
                      (49, "XCENTER"), (59, "YCENTER"), (71, "MAG"), (77, "MERR")):
        first = _tab(first, col) + word
    if wrap:
        return f"\n{first}\n\n"
    line = first + " "
    for _ in range(nfilters - 1):
        part = ""
        for col, word in ((2, "FILTER"), (19, "OTIME"), (25, "AIRMASS"), (34, "XCENTER"),
                          (44, "YCENTER"), (56, "MAG"), (62, "MERR")):
            part = _tab(part, col) + word
        line += part + " "
    return f"\n{line}\n\n"


@dataclass
class Result:
    written: list[tuple[str, int]]       # each set's label and its stars
    notes: list[str]


def _entry_line(entry: ImEntry, label: str, k: int, stars: Stars) -> str:
    ifilter = label if entry.ifilter == "INDEF" else entry.ifilter
    return _star_line(ifilter, entry.otime, entry.xairmass,
                      f32(stars.x[k] - entry.xshift), f32(stars.y[k] - entry.yshift),
                      stars.mag[k], stars.merr[k])


def _indef_line(label: str) -> str:
    return _star_line(label, NAN, NAN, NAN, NAN, NAN, NAN)


def merge(out: list[str], label: str, syms: list[ImEntry | None], filters: list[str],
          stars: Stars, ysort: list[int], matched: list[bool], ndata: int,
          tolerance: float, allfilters: bool, wrap: bool, useid: bool) -> int:
    """ph_merge: the stars matched by position across the set's filters."""
    nfilters = len(filters)
    nframe = 0
    for sym in syms:
        if sym is not None:
            for k in range(sym.offset, sym.offset + sym.nentries):
                matched[k] = False
            if sym.nentries > 0:
                nframe += 1
    tol = f32(tolerance)
    tol2 = f32(tol * tol)
    nobjs = 0
    ntimes = 1 if allfilters else nfilters
    for i in range(ntimes):
        sym_i = syms[i]
        if sym_i is None or sym_i.offset < 0:
            continue
        bi, ei = sym_i.offset, sym_i.offset + sym_i.nentries
        nmatch = 0
        index = [[-1] * nfilters for _ in range(ndata)]
        for j in range(i + 1, nfilters):
            sym_j = syms[j]
            if sym_j is None or sym_j.offset < 0:
                continue
            bj, ej = sym_j.offset, sym_j.offset + sym_j.nentries
            nmatch = 0
            for kk in range(bi, ei):
                k = ysort[kk]
                if matched[k] or np.isnan(stars.x[k]) or np.isnan(stars.y[k]):
                    continue
                nmatch += 1
                index[k - bi][i] = k
                maxrsq = tol2
                lsort = -1
                for ll in range(bj, ej):
                    m = ysort[ll]
                    if matched[m] or np.isnan(stars.x[m]) or np.isnan(stars.y[m]):
                        continue
                    dy = f32(stars.y[m] - stars.y[k])
                    if dy > tol:
                        break
                    dx = f32(stars.x[m] - stars.x[k])
                    distsq = f32(f32(dx * dx) + f32(dy * dy))
                    if distsq > maxrsq:
                        continue
                    if lsort >= 0:
                        matched[ysort[lsort]] = False
                    matched[m] = True
                    index[k - bi][j] = m
                    maxrsq = distsq
                    lsort = ll
            if nmatch == 0:
                break
        if nframe == 1 or (not allfilters and i == nfilters - 1):
            for kk in range(bi, ei):
                k = ysort[kk]
                if matched[k] or np.isnan(stars.x[k]) or np.isnan(stars.y[k]):
                    continue
                nmatch += 1
                index[k - bi][i] = k
        for j in range(ei - bi):
            if nmatch <= 0:
                break
            row = index[j]
            starok = (all(v >= 0 for v in row) if allfilters
                      else any(v >= 0 for v in row))
            if not starok:
                continue
            if useid:
                width = DEF_LENLABEL
            elif nobjs == 1:
                width = max(DEF_LENLABEL, len(label))
            else:
                width = max(DEF_LENLABEL, len(label) + DEF_LENINDEX)
            idlabel = ""
            if useid:
                idlabel = next((stars.objid[v] for v in row if v >= 0), "")
            line = ""
            for k in range(nfilters):
                if k == 0:
                    if useid and idlabel:
                        line += _label(idlabel, width)
                    elif nmatch == 1 and nobjs == 0:
                        line += _label(label, width)
                    else:
                        line += _label(f"{label}-{nobjs + 1}", width)
                elif wrap:
                    line += _label("*", width)
                else:
                    line += "  "
                if row[k] < 0:
                    line += _indef_line(filters[k])
                else:
                    line += _entry_line(syms[k], filters[k], row[k], stars)
                if wrap or k == nfilters - 1:
                    out.append(line + "\n")
                    line = ""
            nobjs += 1
    return nobjs


def join(out: list[str], label: str, syms: list[ImEntry | None], filters: list[str],
         stars: Stars, allfilters: bool, wrap: bool, useid: bool) -> int:
    """ph_join: with no tolerance, the n-th star of each filter together."""
    nfilters = len(filters)
    if allfilters:
        nobjs = min((s.nentries if s is not None else 0) for s in syms) if syms else 0
    else:
        nobjs = max([s.nentries for s in syms if s is not None], default=0)
    if nobjs <= 0:
        return 0
    if useid:
        width = DEF_LENLABEL
    elif nobjs == 1:
        width = max(DEF_LENLABEL, len(label))
    else:
        width = max(DEF_LENLABEL, len(label) + DEF_LENINDEX)
    begin = [s.offset if s is not None and s.nentries > 0 else -1 for s in syms]
    end = [s.offset + s.nentries - 1 if s is not None and s.nentries > 0 else -1
           for s in syms]
    for i in range(1, nobjs + 1):
        idlabel = ""
        if useid:
            idlabel = next((stars.objid[b] for b in begin if b >= 0), "")
        line = ""
        for j in range(nfilters):
            if j == 0:
                if useid and idlabel:
                    line += _label(idlabel, width)
                elif nobjs == 1:
                    line += _label(label, width)
                else:
                    line += _label(f"{label}-{i}", width)
            elif wrap:
                line += _label("*", width)
            else:
                line += "  "
            dptr = begin[j]
            if dptr < 0:
                line += _indef_line(filters[j])
            else:
                line += _entry_line(syms[j], filters[j], dptr, stars)
            if wrap or j == nfilters - 1:
                out.append(line + "\n")
                line = ""
            if dptr < 0:
                continue
            begin[j] = -1 if dptr >= end[j] else dptr + 1
    return nobjs


def format_file(filters: list[str]) -> str:
    """ph_format: the observations section of a configuration, for this
    file's columns."""
    lines = ["# Declare the observations file variables", "", "observations", ""]
    col = FIRST_COLUMN
    for f in filters:
        for name, what in ((f"T{f}", "time of observation"), (f"X{f}", "airmass"),
                           (f"x{f}", "x coordinate"), (f"y{f}", "y coordinate"),
                           (f"m{f}", "instrumental magnitude"),
                           (f"error(m{f})", "magnitude error")):
            text = _tab(_tab(name, 15) + str(col), 30)
            lines.append(f"{text}# {what} in filter {f}")
            col += 1
        col -= 1
        lines.append("")
        col += DELTA_COLUMN
    return "\n".join(lines) + "\n"


def format_path(observations: str | Path) -> Path:
    """IRAF writes f<observations>.dat beside it."""
    path = Path(observations)
    return path.with_name(f"f{path.name}.dat")


def obsfile(photfiles: list[str], incolumns: str, idfilters: str, imsets: str,
            observations: str, wrap: bool = True, obsparams: str = "",
            obscolumns: str = "2 3 4 5", minmagerr: float = 0.001, shifts: str = "",
            apercors: str = "", normtime: bool = False, tolerance: float = 5.0,
            allfilters: bool = True, verbose: bool = True) -> Result:
    """t_obsfile, without its interactive entry of the sets."""
    cols = columns(incolumns)
    if len(cols) < CAT_NFIELDS:
        raise ValueError("There are too few columns defined in the incolumns string")
    filters = getlabels(idfilters)
    notes: list[str] = []
    table = read_sets(imsets, len(filters), notes, verbose)
    if obsparams:
        read_obsparams(table, obsparams, obs_columns(obscolumns), notes, verbose)
    if shifts:
        read_shifts(table, shifts, notes, verbose)
    if apercors:
        read_apercors(table, apercors, notes, verbose)
    stars = Stars()
    sortimid = False
    for path in photfiles:
        sortimid = read_photometry(table, path, cols, stars, normtime, notes,
                                   verbose) or sortimid
    useid = cols[CAT_ID] > 0
    ysort = sort_stars(table, stars, sortimid, tolerance)
    correct(table, stars, minmagerr)

    out = [header(len(filters), wrap)]
    written: list[tuple[str, int]] = []
    matched = [False] * len(stars.x)
    entries = table.entries
    nsets = max((e.setno for e in entries), default=0)
    ptr = 0
    for setno in range(1, nsets + 1):
        if ptr >= len(entries):
            break
        syms: list[ImEntry | None] = [None] * len(filters)
        label = entries[ptr].label
        ndata = 0
        while ptr < len(entries) and entries[ptr].setno == setno:
            entry = entries[ptr]
            latest = table.find(entry.name)
            if latest is not entry:
                entry.offset, entry.nentries = latest.offset, latest.nentries
                entry.xairmass, entry.otime = latest.xairmass, latest.otime
                entry.ifilter = latest.ifilter
            filterno, _ = strdic(entry.ifilter, filters)
            if entry.ifilter == "INDEF":
                filterno = next((k + 1 for k in range(len(filters)) if syms[k] is None),
                                filterno)
            if filterno <= 0:
                if verbose:
                    notes.append(f"Warning: Image set: {setno}  label: {entry.label}  "
                                 f"image: {entry.name}")
                    notes.append(f"\tWarning: Filter {entry.ifilter} does not belong to "
                                 f"filter set ,{','.join(filters)}")
            elif syms[filterno - 1] is not None:
                if verbose:
                    notes.append(f"Warning: Image set: {setno}  label: {entry.label}  "
                                 f"image: {entry.name}")
                    notes.append(f"\tData for filter {filters[filterno - 1]} is redundant")
            else:
                syms[filterno - 1] = entry
                ndata = max(ndata, entry.nentries)
            ptr += 1
        if ndata <= 0:
            if verbose:
                notes.append(f"Image set: {setno}  Label: {label} has no data")
            continue
        if tolerance <= 0.0:
            n = join(out, label, syms, filters, stars, allfilters, wrap, useid)
        else:
            n = merge(out, label, syms, filters, stars, ysort, matched, ndata, tolerance,
                      allfilters, wrap, useid)
        written.append((label, n))
    Path(observations).write_text("".join(out))
    format_path(observations).write_text(format_file(filters))
    return Result(written, notes)


def summary(observations: str, written: list[tuple[str, int]]) -> str:
    """What IRAF prints on the standard output when verbose."""
    lines = ["", f"Observations file: {observations}"]
    lines += [f"\tImage set: {label}  {n} stars written to the observations file"
              for label, n in written]
    return "\n".join(lines) + "\n\n"
