"""graph -- plot's graphing utility: lists of y or x,y, and image sections reduced
to a vector, on one pair of axes.

From pkg/plot/t_graph.x, with what it calls: im_projection (improject.x), plt_wcs
(pltwcs.x), init_marker, hgpline, and from sys/gio the parts that decide the window
-- gascale, grscale, gswind and glabax's glb_setup (fp_nondegenr, the switch to
elog when a log axis reaches zero, glb_find_ticks and gtickr, which round the
window out to the next major tick when `round` is set and always on a log axis).
Arithmetic is IRAF's: single precision throughout, the projection summed line by
line, the image's world coordinates through mw_ctranr's real copy of the compiled
transform.

What IRAF does and harwood keeps:

* A list's first line with numbers on it decides whether it is y alone or x,y; a
  later line with one number in an x,y list is left out ("only 1 arg"), a line
  with none too ("no args"), and so is one whose comment does not begin in the
  first column. Numbers are read as IRAF reads them -- h:m:s, d exponents, INDEF
  (a break in the line) -- and a comma ends them: "10,7" is one number.
* An operand that is not a text file is an image, so a list that is not there is
  reported as an image that cannot be opened, and the rest are still plotted.
* The title is the first image's, with its section on a second line; the x label
  the first image's that has one; the x format the last image's.
* The window is the data's, through gascale and grscale, unless wx1 != wx2 (or
  wy1 != wy2); an H or M format multiplies the window given by 15. Without the box
  glabax never runs: a window with no extent is left so and nothing is rounded.
* bhist's bottom is the window's wy1, after glabax has rounded it.
* Line types and colours take the next number of their lists, the last one when
  the list is over, and wrap at 4 and 9.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research
in Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import glob
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

f32 = np.float32
MAX_CURVES = 20
SZ_BUF = 2048
EPSILONR = f32(1.192e-7)
LOG10E = 0.43429448190325182765

#: init_marker's names; "line", "lhist" and "bhist" are boxes when marking.
MARKERS = ("point", "box", "plus", "cross", "circle", "hebar", "vebar", "hline",
           "vline", "diamond")
#: hgpline's plot types; any other marker is a connected line.
PLOT_TYPES = ("line", "lhist", "bhist")
TRANSFORMS = ("linear", "log", "elog")


class GraphError(RuntimeError):
    """What stops IRAF's graph: its error message."""


@dataclass
class Curve:
    x: np.ndarray
    y: np.ndarray
    size: np.ndarray | None = None
    ltype: int = 1
    color: int = 1


@dataclass
class Stroke:
    """What GIO is told to draw, in world coordinates: a polyline (gamove to the first
    point, gadraw to the rest; NaN is INDEF, a break), a set of points (gpmark of
    point marks: a polymarker), or a mark (gmark) of a kind and size."""

    kind: str                       # "line", "points" or the mark's name
    x: np.ndarray
    y: np.ndarray
    ltype: int = 1
    color: int = 1
    xsize: float = 0.0
    ysize: float = 0.0


@dataclass
class Plot:
    curves: list[Curve]
    window: tuple[float, float, float, float]
    xtran: str = "linear"
    ytran: str = "linear"
    viewport: tuple[float, float, float, float] | None = None
    aspect: bool = False
    box: bool = True
    ticklabels: bool = True
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    xformat: str = ""
    yformat: str = ""
    pointmode: bool = False
    marker: str = "box"
    mark: str = "box"
    rdmarks: bool = False
    szmarker: float = 0.0
    err: str = ""
    strokes: list[Stroke] = field(default_factory=list)

    @property
    def title_lines(self) -> list[str]:
        """glabax's title block: a line for each newline that something follows."""
        if not self.title:
            return []
        text = self.title[:-1] if self.title.endswith("\n") else self.title
        return text.split("\n")


# ------------------------------------------------------------- numbers


def fp_normr(x):
    tol = f32(EPSILONR * f32(10))
    absx, expon = abs(f32(x)), 0
    if absx > 0:
        while absx < f32(f32(1) - tol):
            absx = f32(absx * f32(10))
            expon -= 1
            if absx == 0:
                return f32(0), 0
        while absx >= f32(f32(10) + tol):
            absx = f32(absx / f32(10))
            expon += 1
    return (-absx if x < 0 else absx), expon


def fp_equalr(x, y) -> bool:
    x, y = f32(x), f32(y)
    if x == y:
        return True
    if x == 0 or y == 0:
        return False
    nx, ex = fp_normr(x)
    ny, ey = fp_normr(y)
    if ex != ey:
        return False
    return f32(f32(1) + abs(f32(nx - ny))) <= f32(f32(1) + f32(EPSILONR * f32(32)))


def fp_nondegenr(x1, x2):
    """Widen a range of no extent by 1% (at least 0.01) each way, until it has one."""
    x1, x2 = f32(x1), f32(x2)
    n = 0
    while fp_equalr(x1, x2):
        x1 = f32(x1 - max(f32(abs(x1) * f32(0.01)), f32(0.01)))
        x2 = f32(x2 + max(f32(abs(x2) * f32(0.01)), f32(0.01)))
        n += 1
    return x1, x2, n > 0


def log10r(x) -> f32:
    """log10 of a REAL as f2c's r_lg10 has it: log10e * log(x), in double."""
    return f32(LOG10E * math.log(float(x)))


def elogr(x) -> f32:
    x = f32(x)
    if x > 10:
        return log10r(x)
    if x >= -10:
        return f32(x / f32(10))
    return f32(-log10r(-x))


def aelogr(x) -> f32:
    x = f32(x)
    if x > 1:
        return f32(10.0 ** float(x))
    if x >= -1:
        return f32(x * f32(10))
    return f32(-(10.0 ** float(-x)))


def _pow_ri(n: int) -> float:
    """f2c's pow_ri(10., n): in double, the reciprocal first for a negative power."""
    from .iraffmt import pow_di

    return pow_di(10.0, n)


def fp_fixr(x) -> f32:
    absx = abs(f32(x))
    scaled, expon = absx, 0
    while scaled > 2147483647:
        scaled = f32(scaled / f32(10))
        expon += 1
    high = f32(int(scaled) * _pow_ri(expon))
    low = int(f32(absx - high))                 # a long, as fp_fixr declares it
    return f32(high + f32(low)) if x > 0 else f32(-high - f32(low))


def _gt_distance(x, step):
    absx = abs(f32(x))
    ltick = f32(fp_fixr(f32(absx / step)) * step)
    rtick = f32(ltick + step)
    if abs(f32(absx - ltick)) < abs(f32(rtick - absx)):
        return f32(absx - ltick), (-ltick if x < 0 else ltick)
    return f32(rtick - absx), (-rtick if x < 0 else rtick)


def gtickr(x1, x2, rough_nticks: int, logflag: bool):
    """The first major tick and the step between them, as gtick.gx places them."""
    x1, x2 = f32(x1), f32(x2)
    tol = f32(EPSILONR * f32(10))
    ndiv = max(1, rough_nticks - 1)
    while True:
        x = f32(1) if logflag else abs(f32(f32(x2 - x1) / f32(ndiv)))
        x, expon = fp_normr(x)
        if x < 1.5:
            x = f32(1)
        elif x < 2.5:
            x = f32(2)
        elif x < 4.0:
            x = f32(2.5)
        elif x < 7.5:
            x = f32(5)
        else:
            x, expon = f32(1), expon + 1
        step = f32(1) if logflag else f32(float(x) * _pow_ri(expon))
        distance, tick1 = _gt_distance(x1, step)
        if not f32(distance / step) < tol:
            if x1 < x2 and tick1 < x1:
                tick1 = f32(tick1 + step)
            elif x1 > x2 and tick1 > x1:
                tick1 = f32(tick1 - step)
        if x1 > x2:
            step = -step
        ndiv -= 1
        if abs(abs(x) - 1) < tol or not logflag or ndiv == 0:
            return f32(tick1), f32(step)


def round_axis(p1, p2, nmajor: int, scaling: str, round_: bool):
    """glb_find_ticks on one axis: with round set, or on a log axis, the window out
    to the major tick past each end. Fewer than three major ticks: no ticks at all,
    and nothing changes."""
    p1, p2 = f32(p1), f32(p2)
    if nmajor <= 2 or not (round_ or scaling != "linear"):
        return p1, p2
    if scaling == "log":
        q1, q2 = log10r(p1), log10r(p2)
    elif scaling == "elog":
        q1, q2 = elogr(p1), elogr(p2)
    else:
        q1, q2 = p1, p2
    tick1, step = gtickr(q1, q2, nmajor, scaling != "linear")
    if not fp_equalr(q1, tick1):
        tick1 = f32(tick1 - step)
        q1 = tick1
    length = f32(f32(q2 - q1) / step)
    if not fp_equalr(f32(q1 + f32(int(length) * step)), q2):
        q2 = f32(q1 + f32((int(length) + 1) * step))
    if scaling == "elog":
        return aelogr(q1), aelogr(q2)
    if scaling == "log":
        return f32(10.0 ** float(q1)), f32(10.0 ** float(q2))
    return q1, q2


def _minmax(v):
    """gascale's scan: the first definite value, then smaller ones and else larger."""
    finite = v[~np.isnan(v)]
    if finite.size == 0:
        return None
    return f32(finite.min()), f32(finite.max())


# ------------------------------------------------------------- reading


def _lexnum_number(line: str, ip: int):
    """gctod at ip: (value or None for INDEF, new ip), or None when no number is there."""
    from . import evexpr, iraffmt

    while ip < len(line) and line[ip] in " \t":
        ip += 1
    kind, length = evexpr.lexnum(line, ip)
    if kind == evexpr.NON or length == 0:
        return None
    text = line[ip:ip + length]
    if text.startswith("INDEF"):
        return float("nan"), ip + length
    if kind == evexpr.OCT:
        value = float(int(text.rstrip("bB"), 8))
    elif kind == evexpr.HEX:
        value = float(int(text.rstrip("xX"), 16))
    elif kind == evexpr.DEC:
        value = float(int(text))
    else:
        value = iraffmt.ctod(text)
        if value is None:
            return None
    return float(f32(value)), ip + length


def read_list(lines: list[str], name: str, rdmarks: bool, say) -> Curve:
    """gg_rdlist2: x,y pairs, or y alone numbered from 1, and with rdmarks a third
    column of mark sizes."""
    xs, ys, sizes = [], [], []
    ncols, szmark = 0, 1e-2
    last = [0.0, 0.0, 0.0]          # gargr's locals: what a failed read leaves
    for lineno, raw in enumerate(lines, 1):
        line = raw if raw.endswith("\n") else raw + "\n"
        if line.startswith("#"):
            continue
        if not line.strip(" \t\n"):
            continue
        values, ip = [], 0
        for _ in range(3 if rdmarks else 2):
            got = _lexnum_number(line, ip)
            if got is None:
                break
            values.append(got[0])
            ip = got[1]
        nscan = len(values)
        for k, v in enumerate(values):
            last[k] = v
        xval, yval = last[0], last[1]
        if rdmarks:
            szmark = last[2] if nscan >= 3 else szmark
        if ncols == 0 and nscan > 0:
            ncols = nscan
        if nscan == 0:
            say(f"no args; {name}, line {lineno}: {line}\n")
            continue
        if nscan == 1:
            if ncols == 2:
                say(f"only 1 arg; {name}, line {lineno}: {line}\n")
                continue
            yval, xval = xval, float(len(xs) + 1)
        elif nscan == 2 and rdmarks:
            say(f"no szmark field; {name}, line {lineno}: {line}\n")
            szmark = 1e-2
        xs.append(xval)
        ys.append(yval)
        if rdmarks:
            sizes.append(szmark)
    return Curve(np.asarray(xs, dtype=f32), np.asarray(ys, dtype=f32),
                 np.asarray(sizes, dtype=f32) if rdmarks else None)


def _section_parts(section: str | None) -> list[str]:
    if not section:
        return []
    return [p.strip() for p in section.strip("[]").split(",")]


def read_image(operand: str, axis: int, wcs: str):
    """gg_rdimage2: the projection of a section onto one axis (the mean over the
    others), its coordinates in `wcs`, the image's title with the section under it,
    and plt_wcs's label and format."""
    from . import images

    spec = images.parse(operand)
    try:
        data, header = images.read(spec)
    except (OSError, ValueError) as error:
        raise GraphError(f"Cannot open image ({operand})") from error
    data = np.asarray(data, dtype=f32)
    full = list(data.shape[::-1])                      # IRAF's axis order
    parts = _section_parts(spec.section)
    dropped = [i for i, part in enumerate(parts)
               if i < len(full) and re.fullmatch(r"[+-]?\d+", part)]
    kept = [i for i in range(len(full)) if i not in dropped]
    lens = [full[i] for i in kept]
    ndim = len(lens)
    if axis < 1 or axis > ndim:
        raise GraphError("Attempt to take projection over nonexistent axis")
    npix = lens[axis - 1]
    cube = data.reshape(lens[::-1] if lens else (1,))
    rows = cube.reshape(-1, lens[0])
    pv = np.zeros(npix, dtype=f32)
    if axis == 1:
        for row in rows:
            pv = (pv + row).astype(f32)
    else:
        index = np.indices(lens[:0:-1]).reshape(len(lens) - 1, -1)[::-1]
        for k, row in enumerate(rows):
            i = index[axis - 2, k]
            pv[i] = f32(pv[i] + np.cumsum(row, dtype=f32)[-1])
    total = 1
    for i, n in enumerate(lens, 1):
        total *= min(npix, n) if i == axis else n
    nsum = total // min(npix, lens[axis - 1])
    pv = (pv / f32(nsum)).astype(f32)

    title = str(header.get("OBJECT", "")).strip() if header is not None else ""
    if "\n" not in title:
        title += "\n"
    if spec.section:
        title += "[" + spec.section.strip("[]") + "]"
    x, label, xformat = _plt_wcs(header, full, kept, dropped, axis, npix, wcs)
    return Curve(x, pv, np.zeros(npix, dtype=f32)), title, label, xformat


def _plt_wcs(header, full, kept, dropped, axis, npix, wcs):
    from .implot import _wat_attributes

    n = len(full)
    try:
        wcsdim = max(n, int(header.get("WCSDIM", n)))
    except (TypeError, ValueError):
        wcsdim = n
    paxis = kept[axis - 1]
    dx = f32(f32(f32(npix) - f32(1)) / f32(npix - 1)) if npix > 1 else f32(0)
    xin = np.ones(wcsdim, dtype=f32)
    for i in range(wcsdim):
        if i in dropped or i >= n:
            xin[i] = f32(1)
        else:
            xin[i] = f32(f32(1 + full[i]) / f32(2))
    logical = np.array([f32(f32(1) + f32(f32(i) * dx)) for i in range(npix)], dtype=f32)
    if wcs == "logical":
        x = logical
    else:
        matrix, vector, rotated = _compiled_real(header, wcsdim, wcs)
        x = np.empty(npix, dtype=f32)
        for k in range(npix):
            p = xin.copy()
            p[paxis] = logical[k]
            if not rotated:
                x[k] = f32(f32(matrix[paxis, paxis] * p[paxis]) + vector[paxis])
            else:
                v = vector[paxis]
                for i in range(wcsdim):
                    v = f32(v + f32(matrix[paxis, i] * p[i]))
                x[k] = v
    xformat = ""
    if wcs == "logical":
        label = ("Column (pixels)", "Line (pixels)")[axis - 1] if axis <= 2 else "Pixels"
    elif wcs == "physical":
        label = ("Column (pixels)", "Line (pixels)")[paxis] if paxis <= 1 else "Pixels"
    else:
        attributes = _wat_attributes(header, paxis + 1) if _has_world(header, wcsdim) \
            else {}
        label = ""
        if "label" in attributes:
            label = attributes["label"]
            if "units" in attributes:
                label = f"{attributes['label']} ({attributes['units']})"
        xformat = attributes.get("format", "")
    return x, label, xformat


def _has_world(header, n: int) -> bool:
    from . import mwcs

    return header is not None and any(mwcs._has_world(header, i, n)
                                      for i in range(1, n + 1))


def _compiled_real(header, n: int, wcs: str):
    """mw_sctran's logical to `wcs` transform, compiled in double and kept, as
    mw_ctranr applies it, as its single-precision copy; and whether it rotates."""
    from . import mwcs

    num = mwcs._number
    ltm = np.array([[num(header, f"LTM{i + 1}_{j + 1}", float(i == j)) for j in range(n)]
                    for i in range(n)])
    ltv = np.array([num(header, f"LTV{i + 1}", 0.0) for i in range(n)])
    inverse = np.linalg.inv(ltm)
    full = inverse.copy()
    shift = -mwcs._vmul(inverse, ltv)
    if wcs == "world" and _has_world(header, n):
        types = [str(header.get(f"CTYPE{i + 1}", "")).upper() for i in range(n)]
        if any(re.search(r"-[A-Z0-9]{3}$", ctype) for ctype in types):
            raise GraphError("graph does not plot a celestial world system: use wcs "
                             "logical or physical")
        matrix, inv, wshift = mwcs._compiled(header, n)
        world = mwcs._mmul(matrix, inv)
        for axis in range(n):
            if mwcs._has_world(header, axis + 1, n):
                full[axis], shift[axis] = world[axis], wshift[axis]
    off = full - np.diag(np.diag(full))
    rotated = bool((np.abs(off) > 100 * 2.220446049250313e-16).any())
    return full.astype(f32), shift.astype(f32), rotated


# ------------------------------------------------------------- the task


def operands(text: str) -> list[str]:
    """imtopen's list: commas, @files and patterns; a name that matches nothing is
    kept, to fail when it is opened, as IRAF's."""
    from .images import split_list

    out = []
    for piece in split_list(text):
        if piece.startswith("@"):
            path = Path(piece[1:])
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        out.extend(operands(line.strip()))
            continue
        name = re.sub(r"\[.*$", "", piece)
        found = sorted(glob.glob(name)) if any(c in name for c in "*?") else []
        out.extend(f + piece[len(name):] for f in found) if found else out.append(piece)
    return out


def is_list(operand: str) -> bool:
    """gg_optype: STDIN and text files are lists, anything else an image."""
    stripped = operand.strip()
    if len(stripped) <= 1 or stripped.startswith("STDIN"):
        return True
    if stripped.endswith("]"):
        return False
    path = Path(stripped)
    if not path.is_file():
        return False
    with open(path, "rb") as fd:
        head = fd.read(2880)
    if head.startswith(b"SIMPLE  =") or b"\0" in head:
        return False
    return path.suffix.lower() not in (".fits", ".fit", ".fts", ".imh", ".pl")


def _ints(text: str) -> list[int]:
    return [int(w) for w in re.findall(r"[+-]?\d+", text.replace(",", " "))]


def graph(params: dict, stdin: str | None = None) -> Plot:
    """t_graph and ggplot. `params` are the task's (graph.par's names); `stdin` the
    text of a redirected standard input, which IRAF's graph then reads whatever the
    input says."""
    p = {**DEFAULTS, **params}
    err: list[str] = []
    say = err.append
    names = ["STDIN"] if stdin is not None else operands(str(p["input"]))

    pointmode = bool(p["pointmode"])
    marker = str(p["marker"])
    szmarker, rdmarks, mark = 0.0, False, "box"
    if pointmode:
        if marker in MARKERS:
            mark = marker
        elif marker not in PLOT_TYPES:
            say("Unrecognized marker type, using 'box'\n")
        if mark != "point":
            szmarker = float(f32(p["szmarker"]))
            rdmarks = szmarker <= 0

    title = xlabel = xformat = ""
    curves: list[Curve] = []
    for operand in names:
        if len(curves) + 1 > MAX_CURVES:
            raise GraphError("Maximum of 20 curves can be overplotted")
        try:
            if is_list(operand):
                if operand.strip().startswith("STDIN"):
                    lines = (stdin or "").splitlines(keepends=True)
                else:
                    lines = Path(operand.strip()).read_text(
                        encoding="utf-8", errors="replace").splitlines(keepends=True)
                curves.append(read_list(lines, operand, rdmarks, say))
            else:
                curve, ititle, label, xformat = read_image(operand, int(p["axis"]),
                                                           str(p["wcs"]))
                if not title:
                    title = ititle
                if not xlabel:
                    xlabel = label
                curves.append(curve)
        except GraphError as error:
            say(f"Warning: {error}\n")
    if not curves:
        raise GraphError("No curves read")

    append = bool(p["append"])
    if append or p["overplot"]:
        raise GraphError("harwood has no graphics device to append to: give every "
                         "curve to one graph")
    wx1, wx2, wy1, wy2 = (f32(p[k]) for k in ("wx1", "wx2", "wy1", "wy2"))
    xauto, yauto = fp_equalr(wx1, wx2), fp_equalr(wy1, wy2)
    viewport = None
    vx = tuple(float(f32(p[k])) for k in ("vx1", "vx2", "vy1", "vy2"))
    if not fp_equalr(vx[0], vx[1]) and not fp_equalr(vx[2], vx[3]):
        viewport = vx
    box = bool(p["box"])
    ylabel = yformat = ""
    ticklabels = False
    if box:
        if p["title"] != "imtitle":
            title = str(p["title"])
        if p["xlabel"] != "wcslabel":
            xlabel = str(p["xlabel"])
        ylabel = str(p["ylabel"])
        if p["xformat"] != "wcsformat":
            xformat = str(p["xformat"])
        yformat = str(p["yformat"])
        ticklabels = bool(p["ticklabels"])

    xtran = "log" if p["logx"] else "linear"
    ytran = "log" if p["logy"] else "linear"

    def iformat(value, fmt):
        return f32(value * f32(15)) if re.search("[HM]", fmt) else value

    window = [iformat(wx1, xformat), iformat(wx2, xformat), iformat(wy1, yformat),
              iformat(wy2, yformat)]

    if p["lintran"]:
        for c in curves:
            c.x = lintran(c.x, f32(p["p1"]), f32(p["p2"]), f32(p["q1"]), f32(p["q2"]))
    if p["transpose"]:
        for c in curves:
            c.x, c.y = c.y, c.x

    for axis, auto in ((0, xauto), (1, yauto)):
        if not auto:
            continue
        first = _minmax(curves[0].x if axis == 0 else curves[0].y)
        if first is None:
            raise GraphError("Entire graphics vector is indefinite")
        low, high = first
        for c in curves[1:]:
            more = _minmax(c.x if axis == 0 else c.y)
            if more is not None:
                low, high = min(low, more[0]), max(high, more[1])
        window[2 * axis], window[2 * axis + 1] = low, high

    if box:
        window[0], window[1], _ = fp_nondegenr(window[0], window[1])
        window[2], window[3], _ = fp_nondegenr(window[2], window[3])
        if xtran != "linear" and (window[0] <= 0 or window[1] <= 0):
            xtran = "elog"
        if ytran != "linear" and (window[2] <= 0 or window[3] <= 0):
            ytran = "elog"
        window[0], window[1] = round_axis(window[0], window[1], int(p["majrx"]), xtran,
                                          bool(p["round"]))
        window[2], window[3] = round_axis(window[2], window[3], int(p["majry"]), ytran,
                                          bool(p["round"]))

    ltypes, colors = _ints(str(p["ltypes"])), _ints(str(p["colors"]))
    ltype = color = 0
    for i, c in enumerate(curves):
        ltype = ltype + 1 if not ltypes else (ltypes[i] if i < len(ltypes) else ltype)
        ltype = (ltype - 1) % 4 + 1
        color = color + 1 if not colors else (colors[i] if i < len(colors) else color)
        color = (color - 1) % 9 + 1
        c.ltype, c.color = ltype, color

    plot = Plot(curves=curves, window=tuple(float(w) for w in window), xtran=xtran,
                ytran=ytran, viewport=viewport, aspect=not p["fill"], box=box,
                ticklabels=ticklabels, title=title, xlabel=xlabel, ylabel=ylabel,
                xformat=xformat, yformat=yformat, pointmode=pointmode, marker=marker,
                mark=mark, rdmarks=rdmarks, szmarker=szmarker, err="".join(err))
    plot.strokes = _strokes(plot)
    return plot


def lintran(x, p1in, p2in, q1, q2) -> np.ndarray:
    """gg_lintran: pixels p1..p2 (1..npix when both are 0) onto q1..q2."""
    if p1in == 0 and p2in == 0:
        p1, p2 = f32(1), f32(len(x))
    else:
        p1, p2 = p1in, p2in
    xscale = f32(q2 - q1) if f32(p2 - p1) == 0 else f32(f32(q2 - q1) / f32(p2 - p1))
    return ((((x - p1).astype(f32)) * xscale).astype(f32) + q1).astype(f32)


def _strokes(plot: Plot) -> list[Stroke]:
    out = []
    for c in plot.curves:
        n = len(c.x)
        if plot.pointmode:
            if plot.mark == "point":
                out.append(Stroke("points", c.x, c.y, c.ltype, c.color))
                continue
            if not plot.rdmarks:
                sizes = np.full(n, plot.szmarker, dtype=f32)
            else:
                sizes = c.size if c.size is not None else np.zeros(n, dtype=f32)
                if plot.szmarker < 0:
                    sizes = (sizes * f32(plot.szmarker)).astype(f32)
            for j in range(n):
                sx = sy = float(sizes[j])
                if plot.rdmarks and plot.mark == "vebar":
                    sx = 1.0
                elif plot.rdmarks and plot.mark == "hebar":
                    sy = 1.0
                out.append(Stroke(plot.mark, c.x[j:j + 1], c.y[j:j + 1], c.ltype, c.color,
                                  sx, sy))
            continue
        kind = plot.marker if plot.marker in PLOT_TYPES else "line"
        if kind == "line" or n < 2:
            out.append(Stroke("line", c.x, c.y, c.ltype, c.color))
            continue
        x, y = c.x, c.y
        xs, ys = [], []
        bottom = f32(plot.window[2])

        def half(a, b):
            return f32(f32(f32(f32(3) * a) - b) / f32(2))
        xs.append(half(x[0], x[1]))
        if kind == "lhist":
            ys.append(y[0])
            for i in range(n - 1):
                x2 = f32(f32(x[i] + x[i + 1]) / f32(2))
                xs += [x2, x2]
                ys += [y[i], y[i + 1]]
            xs.append(half(x[n - 1], x[n - 2]))
            ys.append(y[n - 1])
        else:
            ys.append(bottom)
            xs.append(xs[0])
            ys.append(y[0])
            for i in range(n - 1):
                x2 = f32(f32(x[i] + x[i + 1]) / f32(2))
                xs += [x2, x2, x2]
                ys += [y[i], bottom, y[i + 1]]
            x2 = half(x[n - 1], x[n - 2])
            xs += [x2, x2]
            ys += [y[n - 1], bottom]
        out.append(Stroke("line", np.asarray(xs, dtype=f32), np.asarray(ys, dtype=f32),
                          c.ltype, c.color))
    return out


DEFAULTS = {
    "input": "", "wx1": 0.0, "wx2": 0.0, "wy1": 0.0, "wy2": 0.0, "wcs": "logical",
    "axis": 1, "transpose": False, "pointmode": False, "marker": "box",
    "szmarker": 5e-3, "ltypes": "", "colors": "", "logx": False, "logy": False,
    "box": True, "ticklabels": True, "xlabel": "wcslabel", "ylabel": "",
    "xformat": "wcsformat", "yformat": "", "title": "imtitle", "lintran": False,
    "p1": 0.0, "p2": 0.0, "q1": 0.0, "q2": 1.0, "vx1": 0.0, "vx2": 0.0, "vy1": 0.0,
    "vy2": 0.0, "majrx": 5, "minrx": 5, "majry": 5, "minry": 5, "overplot": False,
    "append": False, "device": "stdgraph", "round": False, "fill": True,
}
