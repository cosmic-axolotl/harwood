"""GTOOLS: what a graph keeps from one draw to the next -- its window, which way
its axes run, its marks, its title block -- and the two ways a person changes
it: the key after `w`, and the colon commands that begin with `/`.

From pkg/xtools/gtools (gtinit, gtwindow, gtcolon, gtswind, gtset, gtget,
gtlabax, gtplot). A task holds one `GTools` and asks it, when it draws, for the
window (`swind`) and the title block (`title_block`); nothing here draws. What
the task sets each time it draws -- in inlfit the parameters line, the title, the
labels and their units -- overwrites what a colon command put there, as in IRAF,
whose help says so ("Some applications set the values every time the graph is
redrawn").

The quirks kept, because they decide what is drawn and printed:

* `:/shift` with a coordinate left out uses the last real any colon command
  read: IRAF's gargr leaves its static local as it was when there is nothing to
  read, and gt_colon uses the value without asking nscan.
* `:/window` is not reachable: the dictionary spells it "expand" a second time,
  so the name falls through to gt_colon1, which knows no such command.
* Under `:/transpose`, `w` sets the limits of the data's x from the screen's x:
  gt_window reads the cursor without gt_gcur1, which is what swaps them.
* The window's limits are set and read through gt_setr and gt_getr, which swap
  min and max on a flipped axis, so `w j` sets the edge on the left of the screen
  whichever way the axis runs; `w f` flips the axis and leaves the limits, which
  then mean the other edges.
* gt_colon1's axis settings (`:/drawgrid`, `:/nmajor`...) are kept as numbers;
  they change how IRAF's glabax draws the frame, which the panel draws its own way.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import iraffmt
from .gio import widened

HELP = Path(__file__).resolve().parent.parent / "data" / "iraf" / "scr"

MARKS = ("point", "box", "plus", "cross", "diamond", "hline", "vline", "hebar", "vebar",
         "circle")
TYPES = ("mark", "line", "histogram")
TRANSFORMS = ("linear", "logarithmic")
XAXES = ("none", "bottom", "top", "both")
YAXES = ("none", "left", "right", "both")

#: gt_colon's dictionary, "expand" twice as IRAF has it.
CMDS = ("help", "xview", "yview", "xwindow", "ywindow", "sysid", "parameters", "title",
        "subtitle", "comments", "xlabel", "ylabel", "xunits", "yunits", "drawtitle",
        "drawxlabels", "drawylabels", "type", "mark", "line", "xsize", "ysize", "color",
        "xtransform", "ytransform", "xflip", "yflip", "transpose", "xformat", "yformat",
        "xbuf", "ybuf", "clip", "redraw", "expand", "shift", "expand", "uivalues")

#: gt_colon1's; the first seventeen take one number, the rest two.
CMDS1 = ("txup", "txsize", "txpath", "txspacing", "txhjustify", "txvjustify", "txfont",
         "txquality", "txcolor", "drawtitle", "titlesize", "titlejust", "ntitlelines",
         "aspect", "charsize", "titlecolor", "framecolor", "drawaxes", "setaxispos",
         "axispos1", "axispos2", "drawgrid", "round", "labelaxis", "axislabelsize",
         "drawticks", "labelticks", "nmajor", "nminor", "majorlength", "minorlength",
         "majorwidth", "minorwidth", "axiswidth", "ticklabelsize", "gridcolor",
         "axislabelcolor", "axiscolor", "ticklabelcolor", "tickcolor", "axes", "ticks",
         "colors")
_DRAWAXES, _AXES = CMDS1.index("drawaxes") + 1, CMDS1.index("axes") + 1
_INTEGERS1 = {"txup", "txpath", "txhjustify", "txvjustify", "txfont", "txquality",
              "txcolor", "drawtitle", "titlejust", "ntitlelines", "titlecolor", "framecolor",
              "setaxispos", "drawgrid", "round", "labelaxis", "drawticks", "labelticks",
              "nmajor", "nminor", "gridcolor", "axislabelcolor", "axiscolor",
              "ticklabelcolor", "tickcolor"}
_BOOLEANS1 = {"drawgrid", "round", "labelaxis", "drawticks", "labelticks"}

_REAL = re.compile(r"\s*(INDEF[RD]?|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?)(?=\s|$)")


def strdic(word: str, words) -> int:
    """IRAF's strdic, as it walks the dictionary: an exact match wins where it is
    met, an abbreviation of two entries before that is no match at all."""
    word = word.lstrip()
    if not words:
        return 0
    match = 0
    for entry, name in enumerate(words, start=1):
        if word and name.startswith(word):
            if len(name) == len(word):
                return entry
            if match:
                return 0
            match = entry
        elif not word and not name:
            return entry
    return match


class Scan:
    """sscan and gargwrd, gargr, gargb, gargstr: a field at a time, nscan
    counting them up to the first that cannot be read. `last` is gargr's static
    local, what a gargr with nothing to read hands back."""

    def __init__(self, line: str, last: list[float | None]):
        self.line, self.pos, self.n, self.stop = line, 0, 0, False
        self.last = last

    def wrd(self) -> str:
        if self.stop:
            return ""
        rest = self.line[self.pos:]
        stripped = rest.lstrip()
        if not stripped:
            self.stop = True
            return ""
        word = stripped.split()[0]
        self.pos += len(rest) - len(stripped) + len(word)
        self.n += 1
        return word

    def r(self) -> float | None:
        if not self.stop:
            found = _REAL.match(self.line, self.pos)
            if found is None:
                self.stop = True
            else:
                self.pos = found.end()
                self.n += 1
                text = found.group(1)
                self.last[0] = None if text.startswith("INDEF") else float(
                    np.float32(float(text.replace("d", "e").replace("D", "e"))))
        return self.last[0]

    def i(self) -> int | None:
        """gargi: its own gargd, so not the real gargr keeps; INDEF is INDEFI."""
        if self.stop:
            return None
        found = _REAL.match(self.line, self.pos)
        if found is None:
            self.stop = True
            return None
        self.pos = found.end()
        self.n += 1
        text = found.group(1)
        if text.startswith("INDEF"):
            return -2147483647
        value = float(text.replace("d", "e").replace("D", "e"))
        return -2147483647 if abs(value) > 2147483647 else int(value)

    def b(self) -> bool | None:
        if self.stop:
            return None
        rest = self.line[self.pos:]
        stripped = rest.lstrip()
        if not stripped or stripped[0] not in "YyNn":
            self.stop = True
            return None
        word = re.match(r"[A-Za-z_]*", stripped)[0]
        self.pos += len(rest) - len(stripped) + len(word)
        self.n += 1
        return stripped[0] in "Yy"

    def str(self) -> str:
        """gargstr, and the loop after it that skips the white space."""
        if self.stop:
            return ""
        text = self.line[self.pos:].split("\n")[0]
        self.pos = len(self.line)
        self.n += 1
        return text.lstrip(" \t")


def _g(value: float | None) -> str:
    return "INDEF" if value is None else iraffmt.g(float(value))


def _b(value: bool) -> str:
    return "YES" if value else "NO"


def _nint(value: float | None) -> int:
    return 0 if value is None else int(math.floor(value + 0.5)) if value >= 0 else -int(
        math.floor(-value + 0.5))


def elog(value: float) -> float:
    """GIO's elog: log10 beyond 10 either way, a straight line through zero within."""
    if value > 10:
        return math.log10(value)
    if value < -10:
        return -math.log10(-value)
    return value / 10


def unelog(value: float) -> float:
    if value > 1:
        return 10.0 ** value
    if value < -1:
        return -(10.0 ** -value)
    return value * 10


def elog_decades(low: float, high: float) -> tuple[float, float]:
    """What glabax does to a logarithmic axis: out to whole units of elog, so
    [0.885, 24.4] is drawn as [0, 100]. Found from IRAF's metacode; the direction
    the window runs in is kept."""
    a, b = min(low, high), max(low, high)
    a, b = unelog(math.floor(elog(a))), unelog(math.ceil(elog(b)))
    return (a, b) if low <= high else (b, a)


@dataclass
class GTools:
    """gt_init's GTOOLS structure; None is INDEF."""
    vxmin: float | None = None
    vxmax: float | None = None
    vymin: float | None = None
    vymax: float | None = None
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None
    xtran: str = "linear"
    ytran: str = "linear"
    xsize: float = 2.0
    ysize: float = 2.0
    sysid: bool = True
    params: str = ""
    title: str = ""
    subtitle: str = ""
    comments: str = ""
    xlabel: str = ""
    ylabel: str = ""
    xunits: str = ""
    yunits: str = ""
    drawtitle: bool = True
    drawxlabels: bool = True
    drawylabels: bool = True
    xformat: str = ""
    yformat: str = ""
    xbuf: float = 0.03
    ybuf: float = 0.03
    lclip: float = 0.0
    hclip: float = 0.0
    xflip: bool = False
    yflip: bool = False
    transpose: bool = False
    type: int = 1                   #: 1 mark, 2 line, 3 histogram
    mark: str = "plus"
    line: int = 1
    color: int = 1
    axes: dict = field(default_factory=dict)    #: gt_colon1's, by name
    #: gargr's static local, shared with what else in the task reads reals.
    last_real: list = field(default_factory=lambda: [0.0])

    # ------------------------------------------------------------ gt_sets

    def set_type(self, text: str) -> str:
        """gt_sets GTTYPE: a mark's name is a graph of marks; line2, hist3 a line
        type with its style. Gives what IRAF writes to the error stream."""
        found = strdic(text, MARKS)
        if found:
            self.type, self.mark = 1, MARKS[found - 1]
            return ""
        word = text[:10]
        if len(text) > 4 and text[4].isdigit():
            self.line = int(text[4])
            word = word[:4]
        found = strdic(word, TYPES)
        if not found:
            return f"Unknown graph type `{text}'\n"
        self.type = found
        return ""

    def set_mark(self, text: str) -> str:
        found = strdic(text, MARKS)
        if not found:
            return f"Unknown mark type `{text}'\n"
        self.mark = MARKS[found - 1]
        return ""

    def set_transform(self, axis: str, text: str) -> str:
        found = strdic(text, TRANSFORMS)
        if not found:
            return f"Unknown {axis.upper()} transformation type `{text}'\n"
        setattr(self, f"{axis}tran", TRANSFORMS[found - 1])
        return ""

    def type_name(self) -> str:
        """gt_gets GTTYPE: a graph of marks says which mark."""
        if self.type == 1:
            return self.mark
        return f"{'line' if self.type == 2 else 'hist'}{self.line}"

    def reset_window(self) -> None:
        self.xmin = self.xmax = self.ymin = self.ymax = None

    # ------------------------------------------------- gt_setr and gt_getr

    def _flipped(self, name: str) -> str:
        """The field a limit is kept in: on a flipped axis, min is max."""
        if getattr(self, f"{name[0]}flip"):
            return name[0] + ("max" if name.endswith("min") else "min")
        return name

    def setr(self, name: str, value: float | None) -> None:
        """gt_setr of GTXMIN, GTXMAX, GTYMIN or GTYMAX, as a real."""
        setattr(self, self._flipped(name),
                None if value is None else float(np.float32(value)))

    def getr(self, name: str) -> float | None:
        return getattr(self, self._flipped(name))

    # ------------------------------------------------------------ windows

    def swind(self, x: tuple[float, float], y: tuple[float, float]) -> tuple:
        """gascale's range of the data, then gt_swind: the limits set, the room
        around what was not, the flips, and the axes swapped under transpose;
        then what glabax does to a logarithmic axis. The window as drawn, x1 x2 y1
        y2 on the screen."""
        def axis(span, low, high, buf, flip):
            r = np.float32
            a, b = r(span[0]), r(span[1])
            d = r(b - a)
            a = float(r(a - r(buf) * d)) if low is None else low
            b = float(r(b + r(buf) * d)) if high is None else high
            if low is None and high is None:
                a, b = widened(a, b)
            return (b, a) if flip else (a, b)

        xs = axis(x, self.xmin, self.xmax, self.xbuf, self.xflip)
        ys = axis(y, self.ymin, self.ymax, self.ybuf, self.yflip)
        if self.transpose:
            xs, ys = ys, xs
        xlog, ylog = self.logs()
        if xlog:
            xs = elog_decades(*xs)
        if ylog:
            ys = elog_decades(*ys)
        return (*xs, *ys)

    def logs(self) -> tuple[bool, bool]:
        """Which of the screen's axes are logarithmic."""
        logs = (self.xtran == "logarithmic", self.ytran == "logarithmic")
        return logs[::-1] if self.transpose else logs

    def window1(self, key: str, wx: float | None, wy: float | None,
                window: tuple[float, float, float, float]) -> tuple[str, bool]:
        """gt_window1: one key, the cursor at (wx, wy), the window as drawn. What
        it prints and whether to redraw."""
        r = np.float32
        x1, x2, y1, y2 = (r(v) for v in window)
        dx, dy = r(x2 - x1), r(y2 - y1)
        wx = r(r(x1 + x2) / r(2)) if wx is None else r(wx)
        wy = r(r(y1 + y2) / r(2)) if wy is None else r(wy)
        half, quarter, most = r(2), r(4), r(0.75)
        sets = {
            "a": dict(xmin=None, xmax=None, ymin=None, ymax=None),
            "b": dict(ymin=wy),
            "c": dict(xmin=wx - dx / half, xmax=wx + dx / half, ymin=wy - dy / half,
                      ymax=wy + dy / half),
            "d": dict(ymin=y1 - most * dy, ymax=y2 - most * dy),
            "j": dict(xmin=wx),
            "k": dict(xmax=wx),
            "l": dict(xmin=x1 - most * dx, xmax=x2 - most * dx),
            "m": dict(xmin=None, xmax=None),
            "n": dict(ymin=None, ymax=None),
            "p": dict(xmin=wx - dx, xmax=wx + dx, ymin=wy - dy, ymax=wy + dy),
            "r": dict(xmin=x1 + most * dx, xmax=x2 + most * dx),
            "t": dict(ymax=wy),
            "u": dict(ymin=y1 + most * dy, ymax=y2 + most * dy),
            "x": dict(xmin=wx - dx / quarter, xmax=wx + dx / quarter),
            "y": dict(ymin=wy - dy / quarter, ymax=wy + dy / quarter),
            "z": dict(xmin=wx - dx / quarter, xmax=wx + dx / quarter, ymin=wy - dy / quarter,
                      ymax=wy + dy / quarter),
        }
        if key == "?":
            return pagefile("gtwindow.key"), False
        if key == "f":
            self.xflip = not self.xflip
        elif key == "g":
            self.yflip = not self.yflip
        elif key in sets:
            for name, value in sets[key].items():
                self.setr(name, value)
        else:
            return "\a", False
        return "", True

    def window2(self, first: tuple[float, float], second: tuple[float, float],
                window: tuple[float, float, float, float]) -> bool:
        """gt_window2 for e: the two corners, each axis only if they differ."""
        x1, x2, y1, y2 = window
        (wx1, wy1), (wx2, wy2) = first, second
        if abs(wx2 - wx1) > 0.001 * abs(x2 - x1):
            self.setr("xmin", wx1)
            self.setr("xmax", wx2)
        if abs(wy2 - wy1) > 0.001 * abs(y2 - y1):
            self.setr("ymin", wy1)
            self.setr("ymax", wy2)
        return True

    # ------------------------------------------------------- colon commands

    def colon(self, text: str, window: tuple[float, float, float, float]
              ) -> tuple[str, str, bool]:
        """gt_colon: a command that begins with '/'. What it prints on each stream,
        and whether it asks for a new graph."""
        if not text.startswith("/"):
            return "", "", False
        scan = Scan(text[1:], self.last_real)
        word = scan.wrd()
        found = strdic(word, CMDS)
        name = CMDS[found - 1] if found else ""
        out, err, redraw = "", "", False
        if name == "help":
            out = pagefile("gtools.key")
        elif name in ("xview", "yview", "xwindow", "ywindow"):
            a, b = scan.r(), scan.r()
            low, high = {"xview": ("vxmin", "vxmax"), "yview": ("vymin", "vymax"),
                         "xwindow": ("xmin", "xmax"), "ywindow": ("ymin", "ymax")}[name]
            get, put = ((self.getr, self.setr) if name.endswith("window") else
                        (lambda n: getattr(self, n), lambda n, v: setattr(self, n, v)))
            if scan.n == 3:
                put(low, a)
                put(high, b)
            else:
                out = f"{name} = {_g(get(low))} {_g(get(high))}\n"
        elif name in ("sysid", "drawtitle", "drawxlabels", "drawylabels", "xflip", "yflip"):
            value = scan.b()
            if scan.n == 2:
                setattr(self, name, value)
            else:
                said = name[:-1] if name.startswith("draw") and name.endswith("s") else name
                out = f"{said} = {_b(getattr(self, name))}\n"
        elif name in ("parameters", "title", "subtitle", "comments", "xlabel", "ylabel",
                      "xunits", "yunits", "xformat", "yformat"):
            setattr(self, "params" if name == "parameters" else name, scan.str())
        elif name in ("type", "mark", "xtransform", "ytransform"):
            value = scan.wrd()
            if scan.n == 2:
                err = (self.set_type(value) if name == "type" else
                       self.set_mark(value) if name == "mark" else
                       self.set_transform(name[0], value))
            else:
                shown = (self.type_name() if name == "type" else self.mark if name == "mark"
                         else getattr(self, f"{name[0]}tran"))
                out = f"{name} = {shown}\n"
        elif name in ("line", "color"):
            value = scan.i()
            if scan.n == 2:
                setattr(self, name, value)
            else:
                out = f"{name} = {getattr(self, name)}\n"
        elif name in ("xsize", "ysize", "xbuf", "ybuf"):
            value = scan.r()
            if scan.n == 2:
                setattr(self, name, value)
            else:
                out = f"{name} = {_g(getattr(self, name))}\n"
        elif name == "clip":
            a, b = scan.r(), scan.r()
            if scan.n == 3:
                self.lclip, self.hclip = a, b
            else:
                out = f"clip = {_g(self.lclip)} {_g(self.hclip)}\n"
        elif name == "transpose":
            self.transpose = not self.transpose
        elif name == "redraw":
            redraw = True
        elif name == "expand":
            values = [scan.r() for _ in range(4)]
            if scan.n == 5:
                for attr, value in zip(("xmin", "xmax", "ymin", "ymax"), values, strict=True):
                    if value != self.getr(attr):
                        self.setr(attr, value)
                        redraw = True
        elif name == "shift":
            redraw = self._shift(scan.r(), scan.r())
        elif name == "uivalues":
            pass                            # values for a GUI that is not there
        else:
            self._colon1(text)
        return out, err, redraw

    def _shift(self, x: float | None, y: float | None) -> bool:
        redraw = False
        for value, low, high in ((x, "xmin", "xmax"), (y, "ymin", "ymax")):
            a, b = self.getr(low), self.getr(high)
            if value is None:
                if a is not None or b is not None:
                    self.setr(low, None)
                    self.setr(high, None)
                    redraw = True
            elif a is not None and b is not None:
                width = np.float32(b - a)
                offset = np.float32(np.float32(value) - np.float32(a + b) / 2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    moved = abs(offset / width) > 0.001
                if moved:
                    self.setr(low, a + offset)
                    self.setr(high, b + offset)
                redraw = True
        return redraw

    def _colon1(self, text: str) -> None:
        """gt_colon1: how glabax draws the frame; nothing is printed."""
        scan = Scan(text[1:], self.last_real)
        found = strdic(scan.wrd(), CMDS1)
        name = CMDS1[found - 1] if found else ""
        if found < _DRAWAXES:
            value = scan.r()
            if scan.n != 2 or not name:
                return
            self.axes[name] = _nint(value) if name in _INTEGERS1 else value
        elif found < _AXES:
            if name == "drawaxes":
                x, y = strdic(scan.wrd(), XAXES), strdic(scan.wrd(), YAXES)
                if scan.n == 3:
                    if x > 0:
                        self.axes["xdrawaxes"] = x - 1
                    if y > 0:
                        self.axes["ydrawaxes"] = y - 1
                return
            if name in _BOOLEANS1:
                values = [int(bool(scan.b())), int(bool(scan.b()))]
            else:
                values = [scan.r(), scan.r()]
            if scan.n != 3:
                return
            if name in _INTEGERS1:
                values = [_nint(v) for v in values]
            self.axes[f"x{name}"], self.axes[f"y{name}"] = values
        elif name == "axes":
            got = [strdic(scan.wrd(), XAXES), scan.r(), scan.b(),
                   strdic(scan.wrd(), XAXES), scan.r(), scan.b()]
            if scan.n == 7:
                for axis, (draw, width, grid) in (("x", got[:3]), ("y", got[3:])):
                    self.axes[f"{axis}drawaxes"] = draw
                    for part in ("axiswidth", "majorwidth", "minorwidth"):
                        self.axes[f"{axis}{part}"] = width
                    self.axes[f"{axis}drawgrid"] = int(bool(grid))
        elif name == "ticks":
            got = [scan.b(), scan.b(), scan.r(), scan.r(),
                   scan.b(), scan.b(), scan.r(), scan.r()]
            if scan.n == 9:
                for axis, part in (("x", got[:4]), ("y", got[4:])):
                    for key, value in zip(("drawticks", "labelticks", "nmajor", "nminor"),
                                          part, strict=True):
                        self.axes[f"{axis}{key}"] = _nint(float(value))
        elif name == "colors":
            got = [scan.r() for _ in range(12)]
            if scan.n == 13:
                names = ("framecolor", "titlecolor", "xgridcolor", "xaxislabelcolor",
                         "xaxiscolor", "xticklabelcolor", "xtickcolor", "ygridcolor",
                         "yaxislabelcolor", "yaxiscolor", "yticklabelcolor", "ytickcolor")
                for key, value in zip(names, got, strict=True):
                    self.axes[key] = _nint(value)

    # ------------------------------------------------------------ labels

    def title_block(self) -> list[str]:
        """gt_labax's title, less the system banner: the parameters, the title,
        the subtitle and the comments, those that are set; none unless drawtitle."""
        if not self.drawtitle:
            return []
        lines = []
        for part in (self.params, self.title, self.subtitle, self.comments):
            if part.rstrip("\n"):
                lines.extend(part.rstrip("\n").split("\n"))
        return lines

    def label(self, axis: str) -> str:
        """gt_labax's label of the data's x or y: the label and (units)."""
        if not getattr(self, f"draw{axis}labels"):
            return ""
        text = getattr(self, f"{axis}label")
        units = getattr(self, f"{axis}units")
        return text + (f" ({units})" if units else "")

    # ------------------------------------------------------ the nearest point

    def on_screen(self, xs, ys):
        """The data's x and y as the screen's: swapped under transpose, as gt_plot
        draws them and gt_gcur1 swaps the cursor back."""
        return (ys, xs) if self.transpose else (xs, ys)

    def ndc(self, xs, ys, window, viewport, view=None, size=None):
        """gctran to NDC: the screen's coordinates through the window as drawn (or
        the view a panel has zoomed to), logarithmic where the axis is, over the
        viewport's width and height (or the panel's)."""
        x1, x2, y1, y2 = view if view is not None else window
        width, height = size if size is not None else viewport
        xlog, ylog = self.logs()

        def one(values, a, b, log, extent):
            values = np.atleast_1d(np.asarray(values, dtype=float))
            if log:
                values = np.array([elog(v) for v in values])
                a, b = elog(a), elog(b)
            return (values - a) / ((b - a) or 1.0) * extent

        return one(xs, x1, x2, xlog, width), one(ys, y1, y2, ylog, height)

    def nearest(self, xs, ys, x: float, y: float, window, viewport, scale=None,
                use=None) -> int | None:
        """The point nearest the cursor on the screen, as the tasks' _nearest
        procedures find it through gctran: the first of several at one distance,
        as the strict comparison leaves it. xs, ys and the cursor are the screen's;
        `use` the points that may be chosen; `scale` a panel's (view, size)."""
        view, size = scale if scale is not None else (None, None)
        px, py = self.ndc(xs, ys, window, viewport, view, size)
        cx, cy = self.ndc([x], [y], window, viewport, view, size)
        best, found = float(np.finfo(np.float32).max), None
        for i in range(px.size):
            if use is not None and not use[i]:
                continue
            r2 = float(np.float32((px[i] - cx[0]) ** 2 + (py[i] - cy[0]) ** 2))
            if r2 < best:
                best, found = r2, i
        return found


def span(values) -> tuple[float, float]:
    """gascale: the range of the values there are."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return -1.0, 1.0
    return float(values.min()), float(values.max())


#: What gt_plot draws for a type other than marks.
DRAWN = {2: "line", 3: "histogram"}


def pagefile(name: str) -> str:
    """gpagefile: the help file as it is, its standout marks dropped."""
    text = (HELP / name).read_text(encoding="utf-8")
    return text.replace("\x0e", "").replace("\x0f", "")
