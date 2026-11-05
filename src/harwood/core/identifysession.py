"""identify's interactive half: the keys, the graphs, the colon commands.

From noao/onedspec/identify (ididentify, idgraph, idmark, iddofit, iddoshift, idcolon,
idshow, idlog). The arithmetic is core.identify's and icfit's. Nothing here knows about a
screen: a key is a method call, what IRAF would print goes to `stdout`, and `graph` gives
back what IRAF would draw, so the panel and a file of cursor commands are one session.
f hands the keys to an icfit session, as IRAF does, until its q.

The quirks kept, because they decide what is written. A line marked within minsep of a
feature does not make a new one: it loses that feature's user coordinate, to the list or
to what is typed. d in icfit deletes the feature; rejected, it stays with no weight. A
feature's fit does not change when the user coordinate does, only after f. a applies to
the key after next as well as the next, as the counter runs; only c and d look at it.

The graph is gtools' (`core.gtools`), as id_graph1 draws it through gt_swind: w and
the /commands set its window, flip, transpose and mark it. The quirks of that kept:
the keys take the cursor as clgcur gives it, not swapped under :/transpose, and the
features' ticks stay on the screen's x whatever is on it; a window that leaves the
whole spectrum out lets go of its x limits; z's graph, id_graph2, sets its window
itself and none of that applies to it but the drawing; a new dispersion solution
sets the x label again, whatever :/xlabel said.

Not ported: :image, :nsum, :database, :read, :write and :add, the database being the
task's, and one spectrum at a time, so that j, k and o stay on it.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import contextlib
import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import icfitsession, iraffmt
from .gtools import DRAWN, GTools
from .identify import Feature, Identify, find_peaks

f32 = np.float32
PAN, ZOOM = 1, 2
TICK, GAP = 0.03, 0.02        # of the screen's height

#: The keys, as the panel and `?` list them (identify.key).
HELP = [
    ("m", "mark the line at the cursor, then type its coordinate"),
    ("f", "fit the dispersion: icfit's keys, until its q"),
    ("l", "fit, then every line of the list the fit puts on a feature"),
    ("e", "every line of the list, with no fit"),
    ("y", "every peak there is"),
    ("d", "delete the nearest feature"),
    ("c", "centre the nearest feature again"),
    ("a", "all, for the c or d that follows"),
    ("u", "the user coordinate of this feature"),
    ("t", "move this feature to the cursor"),
    ("v", "the weight of this feature"),
    (".", "go to the nearest feature"),
    ("+", "the next feature"),
    ("n", "the next feature"),
    ("-", "the previous feature"),
    (" ", "this feature"),
    ("g", "a shift to the user coordinates"),
    ("s", "a shift: the cursor is at the coordinate typed"),
    ("x", "a shift by correlating the features"),
    ("b", "identify automatically"),
    ("i", "start again: no features, no fit"),
    ("z", "zoom on this feature"),
    ("p", "pan: the whole spectrum"),
    ("j", "the previous line or aperture of the image"),
    ("k", "the next line or aperture of the image"),
    ("o", "the line or aperture typed"),
    ("r", "redraw"),
    ("w", "window, then a key: a b c d e f g j k l m n p r t u x y z ?"),
    ("?", "these keys"),
    (":", "show, features, read, add, write, nsum, match, minsep, zwidth, labels, fwidth, "
          "ftype, threshold"),
    ("q", "finish"),
]

_CMDS = ("show", "features", "image", "nsum", "database", "read", "write", "add", "coordlist",
         "match", "maxfeatures", "minsep", "zwidth", "labels", "fwidth", "ftype", "cradius",
         "threshold")
_LABELS = ("none", "index", "pixel", "coord", "user", "both")
_FTYPES = ("emission", "absorption")
_FIT_AXES = {"h": ("y", "x"), "i": ("y", "v"), "j": ("y", "r"), "k": ("y", "d"),
             "l": ("y", "n")}


def _strdic(word: str, words) -> str | None:
    if word in words:
        return word
    found = [w for w in words if w.startswith(word)]
    return found[0] if len(found) == 1 else None


def _g(value) -> str:
    return iraffmt.g(float(f32(value)))


def _d(value, width: int = 10) -> str:
    """%10.8g of a double, INDEF for none."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "INDEF".rjust(width)
    return iraffmt.printf(float(value), "g", 8, width=width, double=True).rjust(width)


def _skipped(text: str) -> str:
    """id_label: without the spaces and quotes around it."""
    return text.strip(" \t\"'")


@dataclass
class Tick:
    """id_mark: over a line's peak, or under a dip, in the screen's fractions."""
    x: float
    y: float
    upward: bool
    label: str = ""


@dataclass
class Graph:
    """What id_graph draws, on the screen: under :/transpose x and y are the
    spectrum's y and x, but the ticks stay at the features' coordinates."""
    title: str
    xlabel: str
    window: tuple[float, float, float, float]
    x: np.ndarray
    y: np.ndarray
    ticks: list[Tick] = field(default_factory=list)
    ylabel: str = ""
    kind: str = "line"              #: line, mark or histogram, as gtools' type
    mark: str = "plus"
    size: tuple[float, float] = (2.0, 2.0)
    logs: tuple[bool, bool] = (False, False)    #: which of the screen's axes are elog


@dataclass
class Lines:
    """The lines of the image a session moves among with j, k and o: an image's
    lines or columns (`kind` "line" or "column"), nsum at a time, or a multispec
    image's apertures ("ms", their numbers in `numbers`), one at a time.
    `load(line, nsum)` gives a line's spectrum and section (" - Ap 2", "[*,61]");
    `read(image, section, database)` a database record, a Reference, or None;
    `write(image, section, saved, database)` writes one."""

    maxline: tuple[int, int]
    nsum: tuple[int, int]
    line: tuple[int, int]
    load: Callable
    read: Callable
    kind: str = "line"
    numbers: list[int] | None = None
    write: Callable | None = None


@dataclass
class Saved:
    """id_sid: what identify keeps of a line it has left."""

    line: tuple[int, int]
    section: str
    data: np.ndarray
    features: list
    fit: object
    shift: float
    spec: object
    current: int
    newfeatures: bool
    newdb: bool


class IdentifySession:
    """id_identify on a spectrum, and on the other lines of its image."""

    def __init__(self, ident: Identify, image: str, title: str = "", coordlist: str = "",
                 database: str = "", units: str = "angstroms", zwidth: float = 100.0,
                 autowrite: bool = False, crval=None, cdelt=None, section: str = "",
                 lines: Lines | None = None, reidentify: bool = False) -> None:
        self.ident = ident
        self.image, self.coordlist, self.database = image, coordlist, database
        self.section = section
        #: ID_GT, the task's for as long as it runs.
        self.gt = GTools()
        self.gt.set_type("line")
        self.pending = ""               #: "window" after w, "again" after w e
        self._corner: tuple[float, float] = (0.0, 0.0)
        self.title = title or f"identify {image}{section}"
        self.units = units
        self.zwidth = zwidth
        self.autowrite = autowrite
        self.crval, self.cdelt = crval, cdelt
        self.labels = "none"
        self.gtype = PAN
        self.current = 1 if ident.features else 0
        if reidentify and 1 <= ident.current <= len(ident.features):
            self.current = ident.current      # ID_CURRENT, as the last spectrum left it
        self.last = self.current
        self.all = 0
        self.newgraph = True
        self.newcv = False
        self.newdb = False
        self.refit = False
        self.shift = 0.0
        self.stdout = self.stderr = ""
        self._new = ""
        self.state = "keys"
        self.prompt = ""
        self.done = False
        self.write = False
        self.sub: icfitsession.IcfitSession | None = None
        self._cursor = (0.0, 0.0)
        self.lines = lines
        self.line = lines.line if lines is not None else (1, 1)
        self.nsum = lines.nsum if lines is not None else (1, 1)
        self._newline = self.line
        self.saved: dict[tuple[int, int], Saved] = {}
        self.written: list[Saved] = []          # what :write wrote, in order
        #: reidentify.x's loop: the spectrum reidentify has in hand, which it writes
        #: itself; no other line, no other image, and q asks nothing.
        self.reidentify = reidentify
        if not reidentify:
            ident.newfeatures = False
        if lines is not None:
            self._read_record(verbose=True)
        self._labels()

    @property
    def features(self) -> list[Feature]:
        return self.ident.features

    @property
    def title(self) -> str:
        """GTTITLE, which id_gdata sets for each spectrum."""
        return self.gt.title

    @title.setter
    def title(self, text: str) -> None:
        self.gt.title = text

    def _labels(self) -> None:
        """What id_fitdata sets the x label to: the spectrum's, or the units'."""
        fitted = self.ident.fit is not None
        self.gt.xlabel = "Wavelength" if fitted else ""
        self.gt.xunits = self.units if fitted else ""

    # ------------------------------------------------------------- the output

    def _print(self, text: str) -> None:
        self.stdout += text
        self._new += text

    def _eprint(self, text: str) -> None:
        self.stderr += text
        self._new += text

    def take(self) -> str:
        """What was printed since the last time, on either stream."""
        new, self._new = self._new, ""
        if self.sub is not None:
            new += self.sub.take()
        return new

    def _ask(self, state: str, prompt: str) -> None:
        self.state, self.prompt = state, prompt
        self._print(prompt)

    def _feature(self, feature: Feature, end: str) -> str:
        """The line a feature is printed on, or asked about."""
        pix = iraffmt.printf(float(feature.pix), "f", 2, width=10).rjust(10)
        if end == "status":
            return f"{pix} {_d(feature.fit)} {_d(feature.user)} {feature.label}\n"
        return f"{pix} {_d(feature.fit)} ({_d(feature.user)} {feature.label}): "

    # --------------------------------------------------------------- the keys

    def key(self, key: str, x: float = 0.0, y: float = 0.0, rest: str = "") -> None:
        """One key at the cursor, x in the graph's coordinate."""
        if self.sub is not None:
            self.sub.key(key, x, y, rest=rest)
            if self.sub.done:
                self.fit_finished()
            return
        if self.state != "keys":
            return
        if self.pending:
            self._window_key(key, x, y)
            return
        self._cursor = (x, y)
        print_it = True
        if self.all != 0:
            self.all = (self.all + 1) % 3
        ident = self.ident
        if key == "?":
            self._print("".join(f"{k}\t{what}\n" for k, what in HELP))
        elif key == ":":
            print_it = self._colon(rest)
        elif key == " ":
            pass
        elif key == ".":
            if not self.features:
                self._beep()
            else:
                self._nearest(x)
        elif key == "-":
            if self.current == 1:
                self._beep()
            else:
                self.current -= 1
        elif key in "+n":
            if self.current == len(self.features):
                self._beep()
            else:
                self.current += 1
        elif key == "a":
            self.all = 1
        elif key == "b":
            print_it = self._automatic()
        elif key == "c":
            print_it = self._centre(x)
        elif key == "d":
            self._delete(x)
        elif key == "e":
            self._linelist()
            if ident.newfeatures:
                self.newgraph = True
        elif key == "f":
            if self._dofit(True):
                return
        elif key == "g":
            self._doshift(True)
            print_it = False
        elif key == "i":
            ident.fit, ident.shift = None, 0.0
            ident.features = []
            self.current = 0
            self.newcv = ident.newfeatures = self.newgraph = True
        elif key == "l":
            if len(self.features) >= 2:
                self._dofit(False)
            if self.newcv:
                self._fitdata()
                ident.fit_features()
                self.newcv = False
            ident.newfeatures = False
            self._linelist()
            if ident.newfeatures:
                self.refit = True
                self.newdb = True
        elif key == "m":
            self._mark(x)
            return
        elif key in "jko" and self.reidentify:
            self._print("Command not available in REIDENTIFY")
            print_it = False
        elif key == "j":
            self._newline = self._stepped(-1)
        elif key == "k":
            self._newline = self._stepped(1)
        elif key == "o":
            self._ask("line", f"Line/Column/Band ({self.line[0]} {self.line[1]}): ")
            return
        elif key == "p":
            if self.gtype != PAN:
                self.gtype, self.newgraph = PAN, True
        elif key == "q":
            self._finish()
            return
        elif key == "r":
            self.newgraph = True
        elif key == "s":
            self._ask("shift", f"User coordinate ({_d(f32(x))}): ")
            return
        elif key == "x":
            self._correlate()
            print_it = False
        elif key == "t":
            if self.current < 1:
                self._beep()
            else:
                feature = self.features[self.current - 1]
                feature.pix = ident.fit_to_pix(x)
                feature.fit = ident.fitpt(feature.pix)
                ident.newfeatures = True
        elif key == "u":
            if not self.features:
                self._beep()
            else:
                self._ask("user", self._feature(self.features[self.current - 1], "ask"))
                return
        elif key == "v":
            if not self.features:
                self._beep()
            else:
                # With none current IRAF reads a weight it has not got, and sets it.
                now = int(self.features[self.current - 1].wts) if self.current > 0 else 0
                self._ask("weight", f"Weight ({now}): ")
                return
        elif key == "y":
            self._peaks()
        elif key == "z":
            if not self.features:
                self._beep()
            else:
                if self.gtype == PAN:
                    self.newgraph = True
                self.gtype = ZOOM
                self._nearest(x)
        elif key == "w":
            self.pending = "window"
            self._print("window:")
            return
        else:
            self._beep()
        self._tail(print_it)

    def _window_key(self, key: str, x: float, y: float) -> None:
        """gt_window: the key after w, and then what the loop does after any."""
        if self.pending == "window" and key == "e":
            self.pending, self._corner = "again", (x, y)
            self._print("again:")
            return
        if self.pending == "again":
            redraw = self.gt.window2(self._corner, (x, y), self._window)
        else:
            said, redraw = self.gt.window1(key, x, y, self._window)
            self._print(said)
        self.pending = ""
        self.newgraph = self.newgraph or redraw
        self._tail(True)

    def answer(self, text: str | None) -> None:
        """What a prompt read: None at the end of the input."""
        if self.sub is not None:
            self.sub.answer(text)
            return
        state, self.state, self.prompt = self.state, "keys", ""
        print_it = True
        words = (text or "").split()
        if state == "mark" or state == "user":
            feature = self.features[self.current - 1]
            value = _number(words[0]) if words else None
            count = 0 if value is None else min(len(words), 2)
            if count > 0:
                if state == "mark":
                    feature.user, label = self.ident.match_line(value, self.ident.match)
                    feature.label = label or ""
                else:
                    feature.user = value
                    self.ident.newfeatures = True
            if count > 1:
                feature.label = _skipped(text.split(None, 1)[1])
        elif state == "weight":
            if words and _number(words[0]) is not None:
                if self.current > 0:
                    self.features[self.current - 1].wts = float(int(_number(words[0])))
                self.ident.newfeatures = True
        elif state == "line":
            # gargi: a word that is not a number ends the scan there
            maxline = self.lines.maxline if self.lines is not None else (1, 1)
            values = []
            for word in words[:2]:
                value = _number(word)
                if value is None:
                    break
                values.append(int(value))
            first = list(self._newline)
            if values:
                if not 1 <= values[0] <= maxline[0]:
                    self._beep()
                else:
                    first[0] = values[0]
                    if len(values) == 2:
                        if not 1 <= values[1] <= maxline[1]:
                            self._beep()
                        else:
                            first[1] = values[1]
            self._newline = (first[0], first[1])
        elif state == "shift":
            if text is None:
                self.shift = 0.0
            else:
                value = _number(words[0]) if len(words) == 1 else None
                if value is not None:
                    self.shift = float(f32(self._cursor[0])) - value
            print_it = False
            self._recentre()
        elif state == "write":
            self.write = (not words or words[0].lower().startswith(("y", "t"))
                          or words[0] == "+")
            self.done = True
            return
        self._tail(print_it)

    def _linelist(self) -> None:
        """id_linelist: the feature left current is the one its count says, so with
        none added, none."""
        ident = self.ident
        ident.current = -1
        ident.locate()
        self.current = ident.current + 1

    def _beep(self) -> None:
        self._print("\a")

    def _nearest(self, x: float) -> None:
        """id_nearest: the first of two equally near."""
        if not self.features:
            self.current = 0
            return
        delta = [abs(f.fit - x) for f in self.features]
        self.current = delta.index(min(delta)) + 1

    def _mark(self, x: float) -> None:
        ident = self.ident
        pix = ident.center(ident.fit_to_pix(x))
        if pix is None:
            self._print("Center not found: check cursor position")
            if ident.threshold > 0:
                self._print(" and threshold value")
            self._beep()
            self._tail(False)
            return
        fit = ident.fitpt(pix)
        index = ident.newfeature(pix, fit, fit, 1.0)
        self.current = index + 1
        feature = self.features[index]
        feature.ftype = ident.ftype
        feature.fwidth = ident.fwidth
        feature.user, label = ident.match_line(feature.fit, ident.match)
        feature.label = label or ""
        self._ask("mark", self._feature(feature, "ask"))

    def _centre(self, x: float) -> bool:
        ident = self.ident
        if self.all != 0:
            for feature in list(self.features):
                feature.fwidth = ident.fwidth
                pix = ident.center(feature.pix)
                if pix is None:
                    self.features.remove(feature)
                    self.current = min(self.current, len(self.features))
                else:
                    feature.pix, feature.fit = pix, ident.fitpt(pix)
            ident.newfeatures = True
            return True
        if not self.features:
            self._beep()
            return True
        self._nearest(x)
        feature = self.features[self.current - 1]
        pix = ident.center(feature.pix)
        if pix is None:
            self._print("Centering failed\n")
            return False
        feature.pix, feature.fwidth = pix, ident.fwidth
        feature.fit = ident.fitpt(pix)
        ident.newfeatures = True
        return True

    def _delete(self, x: float) -> None:
        if self.all != 0:
            self.ident.features = []
            self.current = 0
            self.ident.newfeatures = self.newgraph = True
        elif not self.features:
            self._beep()
        else:
            self._nearest(x)
            del self.features[self.current - 1]
            self.ident.newfeatures = True
            self.current = min(len(self.features), self.current)
            self.last = 0

    def _peaks(self) -> None:
        ident = self.ident
        peaks = find_peaks(ident.data, 0.0, int(ident.minsep), 0, ident.maxfeatures, 0.0)
        peaks = [float(p) for p in peaks if all(abs(float(p) - f.pix) >= ident.minsep
                                                for f in self.features)]
        for peak in peaks:
            pix = ident.center(peak)
            if pix is None:
                continue
            fit = ident.fitpt(pix)
            user, label = ident.match_line(fit, ident.match)
            index = ident.newfeature(pix, fit, user, 1.0, label or "")
            self.current = index + 1
            self.features[index].ftype = ident.ftype
            self.features[index].fwidth = ident.fwidth

    def _automatic(self) -> bool:
        from .identify import autoidentify

        found = autoidentify(self.ident, self.crval, self.cdelt)
        if not found.found:
            self._print("No solution found\n")
            return False
        self.current = 1 if self.features else 0
        self.newcv = self.ident.newfeatures = self.newgraph = True
        return True

    # ---------------------------------------------------------------- the fit

    def _dofit(self, interactive: bool) -> bool:
        """id_dofit. True if the keys have gone to icfit."""
        ident = self.ident
        use = [f for f in self.features if f.pix is not None and f.user is not None]
        if not self.features or len(use) <= 1:
            if ident.fit is not None:
                ident.fit, ident.shift = None, 0.0
                self.newgraph = self.newcv = True
            return False
        if not interactive:
            ident.dofit()
            self.newcv = self.newgraph = True
            return False
        self._fitted = use
        self.sub = icfitsession.IcfitSession(
            [f.pix for f in use], [f.user for f in use], ident.spec,
            [max(1.0, f.wts) for f in use], "", "Feature positions (pixels)", "Wavelength",
            "mark", (1.0, float(ident.npts)), self.units, _FIT_AXES, "j")
        self.sub.real = False           # icg_fitd: v, x and y read with gargd
        self._print("")
        return True

    def fit_finished(self) -> None:
        """What id_dofit does when icfit's q comes back."""
        ident, sub, use = self.ident, self.sub, self._fitted
        self._print(sub.take())
        self.sub = None
        fit = sub.result()
        ident.spec = sub.spec
        n = len(use)
        rejected = None
        if fit is not None and fit.binned is not None and fit.binned[3].any() \
                and fit.binned[0].size == n:
            rejected = fit.binned[3]
        keep = []
        k = 0
        for feature in self.features:
            if feature.pix is None or feature.user is None:
                keep.append(feature)
                continue
            if k < n and sub.wts[k] != 0:
                feature.pix = float(sub.x[k])
                feature.user = float(sub.y[k])
                feature.wts = float(sub.wts[k])
                if rejected is not None and rejected[k]:
                    feature.wts = 0.0
                keep.append(feature)
            k += 1
        ident.features = keep
        ident.fit = fit if fit is not None and fit.model is not None else None
        ident.shift = 0.0
        self.current = min(self.current, len(keep))
        self.newcv = self.newgraph = True
        self._tail(True)

    def _doshift(self, say: bool) -> None:
        """id_doshift."""
        ident = self.ident
        total, square, count = 0.0, 0.0, 0
        for feature in self.features:
            if feature.user is None or feature.wts == 0:
                continue
            delta = feature.user - ident.fitpt(feature.pix)
            total += delta
            square += delta * delta
            count += 1
        if not count:
            return
        total /= count
        square /= count
        if say:
            spread = math.sqrt(max(square - total * total, 0.0))
            self._print(f"{self.image}{self.section}: Coordinate shift={_five(total)}, "
                        f"rms={_five(spread)}, npts={count:3d}\n")
        ident.shift += total
        self.newcv = self.newgraph = True

    def _correlate(self) -> None:
        """x: id_shift(id, -1, -0.05) -- identify automatically near the solution
        there is, and the shift is how far the new one lies from it."""
        from .identify import shift_search

        shift = shift_search(self.ident, -1.0, -0.05)
        if shift is None:
            self._print("No solution found\n")
            self._beep()
            return
        self.shift = shift
        self._recentre()

    def _recentre(self) -> None:
        """The features shifted and centred again: the end of s and x."""
        ident = self.ident
        self.newcv = ident.newfeatures = self.newgraph = True
        shift = self.shift
        if not self.features:
            self._print(f"User coordinate shift={_five(shift)}\n")
            ident.shift += shift
            return
        before = len(self.features)
        pix_shift = z_shift = 0.0
        kept = []
        for i, feature in enumerate(self.features, 1):
            width = ident.fwidth
            ident.fwidth = feature.fwidth or width
            pix = ident.center(ident.fit_to_pix(feature.fit + shift))
            ident.fwidth = width
            if pix is None:
                if self.current == i:
                    self.current = i + 1
                continue
            fit = ident.fitpt(pix)
            pix_shift += pix - feature.pix
            if feature.fit != 0:
                z_shift += (fit - feature.fit) / feature.fit
            feature.pix = pix
            kept.append(feature)
            if self.current == i:
                self.current = len(kept)
        ident.features = kept
        self.current = min(self.current, len(kept))
        if not kept:
            self._print(f"User coordinate shift={_five(shift)}, "
                        "No features found during recentering\n")
            ident.shift += shift
            return
        held = ident.shift
        self._doshift(False)
        ident.fit_features()
        n = len(kept)
        self._print(f"Recentered={n}/{before}, pixel shift="
                    f"{iraffmt.printf(pix_shift / n, 'f', 2)}, "
                    f"user shift={_five(held - ident.shift)}, "
                    f"z={_seven3(z_shift / n)}, rms={_five(ident.rms())}\n")

    # ------------------------------------------------------------ other lines

    def _stepped(self, sign: int) -> tuple[int, int]:
        """j and k: nsum lines back or on, round to the other end and to the next
        band, as ididentify.x wraps them."""
        maxline = self.lines.maxline if self.lines is not None else (1, 1)
        nsum = self.nsum
        first, second = self.line[0] + sign * nsum[0], self.line[1]
        if not 1 <= first <= maxline[0]:
            first -= sign * maxline[0]
            second = self.line[1] + sign * nsum[1]
            if not 1 <= second <= maxline[1]:
                second -= sign * maxline[1]
        return first, second

    def _save(self) -> None:
        """id_saveap: this line kept, in its place if it was kept before."""
        ident = self.ident
        self.saved[self.line] = Saved(
            self.line, self.section, ident.data.copy(), copy.deepcopy(ident.features),
            ident.fit, ident.shift, copy.deepcopy(ident.spec), self.current,
            ident.newfeatures, self.newdb)

    def _move(self, line: tuple[int, int]) -> None:
        """A new line: this one kept, that one's spectrum read; what identify kept of
        it back, else the database's record, else the features carried over."""
        ident = self.ident
        self._save()
        self.line = self._newline = line
        data, self.section = self.lines.load(line, self.nsum)
        ident.set_data(data)
        self.title = f"identify {self.image}{self.section}"
        kept = self.saved.get(line)
        if kept is not None:
            # id_gid keeps the features in hand when it kept none
            if kept.features:
                ident.features = copy.deepcopy(kept.features)
                ident.newfeatures = kept.newfeatures
                self.current = kept.current
                self.newdb = kept.newdb
            ident.shift = kept.shift
            if kept.fit is not None:
                ident.fit = kept.fit
                ident.spec = copy.deepcopy(kept.spec)
                self.newdb = kept.newdb
        else:
            self._read_record(verbose=False)
        self.newcv = self.newgraph = True

    def _ap(self) -> tuple[int, int]:
        """ID_AP: a multispec line's aperture number, else the line itself."""
        if self.lines is not None and self.lines.kind == "ms" and self.lines.numbers:
            return self.lines.numbers[self.line[0] - 1], self.line[1]
        return self.line

    def _section_for(self, name: str, ap: tuple[int, int]) -> str:
        """id_dbsection."""
        kind = self.lines.kind if self.lines is not None else ""
        if kind == "ms":
            return f" - Ap {ap[0]}"
        if name != self.image or not kind:
            return ""
        return f"[*,{ap[0]}]" if kind == "line" else f"[{ap[0]},*]"

    def _dbread(self, name: str, ap: tuple[int, int], add: bool) -> None:
        """:read and :add -- id_dbread, verbose: the record's features in place of
        these, or among them in order of pixel; its solution with them unless added."""
        ident = self.ident
        section = self._section_for(name, ap)
        stored = None if self.lines is None else self.lines.read(name, section, self.database)
        if stored is None:
            # Said by the colon command and again by the loop, whose delayed error
            # message is the same one, as IRAF says it.
            self._eprint(f"Warning: Entry not found: identify {name}{section}\n" * 2)
            return
        if add:
            for feature in copy.deepcopy(stored.features):
                k = len(ident.features)
                while k > 0 and feature.pix < ident.features[k - 1].pix:
                    k -= 1
                ident.features.insert(k, feature)
        else:
            ident.features = copy.deepcopy(stored.features)
            ident.shift = stored.shift
            if stored.fit is not None:
                ident.fit = stored.fit
                ident.spec = copy.deepcopy(stored.spec)
                self.newcv = True
        ident.newfeatures = True
        self.current = 1 if ident.features else 0
        self.newgraph = True
        self._print(f"identify {name}{section}\n")

    def _dbwrite(self, name: str, ap: tuple[int, int]) -> None:
        """:write -- id_dbwrite, verbose, now."""
        section = self._section_for(name, ap)
        ident = self.ident
        kept = Saved(self.line, section, ident.data.copy(), copy.deepcopy(ident.features),
                     ident.fit, ident.shift, copy.deepcopy(ident.spec), self.current,
                     ident.newfeatures, True)
        if self.lines is not None and self.lines.write is not None:
            self.lines.write(name, section, kept, self.database)
        self.written.append(kept)
        self.ident.newfeatures = self.newcv = self.newdb = False
        self._print(f"identify {name}{section}\n")

    def _read_record(self, verbose: bool) -> None:
        """id_dbread of this line, as identify starts and moves: its record found,
        nothing is new; not found, what is in hand is."""
        ident = self.ident
        stored = self.lines.read(self.image, self.section, self.database)
        if stored is None:
            if ident.features or ident.fit is not None:
                self.newdb = True
            return
        ident.features = copy.deepcopy(stored.features)
        ident.fit, ident.shift = stored.fit, stored.shift
        ident.spec = copy.deepcopy(stored.spec)
        # Read, not changed: identify clears ID_NEWFEATURES after id_dbread, as it
        # starts and when it moves, so the record is not written back unless a key
        # changes it.
        ident.newfeatures = False
        self.current = 1 if ident.features else 0
        self.newdb = False
        self.newgraph = True
        if verbose:
            self._print(f"identify {self.image}{self.section}\n")
        with contextlib.suppress(ValueError):
            ident.update_fitdata()

    def records(self) -> list[Saved]:
        """What q writes, in its order: this line, then each line kept since and
        changed, the last kept first."""
        self._save()
        here = self.saved[self.line]
        out = [here] if here.newdb else []
        out += [kept for kept in reversed(list(self.saved.values()))
                if kept.newdb and kept.line != self.line]
        return out

    def _fitdata(self) -> None:
        try:
            self.ident.update_fitdata()
            self._error = ""
        except ValueError as error:
            self._error = str(error)
            return
        self._labels()

    # ---------------------------------------------------------- after each key

    def _tail(self, print_it: bool) -> None:
        """What the loop does after a key, in its order."""
        ident = self.ident
        if ident.newfeatures or self.newcv:
            self.newdb = True
        self._error = ""
        if self._newline != self.line:
            self._move(self._newline)
        if self.refit:
            self._dofit(False)
            self.refit = False
        if self.newcv:
            self._fitdata()
            ident.fit_features()
            self.newcv = False
        if self.gtype == ZOOM and self.last != self.current:
            self.newgraph = True
        for feature in self.features:
            if not feature.fwidth:
                feature.fwidth = ident.fwidth
            if not feature.ftype:
                feature.ftype = ident.ftype
        if self.current > len(self.features):
            self.current = len(self.features)
        if self.current > 0 and print_it and not self._error:
            self._pending_status = self._feature(self.features[self.current - 1], "status")
        else:
            self._pending_status = ""
        self._say_after_graph()

    def _say_after_graph(self) -> None:
        if self.newgraph and self._batch_graph:
            self.graph()
        if self._pending_status:
            self._print(self._pending_status)
        if self._error:
            self._eprint(f"Warning: {self._error}\n")
        self.last = self.current

    _batch_graph = False
    _pending_status = ""
    _error = ""

    def _finish(self) -> None:
        if self.reidentify:
            self.done = True
            return
        if not self.newdb and not any(kept.newdb for kept in self.saved.values()):
            self.done = True
            return
        if self.autowrite:
            self.write = self.done = True
            return
        self._ask("write", "Write feature data to the database (yes)? ")

    # ------------------------------------------------------- the colon commands

    def _colon(self, text: str) -> bool:
        """id_colon: whether the feature is to be printed after."""
        words = text.split()
        if not words:
            return True
        if text.lstrip().startswith("/"):
            out, err, redraw = self.gt.colon(text.lstrip(), self._window)
            self._print(out)
            self._eprint(err)
            self.newgraph = self.newgraph or redraw
            return True
        command = _strdic(words[0], _CMDS)
        value = words[1] if len(words) > 1 else None
        ident = self.ident
        if command == "show":
            self._output(self._show_text(), value)
        elif command == "features":
            self._output(self._features_text(), value)
        elif command == "image":
            if value is None:
                self._print(f"image {self.image}\n")
                return False
            if self.reidentify:
                self._print("Can't change image in REIDENTIFY")
                return False
            self._print(":image is not ported: run identify on the other image\n")
            return False
        elif command == "nsum":
            number = _number(value) if value is not None else None
            if number is None:
                self._print(f"nsum {self.nsum[0]} {self.nsum[1]}\n")
                return False
            second = _number(words[2]) if len(words) > 2 else None
            self.nsum = (int(number), int(second) if second is not None else self.nsum[1])
        elif command == "database":
            if value is None:
                self._print(f"database {self.database}\n")
                return False
            self.database = value
            self.newdb = True
        elif command in ("read", "write", "add"):
            name, ap = self.image, list(self._ap())
            if value is not None:
                name = value
                for i, word in enumerate(words[2:4]):
                    number = _number(word)
                    if number is None:
                        break
                    ap[i] = int(number)
            if command == "write":
                self._dbwrite(name, (ap[0], ap[1]))
            else:
                self._dbread(name, (ap[0], ap[1]), command == "add")
            return False
        elif command == "coordlist":
            if value is None:
                self._print(f"coordlist {self.coordlist}\n")
                return False
            from .identify import read_linelist

            ident.linelist = read_linelist(value)
            self.coordlist = value
        elif command in ("match", "minsep", "fwidth", "cradius", "threshold"):
            number = _number(value) if value is not None else None
            if number is None:
                self._print(f"{command} {_g(getattr(ident, command))}\n")
                return False
            setattr(ident, command, number)
        elif command == "maxfeatures":
            number = _number(value) if value is not None else None
            if number is None:
                self._print(f"maxfeatures {ident.maxfeatures}\n")
                return False
            ident.maxfeatures = int(number)
        elif command == "zwidth":
            number = _number(value) if value is not None else None
            if number is None:
                self._print(f"zwidth {_g(self.zwidth)}\n")
                return False
            self.zwidth = number
            if self.gtype == ZOOM:
                self.newgraph = True
        elif command == "labels":
            if value is None:
                self._print(f"labels {self.labels}\n")
                return False
            self.labels = _strdic(value, _LABELS) or "none"
        elif command == "ftype":
            if value is None:
                self._print(f"ftype {ident.ftype}\n")
                return False
            ident.ftype = _strdic(value, _FTYPES) or ident.ftype
        else:
            self._print("Unrecognized or ambiguous command\a")
            return False
        return True

    def _output(self, text: str, file: str | None) -> None:
        if file is None:
            self._print(text)
            return
        try:
            with open(file, "a", encoding="utf-8") as out:
                out.write(text)
        except OSError as error:
            self._eprint(f"Warning: {error}\n")

    def _show_text(self) -> str:
        """id_show."""
        ident = self.ident
        lines = ["harwood", f"image {self.image}", "nsum 1",
                 f"ftype {ident.ftype}", f"labels {self.labels}",
                 f"maxfeatures {ident.maxfeatures}", f"match {_g(ident.match)}",
                 f"zwidth {_g(self.zwidth)}", f"fwidth {_g(ident.fwidth)}",
                 f"database {self.database}", f"coordlist {self.coordlist}",
                 f"cradius {_g(ident.cradius)}", f"threshold {_g(ident.threshold)}",
                 f"minsep {_g(ident.minsep)}"]
        if ident.fit is not None:
            first = iraffmt.printf(float(ident.fitdata[0]), "g", 8, double=True)
            step = (ident.fitdata[-1] - ident.fitdata[0]) / (ident.npts - 1)
            step = iraffmt.printf(float(step), "g", 8, double=True)
            lines += [f"function = {ident.spec.function}", f"order = {ident.spec.order}",
                      f"Fit at first pixel = {first}",
                      f"Average fit interval = {step}"]
        return "\n".join(lines) + "\n"

    def _features_text(self) -> str:
        """id_log."""
        if not self.features:
            return ""
        lines = ["", time.strftime("%a %H:%M:%S %d-%b-%Y"),
                 f"Features identified in image {self.image}.",
                 f"   {'Pixel':>8} {'Fit':>10} {'User':>10} {'Residual':>10} {'Fwidth':>6} "
                 f"{'Wt':>2} Label"]
        total, count = 0.0, 0
        for i, f in enumerate(self.features, 1):
            resid = None
            if f.user is not None:
                resid = f.fit - f.user
                if f.wts > 0:
                    total += resid * resid
                    count += 1
            pix = iraffmt.printf(float(f.pix), "f", 2, width=8).rjust(8)
            width = iraffmt.printf(float(f32(f.fwidth or self.ident.fwidth)), "f", 2, width=6)
            lines.append(f"{i:2d} {pix} {_d(f.fit)} {_d(f.user)} {_d(resid)} {width.rjust(6)} "
                         f"{int(f.wts):2d} {f.label}")
        if count > 1:
            rms = iraffmt.printf(math.sqrt(total / count), "g", 8, double=True)
            lines.append(f"RMS = {rms}")
        return "\n".join(lines) + "\n"

    # -------------------------------------------------------------- the graph

    _window = (0.0, 1.0, 0.0, 1.0)     #: the window as last drawn, ggwind's

    def graph(self) -> Graph:
        """id_graph1 and id_graph2, and it is drawn."""
        ident, gt = self.ident, self.gt
        x = np.asarray(ident.fitdata, dtype=f32).astype(float)
        y = np.asarray(ident.data, dtype=float)
        if self.gtype == ZOOM and self.current > 0:
            centre = float(f32(self.features[self.current - 1].fit))
            half = float(f32(self.zwidth)) / 2
            x0, x1 = centre - half, centre + half
            inside = y[(x >= x0) & (x <= x1)]
            ymin, ymax = (0.0, 1.0)
            if inside.size:
                ymin, ymax = float(inside.min()), float(inside.max())
            dy = ymax - ymin
            window = (x0, x1, ymin - 0.2 * dy, ymax + 0.2 * dy)
        else:
            xmin, xmax = float(min(x[0], x[-1])), float(max(x[0], x[-1]))
            low, high = gt.getr("xmin"), gt.getr("xmax")
            if (low is not None and xmax < low) or (high is not None and xmin > high):
                gt.setr("xmin", None)
                gt.setr("xmax", None)
            ymin, ymax = float(y.min()), float(y.max())
            dy = float(f32(ymax) - f32(ymin))
            window = gt.swind((xmin, xmax), (float(f32(ymin) - f32(0.2) * f32(dy)),
                                             float(f32(ymax) + f32(0.2) * f32(dy))))
        self._window = window
        ticks = []
        for i, feature in enumerate(self.features, 1):
            if not min(window[:2]) <= feature.fit <= max(window[:2]):
                continue
            pix = max(1, min(int(feature.pix), ident.npts - 1))
            pair = float(y[pix - 1]), float(y[pix])
            upward = (feature.ftype or ident.ftype) == "emission"
            ticks.append(Tick(float(feature.fit), max(pair) if upward else min(pair), upward,
                              self._mark_label(i, feature)))
        self.newgraph = False
        labels = (gt.label("x"), gt.label("y"))
        if gt.transpose:
            labels = labels[::-1]
        sx, sy = gt.on_screen(x, y)
        return Graph("\n".join(gt.title_block()), labels[0], window, sx, sy, ticks, labels[1],
                     DRAWN.get(gt.type, "mark"), gt.mark, (gt.xsize, gt.ysize), gt.logs())

    def _mark_label(self, number: int, feature: Feature) -> str:
        if self.labels == "index":
            return str(number)
        if self.labels == "pixel":
            return iraffmt.printf(float(feature.pix), "f", 2)
        coordinate = (iraffmt.printf(float(feature.user), "f", 4)
                      if feature.user is not None else "")
        if self.labels == "coord":
            return coordinate
        if self.labels == "user":
            return feature.label
        if self.labels == "both":
            return f"{coordinate} {feature.label}".strip() if coordinate else feature.label
        return ""

    # ------------------------------------------------------------- a whole run

    def start(self) -> None:
        """The pass id_identify makes before the first key, as if r were pressed: the
        graph, and the feature in hand when a record was read."""
        self.key("r")

    def run_commands(self, cursor: str, answers: str = "") -> None:
        """A cursor file, `x y wcs key [command]` a line, and the lines the prompts
        read, as IRAF takes them from the terminal. Both go to icfit while it is on. Either
        may be an iterator of lines that sessions one after another share, as reidentify's
        do the cursor file."""
        lines = iter(cursor.splitlines()) if isinstance(cursor, str) else cursor
        replies = iter(answers.splitlines()) if isinstance(answers, str) else answers

        def next_cursor():
            for line in lines:
                words = line.split(None, 4)
                if len(words) >= 4 and not line.lstrip().startswith("#"):
                    return words
            return None

        self._batch_graph = True
        self.start()
        while not self.done:
            if self.sub is not None:
                if self.sub.newgraph:
                    self.sub.graph()
                if self.sub.state not in ("fit", "second"):
                    self.sub.answer(next(replies, None))
                    continue
            elif self.state != "keys":
                self.answer(next(replies, None))
                continue
            words = next_cursor()
            if words is None:
                self.pending = ""           # gt_window returns at the end of the file
                self.key("q")
                continue
            x, y, key = float(words[0]), float(words[1]), words[3].replace("\\040", " ")
            if key.startswith(":"):
                rest = key[1:] + " " + (words[4] if len(words) > 4 else "")
                self.key(":", rest=rest.strip())
            else:
                self.key(key, x, y)


def _number(text: str | None) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _five(value) -> str:
    """%5f."""
    return iraffmt.printf(float(value), "f", 6, width=5, double=True).rjust(5)


def _seven3(value) -> str:
    """%7.3g."""
    return iraffmt.printf(float(value), "g", 3, width=7, double=True).rjust(7)


