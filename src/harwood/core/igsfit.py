"""fitcoords by hand: IRAF's igsfit, noao/twodspec/longslit/transform/igsfit (igsfit.x,
igsgraph.x, igsnearest.x, igsdelete.x, igsundelete.x, igscolon.x, igsparams.x), as
IRAF 2.18 runs it.

The graph is of the feature coordinates fitcoords gathered (core/fitcoords.py): two
of x, y, z (the user coordinate), s (the surface there) and r (the residual), at
first the position along the dispersion against the residual. A point is a plus,
or a cross once deleted (weight zero). The keys:

    c  the values of the point nearest the cursor
    d  delete the point, or every point of its x, y or z (asked: p x y z)
    u  undelete the same way
    f  fit again, with the function and orders the colon commands set
    x  y  choose the graph's axes (asked: x y z s r) -- drawn at the next r
    z  zoom on the points of the nearest one's x, y or z (asked)
    n  l  the next or last set of points while zoomed;  p  all of them again
    r  redraw;  w  gtools' window;  ?  help;  q  quit;  I  interrupt
    :show :corners :function :xorder :yorder, and gtools' :/ commands

What follows from the code, kept:

* the residuals' mean and rms, in the title, are aavgr's -- every point, the
  deleted ones too, and the rms is the standard deviation (n - 1);
* the orders and the function are the common block's: what :xorder sets is the
  next fit's and the next image's, and the log prints it even when no f followed,
  beside the surface fitted before;
* x and y only choose the axes; nothing is drawn until r (or another redraw);
* the zoomed graph's window is the set's own extent, through gt_swind;
* the marks are drawn with gmark, not gt_plot: under :/transpose the window turns
  and the marks do not, so the points leave the graph;
* undeleting a point of the deletion list gives it weight 1;
* :show and :corners print the second corner evaluated at (xmax, xmin), as the
  log does; a :xorder or :yorder below 2 is refused, a bad command rings the bell
  with no new line.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from . import fitcoords as fc
from . import iraffmt
from .gtools import GTools, Scan, pagefile, strdic

f32 = np.float32
MAX_REAL = float(np.finfo(np.float32).max)
INDEFR = fc.INDEFR
#: fc_getcoords' labels, by column.
LABELS = {fc.X: "X (pixels)", fc.Y: "Y (pixels)", fc.Z: "User", fc.S: "Surface",
          fc.R: "Residuals"}
#: The answers to 'x', 'y' (all five) and 'z', 'd', 'u' (x y z), as columns.
COLUMNS = {"x": fc.X, "y": fc.Y, "z": fc.Z, "s": fc.S, "r": fc.R}
FUNCTIONS = ("chebyshev", "legendre")
CMDS = ("show", "function", "xorder", "yorder", "corners")
BELL = "\a\n"
PROMPTS = {
    "x": "Select abscissa (x, y, z, s, r): ",
    "y": "Select ordinate (x, y, z, s, r): ",
    "z": "Zoom type (x, y, z): ",
    "d": "Delete 'p'oint or constant 'x', 'y', or 'z': ",
    "u": "Undelete 'p'oint or constant 'x', 'y', or 'z': ",
}
HELP = "igsfit.key"
#: The keys, for the panel's guide.
KEYS = [
    ("c", "the values of the nearest point"),
    ("d", "delete it, or its x, y or z (then p x y z)"),
    ("u", "undelete the same way"),
    ("f", "fit again"),
    ("x", "the x axis (then x y z s r): drawn at r"),
    ("y", "the y axis (then x y z s r): drawn at r"),
    ("z", "zoom on the nearest one's x, y or z"),
    ("n l", "next or last set, zoomed"),
    ("p", "all the points again"),
    ("r", "redraw"),
    ("w", "window, then a key: a b c d e f g j k l m n p r t u x y z ?"),
    ("q", "quit"),
    (":", "show corners function xorder yorder, and /gtools"),
]


@dataclass
class Common:
    """igsfit.com: the function and orders, shared by every fit of one fitcoords."""
    function: str = "chebyshev"
    xorder: int = 6
    yorder: int = 6


@dataclass
class Frame:
    """What igs_graph draws: the window as drawn (x1 x2 y1 y2 on the screen), the
    viewport's width and height, the title block and labels, and the marks
    (kind, x, y) in the data's world; `over` the marks d and u put on top
    after, ("plus" or "cross", x, y, erased)."""
    window: tuple
    viewport: tuple
    title: list[str]
    xlabel: str
    ylabel: str
    marks: list[tuple[str, float, float]]
    over: list[tuple[str, float, float, bool]] = field(default_factory=list)
    logs: tuple[bool, bool] = (False, False)    #: which screen axes are elog


def aavg(values: np.ndarray) -> tuple[float, float]:
    """aavgr: the mean and the standard deviation in real, the sums in double one
    value after another; with one value the sigma is INDEF."""
    total = sumsq = 0.0
    for v in np.asarray(values, f32):
        v = float(v)
        total += v
        sumsq += v * v
    n = len(values)
    if n == 0:
        return INDEFR, INDEFR
    if n == 1:
        return float(f32(total)), INDEFR
    temp = (sumsq - (total / n) * total) / (n - 1)
    return float(f32(total / n)), float(f32(math.sqrt(temp) if temp >= 0 else 0.0))


def _rg(value: float) -> str:
    return "INDEF" if value >= INDEFR else iraffmt.g(value)


class Session:
    """igs_fit on one set of coordinates: the fit made at once, then the keys, one
    read at a time (`key`). What IRAF prints goes to `out`, its error stream to
    `err`; each graph drawn is a Frame in `frames`. `done` once q is read."""

    def __init__(self, coords: fc.Coords, common: Common, fitname: str) -> None:
        self.coords = coords
        self.common = common
        pts = coords.points
        # fc_fitcoords: the position along the dispersion against the residual; with
        # a single feature (z = x + f(y), or y + f(x)), the position across it
        single = pts[fc.FEATURE, -1] == 1
        across, along = (fc.Y, fc.X) if coords.axis == 1 else (fc.X, fc.Y)
        self.axis = [across if single else along, fc.R]
        self.gt = GTools()
        self.gt.title = f"Fit User Coordinates to Image Coordinates for {fitname}"
        self.out = ""
        self.err = ""
        self.frames: list[Frame] = []
        self.ztype: int | None = None
        self.refpt = 0
        self.state = ""
        self.pending = 0            # the point d or u asked about
        self.first = (0.0, 0.0)     # w e's first corner
        self.done = False
        self.interrupted = False    # the panel's I
        self.scale = None
        self.window = (0.0, 1.0, 0.0, 1.0)
        self.mean = self.rms = INDEFR
        self.sf = None
        self._solve()
        self.wts = pts[fc.W].copy()

    # ------------------------------------------------------------ the fit

    @property
    def pts(self) -> np.ndarray:
        return self.coords.points

    def _solve(self) -> None:
        """igs_solve, xgsvector and asubr, aavgr and igs_params."""
        c = self.common
        self.sf, said = fc.fit(self.coords, c.function, c.xorder, c.yorder)
        self.err += "".join(text + "\n" for text in said)
        self.mean, self.rms = aavg(self.pts[fc.R])
        rms = "INDEF" if self.rms >= INDEFR else iraffmt.g(self.rms, decpl=4)
        self.gt.params = (f"Function = {c.function}, xorder = {c.xorder}, "
                          f"yorder = {c.yorder}, rms = {rms}")

    # ------------------------------------------------------------ the graph

    @property
    def viewport(self) -> tuple[float, float]:
        """glb_set_viewport on stdvdm: eleven characters either side, and above the
        graph a line of the system's and the title block's."""
        lines = int(self.gt.sysid) + len(self.gt.title_block())
        return 1.0 - 2 * 11 * 0.0125, 1.0 - 5 * 0.0294 - (lines + 1) * 0.0294

    def shown(self) -> np.ndarray:
        """The points drawn: all, or the zoomed set's."""
        pts = self.pts
        if self.ztype is None:
            return np.ones(pts.shape[1], bool)
        return pts[self.ztype] == pts[self.ztype, self.refpt]

    def graph(self) -> Frame:
        """igs_graph: gascale over every point, or gswind to the zoomed set's extent;
        then gt_swind, and the marks."""
        x, y = self.axis
        pts = self.pts
        gt = self.gt
        gt.xlabel, gt.ylabel = LABELS[x], LABELS[y]
        shown = self.shown()
        xs, ys = pts[x][shown], pts[y][shown]
        self.window = gt.swind((float(xs.min()), float(xs.max())),
                               (float(ys.min()), float(ys.max())))
        marks = [("cross" if pts[fc.W, i] == 0.0 else "plus", float(pts[x, i]),
                  float(pts[y, i])) for i in np.flatnonzero(shown)]
        frame = Frame(self.window, self.viewport, gt.title_block(),
                      gt.label("y" if gt.transpose else "x"),
                      gt.label("x" if gt.transpose else "y"), marks, logs=gt.logs())
        self.frames.append(frame)
        return frame

    def nearest(self, wx: float, wy: float, weights: str = "") -> int | None:
        """igs_nearest, and igs_nearestd ('kept') and igs_nearestu ('deleted'): in
        NDC through the window as drawn, among the points shown."""
        pts = self.pts
        use = self.shown()
        if weights == "kept":
            use = use & (pts[fc.W] != 0.0)
        elif weights == "deleted":
            use = use & (pts[fc.W] == 0.0)
        x, y = self.axis
        return self.gt.nearest(pts[x], pts[y], wx, wy, self.window, self.viewport,
                               scale=self.scale, use=use)

    def _remark(self, i: int, deleting: bool) -> None:
        pts = self.pts
        x, y = self.axis
        old, new = ("plus", "cross") if deleting else ("cross", "plus")
        frame = self.frames[-1] if self.frames else None
        if frame is not None:
            frame.over.append((old, float(pts[x, i]), float(pts[y, i]), True))
            frame.over.append((new, float(pts[x, i]), float(pts[y, i]), False))
        if deleting:
            pts[fc.W, i] = 0.0
        else:
            if self.wts[i] == 0:
                self.wts[i] = 1.0
            pts[fc.W, i] = self.wts[i]

    def _change(self, refpt: int, dtype: int, deleting: bool) -> None:
        """igs_delete and igs_undelete: the point, or every one (of the zoomed set)
        sharing its x, y or z."""
        pts = self.pts
        if dtype == 0:
            self._remark(refpt, deleting)
            return
        for i in range(pts.shape[1]):
            if self.ztype is not None and pts[self.ztype, i] != pts[self.ztype, refpt]:
                continue
            if pts[dtype, i] != pts[dtype, refpt]:
                continue
            self._remark(i, deleting)

    def _step(self, forward: bool) -> bool:
        """n and l: the set of the next larger, or smaller, value of the zoom's
        coordinate."""
        values = self.pts[self.ztype]
        zval = values[self.refpt]
        found, best = None, f32(MAX_REAL if forward else -MAX_REAL)
        for i, v in enumerate(values):
            if (v > zval and v < best) if forward else (v < zval and v > best):
                found, best = i, v
        if found is None:
            return False
        self.refpt = found
        return True

    # ---------------------------------------------------------- the colon

    def corners(self) -> list[tuple[float, float, float]]:
        """The four corners as :show prints them -- the second at (xmax, xmin)."""
        c = self.coords
        ev = fc.xgseval
        return [(c.xmin, c.ymin, ev(self.sf, c.xmin, c.ymin)),
                (c.xmax, c.ymin, ev(self.sf, c.xmax, c.xmin)),
                (c.xmin, c.ymax, ev(self.sf, c.xmin, c.ymax)),
                (c.xmax, c.ymax, ev(self.sf, c.xmax, c.ymax))]

    def colon(self, text: str) -> None:
        """igs_colon."""
        scan = Scan(text, self.gt.last_real)
        found = strdic(scan.wrd(), CMDS)
        name = CMDS[found - 1] if found else ""
        c = self.common
        d = fc._d
        if name == "show":
            (x1, y1, v1), (x2, y2, v2), (x3, y3, v3), (x4, y4, v4) = self.corners()
            g = iraffmt.g
            self.out += (f"function {c.function}\nxorder {c.xorder}\nyorder {c.yorder}\n"
                         "Fitted coordinates at the corners of the images:\n"
                         f"    ({d(x1)}, {d(y1)}) = {g(v1)}  ({d(x2)}, {d(y2)}) = {g(v2)}\n"
                         f"    ({d(x3)}, {d(y3)}) = {g(v3)}  ({d(x4)}, {d(y4)}) = {g(v4)}\n"
                         f"rms {_rg(self.rms)}\n")
        elif name == "function":
            word = scan.wrd()
            if scan.n == 1:
                self.out += f"function = {c.function}\n"
            elif strdic(word, FUNCTIONS):
                c.function = FUNCTIONS[strdic(word, FUNCTIONS) - 1]
            else:
                self.out += "Unknown or ambiguous function\n"
        elif name in ("xorder", "yorder"):
            value = scan.i()
            if scan.n == 1:
                self.out += f"{name} {getattr(c, name)}\n"
            elif value < 2:
                self.out += f"{name} must be at least 2\n"
            else:
                setattr(c, name, value)
        elif name == "corners":
            self.out += " ".join(f"({d(x)},{d(y)})={iraffmt.g(v)}"
                                 for x, y, v in self.corners()) + "\n"
        else:
            self.out += "Unrecognized or ambiguous command\a"

    # ------------------------------------------------------------ the keys

    def key(self, key: str, wx: float = 0.0, wy: float = 0.0, rest: str = "",
            scale=None) -> None:
        """One read of igs_fit's cursor loop; `rest` a colon command's text, `scale` a
        panel's view and size, for the nearest point as the screen shows it."""
        self.scale = scale
        state, self.state = self.state, ""
        newgraph = False
        if state in ("x", "y"):
            self.out += "\n"
            if key not in COLUMNS:
                self.out += BELL
                return
            at = 0 if state == "x" else 1
            if self.axis[at] != COLUMNS[key]:
                self.axis[at] = COLUMNS[key]
                self.gt.setr(f"{state}min", None)
                self.gt.setr(f"{state}max", None)
        elif state == "z":
            self.out += "\n"
            if key not in ("x", "y", "z"):
                self.out += BELL
                return
            self.ztype = COLUMNS[key]
            newgraph = True
        elif state in ("d", "u"):
            self.out += "\n"
            if key not in ("p", "x", "y", "z"):
                self.out += BELL
                return
            self._change(self.pending, 0 if key == "p" else COLUMNS[key], state == "d")
        elif state == "window":
            if key == "e":
                self.first, self.state = (wx, wy), "window2"
                self.out += "again:"
                return
            said, newgraph = self.gt.window1(key, wx, wy, self.window)
            self.out += said
        elif state == "window2":
            newgraph = self.gt.window2(self.first, (wx, wy), self.window)
        elif key == "?":
            self.out += pagefile(HELP)
        elif key == ":":
            if rest.startswith("/"):
                out, err, newgraph = self.gt.colon(rest, self.window)
                self.out += out
                self.err += err
            else:
                self.colon(rest)
        elif key in ("x", "y"):
            self.state = key
            self.out += PROMPTS[key]
        elif key == "r":
            newgraph = True
        elif key == "z":
            if self.ztype is None:
                self.refpt = self.nearest(wx, wy) or 0
                self.state = "z"
                self.out += PROMPTS["z"]
        elif key == "p":
            if self.ztype is not None:
                self.ztype = None
                newgraph = True
        elif key in ("l", "n"):
            if self.ztype is not None:
                newgraph = self._step(key == "n")
        elif key == "c":
            i = self.nearest(wx, wy)
            if i is not None:
                pts = self.pts
                self.out += " ".join(iraffmt.g(float(pts[k, i])) for k in
                                     (fc.X, fc.Y, fc.Z, fc.W, fc.S, fc.R)) + "\n"
        elif key in ("d", "u"):
            i = self.nearest(wx, wy, "kept" if key == "d" else "deleted")
            if i is None:
                return
            self.pending = i
            self.state = key
            self.out += PROMPTS[key]
        elif key == "f":
            self._solve()
            newgraph = True
        elif key == "w":
            self.state = "window"
            self.out += "window:"
        elif key == "q":
            self.done = True
            return
        elif key == "I":
            raise fc.FitcoordsError("Interrupt")
        else:
            self.out += BELL
        if newgraph:
            self.graph()

    def final(self) -> Frame:
        """The graph igs_fit draws for the plot log as it ends: as things stand."""
        frame = self.graph()
        self.frames.pop()
        return frame


# ------------------------------------------------------------ cursor reads

def reads(text: str) -> Iterator[tuple[float, float, int, str, str]]:
    """IRAF's cursor file, a read a line: 'x y wcs key', or 'x y wcs :command', or a
    bare ':command'; a blank key is written \\040. Gives (x, y, wcs, key, rest)."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(":"):
            yield 0.0, 0.0, 1, ":", stripped[1:]
            continue
        words = stripped.split(None, 3)
        if len(words) < 4:
            continue
        key = words[3]
        if key.startswith(":"):
            yield float(words[0]), float(words[1]), int(float(words[2])), ":", key[1:]
            continue
        key = " " if key == "\\040" else key[0]
        yield float(words[0]), float(words[1]), int(float(words[2])), key, ""


def run_reads(session: Session, cursor: Iterable) -> None:
    """igs_fit's loop over reads, until q or the end of them (EOF). The next call goes
    on where this one stopped, as IRAF reads one cursor file from fit to fit."""
    for x, y, _, key, rest in cursor:
        session.key(key, x, y, rest)
        if session.done:
            return


def to_graph(frame: Frame):
    """A frame as core.graph draws it: the window as drawn, the marks at gt's size 2."""
    from .graph import Graph

    logs = frame.logs
    graph = Graph(frame.title, frame.xlabel, frame.ylabel)
    graph.window = tuple(frame.window)
    if logs[0]:
        graph.xtran = "elog"
    if logs[1]:
        graph.ytran = "elog"
    over = {(x, y): kind for kind, x, y, erased in frame.over if not erased}
    for kind, x, y in frame.marks:
        graph.mark(x, y, over.get((x, y), kind), 2.0, 2.0)
    return graph


def svg(frame: Frame, path):
    """The plot log's graph, written as SVG."""
    return to_graph(frame).write(path)
