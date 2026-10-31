"""icfit's interactive half: the keys, the graphs, the colon commands.

From xtools/icfit (icgfit, icggraph, icgaxes, icgparams, icgcolon, icgsample, icgadd,
icguishow). The same interaction in every task that fits a curve. Nothing here knows
about a screen: a key is a method call, what IRAF would print goes to `stdout`, and
`graph` gives back what IRAF would draw, so the panel and a file of cursor commands
are one session, held to IRAF's output without a window.

The quirks kept, because they decide the numbers and what is printed. A deletion, a
new weight and a new parameter change nothing until f; q does not refit, and what is
used is the fit there was. So a title drawn between d and f says the point is deleted,
and gives the rms of the old fit without it. A key that changes the axes redraws only
if the graph is not on that key already. The title prints the rejection limits and the
grow as whole numbers, as %r does, whatever they are. When the sample is not all the
points, deleted counts the first `sample` weights of the input and not of the sample.

Not ported: w and the /commands, which set a window (the mouse does that here);
:vshow's coefficients and their errors, since the fit here is not in IRAF's basis;
:color, :help and :gui, which have nothing to act on.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from . import icfit, iraffmt
from .gio import with_room

f32 = np.float32
MAX_REAL = float(np.finfo(np.float32).max)

#: The keys, as the panel and `?` list them (icgfit.key).
HELP = [
    ("f", "fit, with what has changed since the last time"),
    ("d", "delete the point nearest the cursor"),
    ("u", "undelete the nearest deleted point"),
    ("h", "graph h: y against x, as first set"),
    ("i", "graph i: x against y"),
    ("j", "graph j: residuals against x"),
    ("k", "graph k: ratio against x"),
    ("l", "graph l: non-linear component against x"),
    ("g", "set the axes of a graph key"),
    ("c", "print the point nearest the cursor"),
    ("a", "add a point at the cursor"),
    ("v", "set the weight of the nearest point"),
    ("x", "set the x of the nearest point"),
    ("y", "set the y of the nearest point"),
    ("s", "mark a sample range: here, then the other end"),
    ("t", "take all the points as the sample"),
    ("z", "drop the sample range nearest the cursor"),
    ("o", "the next graph over this one"),
    ("r", "redraw"),
    ("?", "these keys"),
    (":", "function, order, sample, naverage, niterate, low_reject, high_reject, grow, "
          "markrej, show, xyshow, errors, evaluate, graph"),
    ("q", "accept the last fit: q does not fit"),
]

_CMDS = ("function", "order", "sample", "naverage", "niterate", "low_reject", "high_reject",
         "grow", "markrej", "color", "show", "vshow", "xyshow", "errors", "evaluate", "graph",
         "help", "gui")
_FUNCTIONS = ("chebyshev", "legendre", "spline1", "spline3", "power")
_KEYS = "hijkl"
_AXES = {"h": ("x", "y"), "i": ("y", "x"), "j": ("x", "r"), "k": ("x", "d"), "l": ("x", "n")}


def _strdic(word: str, words) -> str | None:
    if word in words:
        return word
    found = [w for w in words if w.startswith(word)]
    return found[0] if len(found) == 1 else None


def _g(value) -> str:
    return iraffmt.g(float(value))


def _whole(value) -> str:
    """%r of a real: the whole number nearest below, to the eye of the title."""
    return str(int(float(value)))


def _width(value, digits: int, width: int) -> str:
    if value is None or np.isnan(value):
        return "INDEF".rjust(width)
    return iraffmt.printf(float(value), "g", digits, width=width).rjust(width)


@dataclass
class Graph:
    """What icg_graph draws. The data as a line through the points not deleted (or as
    marks, for a task that asks), a cross on each deleted, on the averaged points a
    plus as wide as the group, a diamond on each rejected, the fit dashed, and bars
    for the sample ranges along the bottom."""
    title: str
    params: list[str]
    xlabel: str
    ylabel: str
    window: tuple[float, float, float, float]
    kind: str
    kept: tuple[np.ndarray, np.ndarray]
    deleted: tuple[np.ndarray, np.ndarray]
    averaged: tuple[np.ndarray, np.ndarray]
    rejected: tuple[np.ndarray, np.ndarray]
    fit: tuple[np.ndarray, np.ndarray]
    sample: list[tuple[float, float, float, float]] = field(default_factory=list)
    overplot: bool = False


class IcfitSession:
    """icg_fit: the data, the parameters, and the fit as it last was."""

    def __init__(self, x, y, spec: icfit.FitSpec | None = None, weights=None, title: str = "",
                 xlabel: str = "", ylabel: str = "", kind: str = "line", xrange=None,
                 yunits: str = "", axes: dict | None = None, gkey: str = "h",
                 xwindow=None) -> None:
        self.x = np.asarray(x, dtype=float).copy()
        self.y = np.asarray(y, dtype=float).copy()
        self.wts = (np.ones(self.x.shape) if weights is None
                    else np.asarray(weights, dtype=float).copy())
        self.userwts = self.wts.copy()
        self.spec = spec or icfit.FitSpec()
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.yunits = yunits
        self.kind = kind
        self.xrange = xrange or ((float(self.x.min()), float(self.x.max())) if self.x.size
                                 else (0.0, 1.0))
        self.xwindow = xwindow        #: GTXMIN and GTXMAX, what a task fixes
        self.axes = {**_AXES, **(axes or {})}
        self.gkey = gkey
        self.markrej = True
        self.overplot = False
        self.newgraph = True
        self.fit: icfit.Fit | None = None
        self.window = (0.0, 1.0, 0.0, 1.0)
        self.fiterror = False
        self.stdout = self.stderr = ""
        self._new = ""
        self.state = "fit"
        self.prompt = ""
        self.done = False
        self._first: tuple[float, float] | None = None
        self._pending: tuple | None = None
        self._refit()

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
        return new

    def _ask(self, state: str, prompt: str) -> None:
        self.state, self.prompt = state, prompt
        self._print(prompt)

    @property
    def stale(self) -> bool:
        """Something has changed that only f takes up."""
        return self._changed

    # ---------------------------------------------------------------- the fit

    def _refit(self) -> None:
        """ic_fit, f. The range the spline's pieces are laid over only ever grows. If
        it cannot be done the fit stays what it was, as cv does, and says so."""
        self._changed = False
        if self.x.size:
            self.xrange = (min(self.xrange[0], float(self.x.min())),
                           max(self.xrange[1], float(self.x.max())))
        try:
            fit = icfit.fit(self.x, self.y, self.spec, deleted=self.wts == 0,
                            xrange=self.xrange, weights=self.wts)
        except ValueError as error:
            self.fiterror = True
            self._eprint("Warning: No sample points for fit\n")
            if self.fit is None:
                self.fit = icfit.Fit(self.spec, self.x, self.y, self.wts != 0, self.wts == 0,
                                     None, str(error))
            return
        if fit.model is None:
            self.fiterror = True
            self._eprint("Warning: No sample points for fit\n")
            if self.fit is None:
                self.fit = fit
            return
        self.fiterror = False
        if fit.message == "Singular solution":
            self._print(fit.message + "\n")
        self.fit = fit
        self.newgraph = True

    def _at(self, x) -> np.ndarray:
        return self.fit(np.asarray(x, dtype=float))

    def _axis(self, kind: str, axis: int, x, y) -> np.ndarray:
        """icg_axes: the values along one axis, of these points."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if kind == "x":
            return x.copy()
        if kind == "y":
            return y.copy()
        if self.fit is None or self.fit.model is None:
            return x.copy() if axis == 0 else y.copy()
        if kind == "f":
            return self._at(x)
        if kind == "r":
            return y - self._at(x)
        if kind == "d":
            fit = self._at(x)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(fit != 0, y / fit, 1.0)
        if kind == "n":
            xmin, xmax = self.xrange
            a = float(self._at([xmin])[0])
            b = (float(self._at([xmax])[0]) - a) / (xmax - xmin)
            return y - a - b * (x - xmin)
        if kind == "v":
            fit = self._at(x)
            with np.errstate(divide="ignore", invalid="ignore"):
                return (fit - y) / y * 300000.0
        return x.copy() if axis == 0 else y.copy()    # a user's, and there are none

    def _label(self, kind: str, axis: int) -> str:
        units = f" ({self.yunits})" if self.yunits else ""
        ylabel = self.ylabel + units if self.ylabel else ""
        return {"x": self.xlabel, "y": ylabel, "f": "fit" + units, "r": "residuals" + units,
                "d": "ratio", "n": "non-linear component" + units,
                "v": "Velocity (km/s)"}.get(kind, self.xlabel if axis == 0 else ylabel)

    def _on_graph(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.axes[self.gkey]
        return self._axis(a, 0, x, y), self._axis(b, 1, x, y)

    # --------------------------------------------------------------- the keys

    def key(self, key: str, x: float = 0.0, y: float = 0.0,
            scale: tuple[float, float] | None = None, rest: str = "") -> None:
        """One key at the cursor (x, y). `scale` is the screen's units per data
        unit on each axis, so that nearest means nearest on the screen."""
        if self.state == "second":
            self._second_end(x, y)
            return
        if self.state != "fit":
            return
        if key == "q":
            self.done = True
        elif key == "?":
            self._print("".join(f"{k}\t{what}\n" for k, what in HELP))
        elif key == ":":
            self._colon(rest)
        elif key == "a":
            self._add(x, y)
        elif key == "c":
            self._show_nearest(x, y, scale)
        elif key == "d":
            self._mark(x, y, scale, True)
        elif key == "u":
            self._mark(x, y, scale, False)
        elif key == "f":
            self._refit()
        elif key == "g":
            self._ask("gkey", "Graph key to be defined: ")
        elif key in _KEYS:
            if self.gkey != key:
                self.gkey, self.newgraph = key, True
        elif key == "t":
            self.spec = replace(self.spec, sample="*")
            self._changed = True
        elif key == "o":
            self.overplot = True
        elif key == "r":
            self.newgraph = True
        elif key == "s":
            if "x" in self.axes[self.gkey]:
                if "*" in self.spec.sample:
                    self.spec = replace(self.spec, sample="")
                self._first = (x, y)
                self.state = "second"
                self._print("again:\n")
        elif key in "vxy":
            self._ask_value(key, x, y, scale)
        elif key == "z":
            self._drop_sample(x, y)
        elif key != "w":
            self._print("\a\n")

    def answer(self, text: str | None) -> None:
        """What a prompt read: None at the end of the input."""
        state, self.state, self.prompt = self.state, "fit", ""
        if state == "gkey":
            self._graph_key(text)
        elif state == "gaxes":
            self._graph_axes(text)
        elif state == "weight":
            self._take_added_weight(text)
        elif state in "vxy":
            self._take_value(state, text)

    def _nearest(self, x: float, y: float, scale, part: str) -> int | None:
        """icg_n: `part` all, or those with weight, or those without."""
        xs, ys = self._on_graph(self.x, self.y)
        if scale is None:
            x0, x1, y0, y1 = self.window
            scale = 1.0 / ((x1 - x0) or 1.0), 1.0 / ((y1 - y0) or 1.0)
        best, found = MAX_REAL, None
        for i in range(xs.size):
            if part == "kept" and self.wts[i] == 0:
                continue
            if part == "deleted" and self.wts[i] != 0:
                continue
            r2 = float(((xs[i] - x) * scale[0]) ** 2 + ((ys[i] - y) * scale[1]) ** 2)
            if r2 < best:
                best, found = r2, i
        return found

    def _mark(self, x: float, y: float, scale, delete: bool) -> None:
        """icg_d1 and icg_u1."""
        found = self._nearest(x, y, scale, "kept" if delete else "deleted")
        if found is not None:
            self.wts[found] = 0.0 if delete else self.userwts[found]
            self._changed = True

    def _show_nearest(self, x: float, y: float, scale) -> None:
        found = self._nearest(x, y, scale, "all")
        if found is not None and self.fit is not None:
            self._print(f"x = {_g(self.x[found])}  y = {_g(self.y[found])}  "
                        f"fit = {_g(self._at([self.x[found]])[0])}\n")

    def _add(self, x: float, y: float) -> None:
        """The first half of icg_add: it asks the weight."""
        axes = self.axes[self.gkey]
        if axes == ("y", "x"):
            x, y = y, x
        elif axes != ("x", "y"):
            self._print("Graph must be x vs. y or y vs. x\a\n")
            return
        self._pending = (x, y)
        self._ask("weight", "weight = (1.) ")

    def _take_added_weight(self, text: str | None) -> None:
        weight = 1.0
        if text is not None:
            value = _number(text)
            if value is not None:
                weight = value
        x, y = self._pending
        ascending = self.x[0] < self.x[-1]
        i = self.x.size
        while i > 0 and (x < self.x[i - 1] if ascending else x > self.x[i - 1]):
            i -= 1
        self.x = np.insert(self.x, i, x)
        self.y = np.insert(self.y, i, y)
        self.wts = np.insert(self.wts, i, weight)
        self.userwts = np.insert(self.userwts, i, weight)
        self._changed = True

    def _ask_value(self, key: str, x: float, y: float, scale) -> None:
        found = self._nearest(x, y, scale, "all")
        if found is None:
            return
        self._pending = found
        now = {"v": self.wts, "x": self.x, "y": self.y}[key][found]
        name = "weight" if key == "v" else key
        self._ask(key, f"{name} = ({_g(now)}) ")

    def _take_value(self, key: str, text: str | None) -> None:
        value = None if text is None else _number(text)
        if value is not None:
            {"v": self.wts, "x": self.x, "y": self.y}[key][self._pending] = value
            self._changed = True

    def _graph_key(self, text: str | None) -> None:
        if text is None or not text.strip():
            return
        which = text.strip()[0]
        if which not in _KEYS:
            self._print("Not a graph key\n")
            return
        self._pending = which
        a, b = self.axes[which]
        self._ask("gaxes", f"Set graph axes types ({a}, {b}): ")

    def _graph_axes(self, text: str | None) -> None:
        """Two letters, and whatever one character is between them."""
        if text is None or len(text) < 3:
            return
        self.axes[self._pending] = (text[0], text[2])
        if self._pending == self.gkey:
            self.newgraph = True

    def _second_end(self, x: float, y: float) -> None:
        self.state = "fit"
        self._print("\n")
        x1, y1 = self._first
        x2, y2 = x, y
        if self.axes[self.gkey][0] == "x":
            a, b = x1, x2
        else:
            a, b = y1, y2
        if np.all(self.x == np.floor(self.x)):
            text = f" {_nint(a)}:{_nint(b)}"
        else:
            text = f" {_g(f32(a))}:{_g(f32(b))}"
        self.spec = replace(self.spec, sample=self.spec.sample + text)
        self._changed = True

    def _ranges(self) -> list[tuple[float, float]]:
        """rg_xranges: the sample's runs, by the x at each end."""
        inside = icfit.parse_sample(self.spec.sample, self.x)
        return [(float(self.x[run[0]]), float(self.x[run[-1]])) for run in icfit._runs(inside)]

    def _drop_sample(self, wx: float, wy: float) -> None:
        """icg_dsample, which measures from wx whichever axis the sample is on."""
        if "*" in self.spec.sample or "x" not in self.axes[self.gkey]:
            return
        w = wx if self.axes[self.gkey][0] == "x" else wy
        chosen, diff = 0, MAX_REAL
        ranges = self._ranges()
        for i, (x1, x2) in enumerate(ranges):
            if w < x1:
                if x1 - w < diff:
                    diff, chosen = x1 - wx, i
            elif wx > x2:
                if wx - x2 < diff:
                    diff, chosen = x1 - wx, i
            else:
                diff, chosen = 0.0, i
        text = ""
        for i, (x1, x2) in enumerate(ranges):
            if i != chosen:
                if x1 == int(x1) and x2 == int(x2):
                    text += f" {_nint(x1)}:{_nint(x2)}"
                else:
                    text += f" {_g(f32(x1))}:{_g(f32(x2))}"
        self.spec = replace(self.spec, sample=text or "*")
        self._changed = True

    # ------------------------------------------------------- the colon commands

    def _colon(self, text: str) -> None:
        """icg_colon."""
        words = text.split()
        if not words:
            return
        if words[0].startswith("/"):
            self._print("the /commands are not ported: the mouse sets the window here\n")
            return
        command = _strdic(words[0], _CMDS)
        value = words[1] if len(words) > 1 else None
        if command == "function":
            if value is None:
                self._print(f"function = {self.spec.function}\n")
            else:
                found = _strdic(value, _FUNCTIONS)
                if found is None or found == "power":
                    self._print("Unknown or ambiguous function\n")
                else:
                    self.spec = replace(self.spec, function=found)
                    self._changed = True
        elif command == "order":
            number = _integer(value)
            if number is None:
                self._print(f"order = {self.spec.order}\n")
            elif number < 1:
                self._print("Order must be greater than zero\n")
            else:
                self.spec = replace(self.spec, order=number)
                self._changed = True
        elif command == "sample":
            if value is None:
                self._print(f"sample = {self.spec.sample}\n")
            else:
                self.spec = replace(self.spec, sample=text.split(None, 1)[1].rstrip())
                self._changed = True
        elif command in ("naverage", "niterate"):
            number = _integer(value)
            if number is None:
                self._print(f"{command} = {getattr(self.spec, command)}\n")
            else:
                self.spec = replace(self.spec, **{command: number})
                self._changed = self._changed or command == "naverage"
        elif command in ("low_reject", "high_reject", "grow"):
            number = None if value is None else _number(value)
            if number is None:
                self._print(f"{command} = {_g(f32(getattr(self.spec, command)))}\n")
            else:
                self.spec = replace(self.spec, **{command: number})
        elif command == "markrej":
            if value is None:
                self._print(f"markrej = {'YES' if self.markrej else 'NO'}\n")
            else:
                self.markrej = value.lower().startswith(("y", "t")) or value == "+"
        elif command in ("show", "vshow", "xyshow", "errors"):
            self._show(command, value)
        elif command == "evaluate":
            number = None if value is None else _number(value)
            if number is None:
                self._print("evaluate requires a value to evaluate\n")
            elif self.fit is not None:
                self._print(f"fit({_g(number)}) = {_g(self._at([number])[0])}\n")
        elif command == "graph":
            if len(words) < 4:
                a, b = self.axes[self.gkey]
                self._print(f"graph {self.gkey} {a} {b}\n")
            else:
                self.gkey = words[1][0]
                self.axes[self.gkey] = (words[2][0], words[3][0])
                self.newgraph = True
        elif command in ("color", "help", "gui"):
            pass
        else:
            self._print("Unrecognized command or ambiguous abbreviation\a")

    def _show(self, command: str, file: str | None) -> None:
        text = ""
        if command != "xyshow":
            text = self._show_text()
        if command == "xyshow":
            text = self._xy_text()
        elif command == "errors":
            text += self._errors_text()
        elif command == "vshow":
            text += self._summary_text()
        if file is None:
            self._print(text)
            return
        try:
            with open(file, "a") as out:
                out.write(text)
        except OSError as error:
            self._print(f"Warning: {error}\n")

    def _show_text(self) -> str:
        """ic_fshow."""
        spec = self.spec
        lines = ["", f"# {time.strftime('%a %H:%M:%S %d-%b-%Y')}",
                 "# " + self.title.replace("\n", "\n# ")]
        if self.yunits:
            lines.append(f"# fit units = {self.yunits}")
        lines += [f"# function = {spec.function}", f"# grow = {_g(f32(spec.grow))}",
                  f"# naverage = {spec.naverage}", f"# order = {spec.order}",
                  f"# low_reject = {_g(f32(spec.low_reject))}",
                  f"# high_reject = {_g(f32(spec.high_reject))}",
                  f"# niterate = {spec.niterate}", f"# sample = {spec.sample}"]
        return "\n".join(lines) + "\n"

    def _fitted(self) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, float | None]:
        """What the title, :errors and :vshow all work out: how many were fitted, how
        many deleted, and the points with their weights, the rejected ones at zero."""
        npts = self.x.size
        fit = self.fit
        if self.fiterror:
            return 0, 0, self.x[:0], self.y[:0], self.wts[:0], None
        if fit is None or fit.binned is None:
            return npts, int(np.sum(self.wts == 0)), self.x, self.y, self.wts, None
        xb, yb, wb, rejected = fit.binned
        n = xb.size
        if n == npts:
            x, y, w = self.x, self.y, self.wts.copy()
            if rejected.any():
                w[rejected] = 0.0
            deleted = int(np.sum(self.wts == 0))
        else:
            x, y, w = xb, yb, np.where(rejected, 0.0, wb)
            deleted = int(np.sum(self.wts[:n] == 0))
        rms = None
        if fit.model is not None and not self.fiterror:
            use = w != 0
            rms = (float(np.sqrt(np.mean((y[use] - self._at(x[use])) ** 2)))
                   if use.any() else 0.0)
        return n, deleted, x, y, w, rms

    def _nrejected(self) -> int:
        if self.fiterror or self.fit is None or self.fit.binned is None:
            return 0
        return int(self.fit.binned[3].sum())

    def _errors_text(self) -> str:
        """ic_ferrors, whose second and fourth lines have no comment mark."""
        n, deleted, x, y, w, rms = self._fitted()
        return (f"# total points = {self.x.size}\nsample points = {n}\n"
                f"# nrejected = {self._nrejected()}\ndeleted = {deleted}\n"
                f"# RMS = {_width(rms, 4, 7)}\n"
                f"# square root of reduced chi square = {_width(self._chi(x, y, w), 4, 7)}\n")

    def _summary_text(self) -> str:
        """ic_fvshow, as far as the coefficients."""
        n, deleted, x, y, w, rms = self._fitted()
        return (f"# total points = {self.x.size}\n# sample points = {n}\n"
                f"# nrejected = {self._nrejected()}\n# deleted = {deleted}\n"
                f"# RMS = {_width(rms, 7, 10)}\n"
                f"# square root of reduced chi square = {_width(self._chi(x, y, w), 7, 10)}\n"
                "# the coefficients and their errors are not ported\n")

    def _chi(self, x, y, w) -> float | None:
        """cverrors' reduced chi square, its square root."""
        if self.fit is None or self.fit.model is None:
            return None
        use = w > 0
        spec = self.fit.spec
        pieces = {"spline3": 3, "spline1": 1}.get(spec.function, 0)
        degrees = int(use.sum()) - (spec.order + pieces)
        if degrees <= 0:
            return 0.0
        total = float(np.sum(w[use] * (y[use] - self._at(x[use])) ** 2))
        return float(np.sqrt(total / degrees))

    def _xy_text(self) -> str:
        """ic_fxyshow: the data being fit, not necessarily the input."""
        lines = ["#      X        Y    Y FIT   WEIGHT"]
        if self.fit is None:
            return lines[0] + "\n"
        if self.fit.binned is not None and self.fit.binned[0].size != self.x.size:
            x, y, w, _ = self.fit.binned
        else:
            x, y, w = self.x, self.y, self.wts
        fit = self._at(x)
        for row in zip(x, y, fit, w, strict=True):
            lines.append(" ".join(iraffmt.printf(float(v), "g", width=8).rjust(8)
                                  for v in row))
        return "\n".join(lines) + "\n"

    # -------------------------------------------------------------- the graph

    def _params(self) -> list[str]:
        """icg_params."""
        spec = self.spec
        n, deleted, _, _, _, rms = self._fitted()
        return [f"func={spec.function}, order={spec.order}, "
                f"low_rej={_whole(spec.low_reject)}, "
                f"high_rej={_whole(spec.high_reject)}, niterate={spec.niterate}, "
                f"grow={_whole(spec.grow)}",
                f"total={self.x.size}, sample={n}, rejected={self._nrejected()}, "
                f"deleted={deleted}, RMS={_width(rms, 4, 7)}"]

    def graph(self) -> Graph:
        """icg_graph: the graph, and it is drawn: the next one starts clean."""
        xs, ys = self._on_graph(self.x, self.y)
        overplot, self.overplot = self.overplot, False
        if not overplot:
            self.window = (*(self.xwindow or with_room(xs)), *with_room(ys))
        kept = self.wts != 0
        none = np.zeros(0)
        averaged = rejected = (none, none)
        fit = self.fit
        if fit is not None and fit.binned is not None:
            xb, yb, _, gone = fit.binned
            if xb.size != self.x.size:
                if abs(int(self.spec.naverage)) > 1 or gone.any():
                    bx, by = self._on_graph(xb, yb)
                    if abs(int(self.spec.naverage)) > 1:
                        averaged = (bx, by)
                    if gone.any() and self.markrej:
                        rejected = (bx[gone], by[gone])
            elif gone.any() and self.markrej:
                rejected = (xs[gone], ys[gone])
        curve = (none, none)
        if fit is not None and fit.model is not None and not self.fiterror:
            count = max(self.x.size, 100)
            xmin, xmax = self.xrange
            grid = xmin + np.arange(count) * ((xmax - xmin) / (count - 1))
            curve = self._on_graph(grid, self._at(grid))
        a, b = self.axes[self.gkey]
        graph = Graph(self.title, self._params(), self._label(a, 0), self._label(b, 1),
                      self.window, self.kind, (xs[kept], ys[kept]), (xs[~kept], ys[~kept]),
                      averaged, rejected, curve, self._sample_bars(), overplot)
        self.newgraph = False
        return graph

    def _sample_bars(self) -> list[tuple[float, float, float, float]]:
        """icg_sample."""
        if "*" in self.spec.sample or "x" not in self.axes[self.gkey]:
            return []
        try:
            ranges = self._ranges()
        except ValueError:
            return []
        along_x = self.axes[self.gkey][0] == "x"
        xl, xr, yb, yt = self.window if along_x else (*self.window[2:], *self.window[:2])
        dy = yt - yb
        y1 = yb + dy / 100
        y2 = y1 + dy / 20
        y3 = (y1 + y2) / 2
        bars = []
        for x1, x2 in ranges:
            if xl < x1 < xr:
                bars.append((x1, y1, x1, y2))
            if xl < x2 < xr:
                bars.append((x2, y1, x2, y2))
            bars.append((x1, y3, x2, y3))
        return bars if along_x else [(b, a, d, c) for a, b, c, d in bars]

    # ------------------------------------------------------------- a whole run

    def run_commands(self, cursor: str, answers: str = "") -> None:
        """A cursor file, `x y wcs key [command]` a line, and the lines the prompts
        read, as IRAF takes them from the terminal."""
        lines = iter(cursor.splitlines())
        replies = iter(answers.splitlines())

        def next_cursor():
            for line in lines:
                words = line.split(None, 4)
                if len(words) >= 4 and not line.lstrip().startswith("#"):
                    return words
            return None

        while not self.done:
            if self.newgraph:
                self.graph()
            if self.state not in ("fit", "second"):
                self.answer(next(replies, None))
                continue
            words = next_cursor()
            if words is None:
                self.done = True
                break
            x, y, key = float(words[0]), float(words[1]), words[3]
            if key.startswith(":"):
                rest = (key[1:] + " " + (words[4] if len(words) > 4 else "")).strip()
                self.key(":", rest=rest)
            else:
                self.key(key, x, y)
                if key == "w" and self.state == "fit":
                    next_cursor()

    def result(self) -> icfit.Fit:
        """The fit: the last one, whatever has changed since, as q leaves it."""
        return self.fit


def _number(text: str) -> float | None:
    try:
        return float(text.split()[0])
    except (ValueError, IndexError):
        return None


def _integer(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _nint(value) -> int:
    return int(np.floor(float(value) + 0.5))
