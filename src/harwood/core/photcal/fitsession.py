"""fitparams' interactive mode: the keys, the graphs, the prompts.

From xtools/inlfit (ing_fit and its neighbours) and photcal/fitparams
(ft_trneqs). A person looks at the fit of one equation, deletes the stars that
do not belong, fits again, and goes on to the next. Nothing here knows about a
screen: a key is a method call that prints what IRAF prints, and `graph` gives
back what IRAF would draw, so the panel and a file of cursor commands
(`--gcommands`) are the same session, and the session can be held to IRAF's
output without one.

The quirks kept, because they decide the numbers and what is printed. A deletion
does nothing to the fit until f, and q does not refit: the errors written are
those of the fit it has, over the weights as they now stand. A colon command
changes the next f, not the fit on the screen. The fit starts from the
equation's values each time, and those change only when a converged fit is
saved, so `again` after a save starts from where it ended. What `:const` and
`:fit` change lasts for that equation's turn only, as do the graph keys. Said no
to "quit?" is the same equation over again. `go` saves the rest unasked.

Not ported: `w` and the `:.` and `:/` commands of the graphics cursor (the mouse
zooms the panel), and `I`. In a cursor file `w` takes the next line with it, as
IRAF's does.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import iraffmt
from ..gio import with_room
from ..nlfit import DONE
from . import io
from .config import Config, evaluate
from .fit import (
    FitResult,
    Fitter,
    _w,
    equation_log,
    errors_text,
    log_results_text,
    read_inputs,
    show_text,
)

f32 = np.float32
MAX_REAL = float(np.finfo(np.float32).max)

#: The keys, as the panel lists them (inlgfit.key).
HELP = [
    ("c", "the star nearest the cursor"),
    ("d", "delete the nearest point"),
    ("u", "undelete the nearest deleted point"),
    ("f", "fit again"),
    ("h", "function, fit"),
    ("i", "function, residuals"),
    ("j", "function, ratio"),
    ("k", "first variable, residuals"),
    ("l", "the plot equations, or as i"),
    ("g", "say what h-l draw, as XV,residuals"),
    ("o", "next graph over this one"),
    ("t", "the fit's values, or not"),
    ("r", "redraw"),
    ("?", "these keys"),
    (":", "errors results show const fit nreject low_reject high_reject grow "
          "tolerance maxiter"),
    ("q", "finish this equation"),
]

_COMMANDS = ("show", "low_reject", "high_reject", "nreject", "grow", "errors", "vshow",
             "constant", "fit", "tolerance", "maxiter", "variables", "data", "page", "results")
_MOVES = ("first", "last", "prev", "next", "again", "go", "quit")
_TYPES = ("function", "fit", "residuals", "ratio", "nonlinear", "var", "user")
_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?$")

#: The graph keys: (x, number), (y, number); in_init's.
_KEYS = {"h": (("function", None), ("fit", None)),
         "i": (("function", None), ("residuals", None)),
         "j": (("function", None), ("ratio", None)),
         "k": (("var", 1), ("residuals", None)),
         "l": (("function", None), ("residuals", None))}

SAVE = "Do you want to save fit results ?"
QUIT = "Program will quit. Do you really want to do so ?"
_CHOICES = "|" + "|".join(_MOVES) + "|"


def strdic(word: str, words) -> int:
    """strdic: 1 + where the word is, or the only one it begins; else 0."""
    word = word.strip()
    if not word:
        return 0
    if word in words:
        return list(words).index(word) + 1
    found = [i for i, w in enumerate(words) if w.startswith(word)]
    return found[0] + 1 if len(found) == 1 else 0


def _g(value) -> str:
    return iraffmt.g(float(value))


def _yes(text: str | None, default: bool) -> bool:
    """xt_answer: what is not an answer leaves the default."""
    if text is None:
        return default
    found = strdic(text, ("no", "yes", "NO", "YES"))
    return default if not found else found % 2 == 0


@dataclass
class Graph:
    """What ing_graph draws."""
    title: list[str]
    params: list[str]
    xlabel: str
    ylabel: str
    window: tuple[float, float, float, float]
    kept: tuple[np.ndarray, np.ndarray]         #: as +
    deleted: tuple[np.ndarray, np.ndarray]      #: as x
    rejected: tuple[np.ndarray, np.ndarray]     #: as diamonds
    fit: tuple[np.ndarray, np.ndarray]          #: as boxes, after t
    overplot: bool = False


class FitSession:
    """ft_trneqs, one equation at a time. `key` is for while `state` is "fit",
    `answer` for while it is anything else, when `prompt` says what is asked.
    All that is printed collects in `stdout` and `stderr`, as IRAF's two streams,
    and `take` gives what is new, for a panel."""

    def __init__(self, cfg: Config, observations: list[str], catalogs: list[str], output: str,
                 logfile: str = "", catdir: str = "", log_unmatched: bool = True,
                 log_fit: bool = False, log_results: bool = False,
                 weighting: str = "uniform", **fit) -> None:
        self.cfg = cfg
        self.output, self.logfile = output, logfile
        self.log_fit, self.log_results = log_fit, log_results
        self.weighting = weighting
        self.settings = dict(fit, weighting=weighting)
        self.obs, logs = read_inputs(cfg, observations, catalogs, logfile, catdir,
                                     log_unmatched)
        self._log("".join(logs))
        self.labels = [_vtran(v.name) for v in cfg.obsvars + cfg.catvars]
        self.stdout, self.stderr = "", ""
        self._new = ""
        self.saved: list[FitResult] = []
        self.command = "next"           #: what the prompt shows, the last line typed
        self._move_to = "next"          #: where a blank line goes; "" after a bad one
        self.interactive = True
        self.neq = 0
        self.state = "fit"
        self.prompt = ""
        self._graphkey = ""
        self.newgraph = False           #: the graph is out of date (ing_fit's newgraph)
        self._begin()

    # ------------------------------------------------------------- printing

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

    def _write(self, file: str | None, text: str) -> None:
        """To the terminal, or appended to a file a colon command named."""
        if not file:
            self._print(text)
            return
        try:
            with open(file, "a") as out:
                out.write(text)
        except OSError as error:
            self._eprint(f"Warning: {error}\n")

    def _log(self, text: str) -> None:
        if self.logfile and text:
            with open(self.logfile, "a") as out:
                out.write(text)

    def _ask(self, state: str, prompt: str) -> None:
        self.state, self.prompt = state, prompt
        self._print(prompt)

    # -------------------------------------------------------- the equations

    @property
    def equations(self):
        return self.cfg.equations

    def _begin(self) -> None:
        """ft_rfeval to the first fit of ing_fit, or all of ft_trneq after `go`."""
        self.eq = self.equations[self.neq]
        self.fitter = Fitter(self.cfg, self.eq, self.obs.rows.copy(),
                             **self.settings)
        self.oldwts = self.fitter.wts.copy()
        self.keys = {key: list(axes) for key, axes in _KEYS.items()}
        self.graphkey = "i"
        if self.eq.xplot is not None and self.eq.yplot is not None:
            self.graphkey = "l"
            self.keys["l"] = [("user", 1), ("user", 2)]
        self.plotfit = False
        self.overplot = False
        self.fitter.run()
        self.xmin = self.fitter.x.min(axis=0)
        self.xmax = self.fitter.x.max(axis=0)
        if not self.interactive:
            self._save()
            self._after_save()
        else:
            self.state, self.prompt = "fit", ""
            self.newgraph = True

    def _save(self) -> None:
        """The end of ft_trneq, when the fit is to be kept."""
        result = self.fitter.result()
        io.write_coeffs(self.output, self.eq, result, self.weighting)
        self._log(equation_log(result, self.obs.names, self.log_fit, self.log_results))
        if result.stat == DONE:
            self.eq.values[:] = result.params
        self.saved.append(result)

    def _after_save(self) -> None:
        if self.interactive:
            self._ask_command()
        else:
            self._move("next")

    def _ask_command(self) -> None:
        self._ask("command", f"Command ({_CHOICES}) ({self.command}) : ")

    def _move(self, command: str) -> None:
        """The switch of ft_trneqs, on the command it was given."""
        last = len(self.equations) - 1
        if command == "first":
            self.neq = 0
        elif command == "last":
            self.neq = last
        elif command == "prev":
            if self.neq == 0:
                self._ask_command()
                return
            self.neq -= 1
        elif command == "next":
            self.neq += 1
        elif command == "go":
            self.interactive = False
            self.neq += 1
        elif command == "quit":
            self._ask("quit", f"{QUIT} (yes): ")
            return
        self._check_end(command)

    def _check_end(self, command: str) -> None:
        if self.neq > len(self.equations) - 1:
            if self.interactive and command != "go":
                self._ask("end", f"{QUIT} (yes): ")
            else:
                self._finish()
            return
        self._begin()

    def _finish(self) -> None:
        self.neq = min(self.neq, len(self.equations) - 1)
        self.state, self.prompt = "done", ""

    def answer(self, text: str | None) -> None:
        """A line for the prompt; None is the end of the file of answers."""
        state = self.state
        if state == "save":
            if _yes(text, True):
                self._save()
            self._after_save()
        elif state == "command":
            # A blank line after a bad one is `first`: ft_trncmd has lost track
            # of the command by then, and the word it looks up is word 0.
            word = (text or "").strip() or self._move_to or "first"
            found = strdic(word, _MOVES)
            if not found:
                self._print("\a")
                self.command, self._move_to = word, ""
                self._ask_command()
                return
            self.command = self._move_to = _MOVES[found - 1]
            self._move(self.command)
        elif state == "quit":
            if _yes(text, True):
                self._finish()
            else:
                self._check_end("quit")
        elif state == "end":
            if _yes(text, True):
                self._finish()
            else:
                self.neq = len(self.equations) - 1
                self._ask_command()
        elif state == "graphkey":
            self._define_key(text)
        elif state == "axes":
            self._define_axes(text)

    # ---------------------------------------------------------------- keys

    def key(self, key: str, x: float = 0.0, y: float = 0.0,
            scale: tuple[float, float] | None = None, rest: str = "") -> None:
        """One key at the cursor (x, y). `scale` is the screen's units per data
        unit on each axis, for nearest to mean nearest on the screen; by default
        the graph's window is the unit square. After the last equation the graphs
        can still be looked through, and what would change the results says so."""
        if self.state == "done":
            if key in ("d", "u", "f", ":", "g"):
                self._print("The results are written: run the task again to change them.\n")
                return
        elif self.state != "fit":
            return
        fitter = self.fitter
        if key == "q":
            if self.state == "fit":
                self._ask("save", f"{SAVE} (yes): ")
        elif key == "?":
            self._print("".join(f"{k}\t{what}\n" for k, what in HELP))
        elif key == ":":
            self._colon(rest)
        elif key == "c":
            self._print_nearest(x, y, scale)
        elif key == "d":
            self._mark(x, y, scale, True)
        elif key == "u":
            self._mark(x, y, scale, False)
        elif key == "f":
            restore = fitter.wts > 0
            fitter.wts[restore] = self.oldwts[restore]
            fitter.run()
            self.newgraph = True
        elif key == "g":
            self._ask("graphkey", "Graph key to be defined: ")
        elif key in self.keys:
            if self.graphkey != key:
                self.graphkey, self.newgraph = key, True
        elif key == "o":
            self.overplot = True
        elif key == "r":
            self.newgraph = True
        elif key == "t":
            self.plotfit = not self.plotfit
            self.newgraph = True
        elif key != "w":
            self._print("\a\n")

    def _define_key(self, text: str | None) -> None:
        """ing_defkey, the first prompt."""
        self.state, self.prompt = "fit", ""
        if text is None or not text.strip():
            return
        word = text.strip()[0]
        if word not in self.keys:
            self._eprint("Not a graph key, choose: [h, i, j, k, l]\n")
            return
        self._graphkey = word
        self._ask("axes", "Set graph axis types ("
                  + ", ".join(self._axis_name(*axis) for axis in self.keys[word]) + ") : ")

    def _axis_name(self, kind: str, number: int | None) -> str:
        if kind == "user":
            return f"user{number}"
        if kind == "var":
            if 1 <= number <= len(self.labels):
                return self.labels[number - 1]
            return f"var{number}"
        return kind

    def _define_axes(self, text: str | None) -> None:
        """ing_defkey, the second prompt: X,Y, each a type or a variable's name."""
        self.state, self.prompt = "fit", ""
        if text is None:
            return
        words = [w for w in re.split(r"[, ]+", text.strip()) if w]
        axes = []
        for axis in range(2):
            if not words:
                if axis == 1:
                    self._eprint("Incomplete definition, usage: X,Y\n")
                return
            word = words.pop(0)
            found = strdic(word, _TYPES) if len(word) >= 3 else 0
            if not found:
                number = strdic(word, self.labels)
                if not number:
                    choices = "|" + "|".join(self.labels) + "|"
                    self._eprint(f"Not a defined key type ({word}), choose: [{choices}]\n")
                    return
                axes.append(("var", number))
                continue
            kind = _TYPES[found - 1]
            if kind in ("var", "user"):
                if not words:
                    self._eprint("Incomplete definition, usage: X,Y\n")
                    return
                word = words.pop(0)
                if not word.lstrip("+-").isdigit():
                    self._eprint(f"Not a valid var/user number ({word})\n")
                    return
                number = int(word)
                if kind == "var" and not 1 <= number <= len(self.labels):
                    self._eprint(f"Variable number does not exist ({word})\n")
                    return
                axes.append((kind, number))
            else:
                axes.append((kind, None))
        self.keys[self._graphkey] = axes
        if self.graphkey == self._graphkey:
            self.newgraph = True

    # ---------------------------------------------------------- the points

    def _axis(self, axis: int, fitted: bool = False) -> np.ndarray:
        """ing_axes: the values along axis 0 (x) or 1 (y) of the current graph.
        `fitted`: of the fit's own values, for ing_gf, which puts them where
        the data would be."""
        kind, number = self.keys[self.graphkey][axis]
        fitter = self.fitter
        x = fitter.x
        y = fitter.nl.vector(x) if fitted else fitter.y
        if kind == "var":
            return x[:, number - 1].copy()
        if kind == "function":
            return y.copy()
        fit = fitter.nl.vector(x)
        if kind == "fit":
            return fit
        if kind == "residuals":
            return (y - fit).astype(f32)
        if kind == "ratio":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(fit != 0, y / fit, f32(1.0)).astype(f32)
        if kind == "nonlinear":
            return self._nonlinear(y)
        return self._user(axis, number, y)

    def _nonlinear(self, y: np.ndarray) -> np.ndarray:
        """KEY_NONLINEAR: the data less the fit's slope along each variable."""
        fitter = self.fitter
        z = np.zeros(len(y), dtype=f32)
        point = self.xmin.copy()
        a = evaluate(self.eq.fit.code, point, fitter.nl.param)
        for i in range(len(point)):
            xmin, xmax = self.xmin[i], self.xmax[i]
            point[i] = xmax
            with np.errstate(divide="ignore", invalid="ignore"):
                b = f32(f32(evaluate(self.eq.fit.code, point, fitter.nl.param) - a)
                        / f32(xmax - xmin))
            point[i] = xmin
            z = (z + y - a - b * (fitter.x[:, i] - xmin)).astype(f32)
        return z

    def _user(self, axis: int, number: int, y: np.ndarray) -> np.ndarray:
        """ing_uaxes and ft_plot."""
        fitter = self.fitter
        eq = self.eq
        if eq.xplot is None or eq.yplot is None:
            self._eprint("Warning: User plot function not defined\n")
            if axis == 0:
                column = min(number, fitter.x.shape[1]) - 1
                return fitter.x[:, column].copy()
            return y.copy()
        code = (eq.xplot if number == 1 else eq.yplot).code
        return np.array([evaluate(code, row, fitter.nl.param) for row in fitter.x], dtype=f32)

    def _scale(self, xs, ys) -> tuple[float, float]:
        x0, x1, y0, y1 = _window(xs, ys)
        return 1.0 / ((x1 - x0) or 1.0), 1.0 / ((y1 - y0) or 1.0)

    def _nearest(self, xs, ys, x: float, y: float, scale, part: str) -> int | None:
        """ing_n, ing_d1, ing_u1: the first of several at one distance, as the
        strict comparison leaves it. `part`: all, kept (weight), or deleted."""
        sx, sy = scale if scale is not None else self._scale(xs, ys)
        best, found = MAX_REAL, None
        for i in range(len(xs)):
            if part == "kept" and self.fitter.wts[i] == 0:
                continue
            if part == "deleted" and self.fitter.wts[i] != 0:
                continue
            r2 = float(((xs[i] - x) * sx) ** 2 + ((ys[i] - y) * sy) ** 2)
            if r2 < best:
                best, found = r2, i
        return found

    def _print_nearest(self, x: float, y: float, scale) -> None:
        xs, ys = self._axis(0), self._axis(1)
        i = self._nearest(xs, ys, x, y, scale, "all")
        if i is None:
            return
        fitter = self.fitter
        fit = evaluate(self.eq.fit.code, fitter.x[i], fitter.nl.param)
        self._print(f"{i + 1} {self.obs.names[i]} x={_g(xs[i])} y={_g(ys[i])} "
                    f"func={_g(fitter.y[i])} fit={_g(fit)}, "
                    f"resid={_g(f32(fitter.y[i] - fit))}\n")

    def _mark(self, x: float, y: float, scale, delete: bool) -> None:
        xs, ys = self._axis(0), self._axis(1)
        i = self._nearest(xs, ys, x, y, scale, "kept" if delete else "deleted")
        if i is not None:
            self.fitter.wts[i] = 0 if delete else self.oldwts[i]

    # ------------------------------------------------------- colon commands

    def _colon(self, text: str) -> None:
        """ing_colon: none of them asks for a new graph."""
        words = text.split()
        if not words or words[0].startswith((".", "/")):
            return
        found = strdic(words[0], _COMMANDS)
        if not found:
            self._print("\a\n")
            return
        command = _COMMANDS[found - 1]
        value = words[1] if len(words) > 1 else None
        fitter = self.fitter
        if command == "show":
            self._write(value, self._title() + show_text(fitter.settings()))
        elif command == "errors":
            self._write(value, self._title() + show_text(fitter.settings())
                        + errors_text(fitter.result()))
        elif command == "vshow":
            self._write(value, self._title() + show_text(fitter.settings())
                        + errors_text(fitter.result())
                        + log_results_text(fitter.result(), self.obs.names))
        elif command == "variables":
            self._write(value, self._title() + self._variables())
        elif command == "data":
            self._write(value, self._title() + self._data())
        elif command == "results":
            self._write(value, self._title()
                        + log_results_text(fitter.result(), self.obs.names))
        elif command == "page":
            self._page(value)
        elif command in ("constant", "fit"):
            self._change(command == "fit", words[1:])
        elif command in ("nreject", "maxiter"):
            self._integer(command, value)
        else:
            self._real(command, value)

    def _title(self) -> str:
        """ing_title."""
        return f"\n#{io.cnvtime()}\n#{self.eq.ref.text} = {self.eq.fit.text}\n" \
               f"#{io.STATUS.get(self.fitter.stat, 'Unknown error code')}\n\n"

    def _variables(self) -> str:
        lines = ["", f"{'number':<10.10}  {'variable':<10.10}  {'minimum':>14.14}  "
                     f"{'maximum':>14.14}"]
        for i, label in enumerate(self.labels):
            lines.append(f"{i + 1:<10d}  {label:<10.10}  {_w(self.xmin[i], 14, 'f', 7)}  "
                         f"{_w(self.xmax[i], 14, 'f', 7)}")
        return "\n".join(lines) + "\n\n"

    def _data(self) -> str:
        """ing_data."""
        per_line = 5
        nvars = len(self.labels)
        text = ""
        for j in range(1, nvars + 2):
            if j % per_line == 1:
                text += "\n#"
            if j == 1:
                text += f"{'objectid':>14.14} "
            else:
                text += f"{self.labels[j - 2]:>14.14} "
        text += "\n"
        for i, name in enumerate(self.obs.names):
            for j in range(1, nvars + 2):
                if j == 1:
                    text += f"\n{name:>15.15}"
                elif j % per_line == 1:
                    text += f"\n*{_w(self.fitter.x[i, j - 2], 14)}"
                else:
                    text += f" {_w(self.fitter.x[i, j - 2], 14)}"
        return text + "\n\n"

    def _page(self, name: str | None) -> None:
        if name is None:
            self._print("File to be paged is undefined\n")
            return
        try:
            self._print(Path(name).read_text())
        except OSError as error:
            self._eprint(f"Warning: {error}\n")

    def _change(self, fit: bool, words: list[str]) -> None:
        """ing_change: a parameter between fitted and constant, by name or
        number, and its value, which the next f starts from."""
        if not words:
            self._eprint("Parameter not specified\n")
            return
        names = [p.name for p in self.eq.params]
        number = strdic(words[0], names)
        if not number:
            if not re.match(r"[+-]?\d+", words[0]):
                self._eprint(f"Parameter not found ({words[0]})\n")
                return
            number = int(re.match(r"[+-]?\d+", words[0])[0])
        if not 1 <= number <= len(names):
            self._eprint(f"Parameter out of range ({number})\n")
            return
        fitter = self.fitter
        value = fitter.values[number - 1]
        if len(words) > 1:
            if not _NUMBER.match(words[1]):
                self._eprint(f"Bad parameter value ({words[1]})\n")
                return
            value = f32(words[1].replace("d", "e").replace("D", "e"))
        fitter.values[number - 1] = value
        plist = fitter.plist
        pos = next((i for i, k in enumerate(plist) if k >= number), len(plist))
        present = pos < len(plist) and plist[pos] == number
        if fit and not present:
            plist.insert(pos, number)
        elif not fit and present:
            plist.pop(pos)
        # Found by number, IRAF prints whatever was in memory for the name.
        self._print(f"({names[number - 1]}) changed to {'fitting' if fit else 'constant'} "
                    "parameter, "
                    f"with value={_g(value)}\n")

    def _integer(self, command: str, value: str | None) -> None:
        fitter = self.fitter
        number = None
        if value is not None and _NUMBER.match(value):
            number = int(float(value))
        if number is None:
            self._print(f"{command} = {getattr(fitter, command)}\n")
        else:
            setattr(fitter, command, number)

    def _real(self, command: str, value: str | None) -> None:
        fitter = self.fitter
        name = "tol" if command == "tolerance" else command
        number = None
        if value is not None and _NUMBER.match(value):
            number = float(f32(value.replace("d", "e").replace("D", "e")))
        if number is None:
            self._print(f"{name} = {_g(f32(getattr(fitter, name)))}\n")
        else:
            setattr(fitter, name, number)

    # ------------------------------------------------------------ the graph

    def graph(self) -> Graph:
        """ing_graph, and ing_params for the title and the line under it."""
        fitter = self.fitter
        xs, ys = self._axis(0), self._axis(1)
        wts = fitter.wts
        rejected = fitter.rejected
        kept = wts != 0
        deleted = ~kept
        mark = (np.zeros(0), np.zeros(0))
        fit = mark
        if self.plotfit and fitter.stat == DONE:
            good = wts != 0
            fit = (self._axis(0, fitted=True)[good], self._axis(1, fitted=True)[good])
        result = fitter.result()
        rms = result.rms if fitter.stat == DONE else float("nan")
        four = lambda v: iraffmt.printf(float(v), "g", 4)  # noqa: E731
        params = [f"low_rej={four(fitter.low_reject)}, high_rej={four(fitter.high_reject)}, "
                  f"nreject={fitter.nreject}, grow={four(fitter.grow)}",
                  f"total={len(fitter.y)}, rejected={result.nrejected}, "
                  f"deleted={int(np.sum(wts == 0))}, "
                  f"RMS={'INDEF' if np.isnan(rms) else four(rms)}",
                  f"tolerance={four(f32(fitter.tol))}, maxiter={fitter.maxiter}, "
                  f"iterations={fitter.nl.iter}"]
        title = [f"{self.eq.ref.text} = {self.eq.fit.text}",
                 io.STATUS.get(fitter.stat, "Unknown error code")]
        graph = Graph(title, params, self._label(0), self._label(1), _window(xs, ys),
                      (xs[kept], ys[kept]), (xs[deleted], ys[deleted]),
                      (xs[rejected], ys[rejected]), fit, self.overplot)
        self.overplot = self.newgraph = False
        return graph

    def _label(self, axis: int) -> str:
        kind, number = self.keys[self.graphkey][axis]
        if kind == "var":
            return self.labels[number - 1]
        if kind == "user":
            plot = self.eq.xplot if number == 1 else self.eq.yplot
            return plot.text if plot is not None else f"user{number}"
        return kind

    # ------------------------------------------------------------ a whole run

    def run_commands(self, cursor: str, answers: str = "") -> None:
        """A cursor file, `x y wcs key [command]` a line, and the lines the
        prompts read, as IRAF takes them from the terminal. When the cursor file
        ends each equation is left as q leaves it; when the answers end each
        prompt takes its default."""
        lines = iter(cursor.splitlines())
        replies = iter(answers.splitlines())

        def next_cursor():
            for line in lines:
                words = line.split(None, 4)
                if len(words) >= 4 and not line.lstrip().startswith("#"):
                    return words
            return None

        unanswered = 0
        while self.state != "done":
            if self.newgraph:
                self.graph()
            if self.state != "fit":
                reply = next(replies, None)
                unanswered = unanswered + 1 if reply is None else 0
                # IRAF asks for ever if the default it is left with goes nowhere.
                if unanswered > 2 * len(self.equations) + 4:
                    self._finish()
                    break
                self.answer(reply)
                continue
            words = next_cursor()
            if words is None:
                self.key("q")
                continue
            x, y, key = float(words[0]), float(words[1]), words[3]
            if key.startswith(":"):
                rest = (key[1:] + " " + (words[4] if len(words) > 4 else "")).strip()
                self.key(":", rest=rest)
            else:
                self.key(key, x, y)
                if key == "w":
                    next_cursor()


def _window(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    return (*with_room(xs), *with_room(ys))


def _vtran(name: str) -> str:
    """pr_vtran."""
    if "@E_" in name:
        return f"er({name.split('@E_', 1)[1]})"
    if "@W_" in name:
        return f"wt({name.split('@W_', 1)[1]})"
    return name
