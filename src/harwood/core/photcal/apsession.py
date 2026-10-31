"""mkapfile's interactive mode: the keys, the plots, the colon commands.

From photcal/mkobsfile (phaigrow.x). A person looks at each image's curve of
growth, deletes the points that do not belong, and fits again. Nothing here
knows about a screen: a key is a method call that gives back what IRAF would
print, and `graph` gives back what IRAF would draw, so the panel and a file of
cursor commands (`--gcommands`) are the same session, and the session can be
held to IRAF's output without one.

The quirks kept, because they decide the numbers and what is printed. A deleted
point takes every aperture beyond it with it: `inap` is a count, not a mask. A
deletion does nothing to the fit until f, and until then the status says the
fit is out of date; q does not refit, and writes the corrections from the fit
it has, over the points as they now stand. f goes back to the first image. And
the fit's convergence test starts from what the last fit left (a static), so a
second f with nothing changed is not quite the first.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import iraffmt
from ..gio import widened
from .apfile import (
    MAX_MTERMS,
    Curve,
    Image,
    Result,
    Star,
    _f,
    correction,
    evaluate,
    fit_model,
    limits,
    log_title,
    show_initial,
    show_model,
    show_seeing,
    write_results,
)

f32 = np.float32
MAX_REAL = float(np.finfo(np.float32).max)

#: The keys, as the panel and `?` list them (apfile.key).
HELP = [
    ("m", "the observed and model curve of growth, by radius"),
    ("r", "the residuals by radius"),
    ("b", "the residuals by magnitude"),
    ("x", "the residuals by x"),
    ("y", "the residuals by y"),
    ("a", "the aperture correction by radius"),
    ("d", "delete the point at the cursor, and the apertures beyond it"),
    ("u", "undelete the point at the cursor, and those before it"),
    ("f", "fit again, without the deleted points"),
    ("c", "print the star at the cursor"),
    ("w", "print the aperture correction"),
    ("g", "redraw"),
    ("n", "next image"),
    ("p", "previous image"),
    ("?", "these keys"),
    (":", "show, image, nparams, swings, pwings, pgauss, rgescale, xwings, smallap, largeap"),
    ("q", "finish, and write the results"),
]

_GRAPHS = {"m", "r", "b", "x", "y", "a"}
_COMMANDS = ("show", "image", "nparams", "swings", "pwings", "pgauss", "rgescale", "xwings",
             "smallap", "largeap")
_SHOW = ("model", "seeing", "parameters")
_LABELS = {"m": "radius", "r": "radius", "b": "mag[1]", "x": "x", "y": "y", "a": "y"}


def _three_digits(value: float) -> str:
    """%.3g of a real, as the graphs' titles print the rms."""
    return "INDEF" if np.isnan(value) else iraffmt.g(value, decpl=3)


def _strdic(word: str, words: tuple[str, ...]) -> str | None:
    """strdic: the word, or the only one it begins; None if none or several."""
    if word in words:
        return word
    found = [w for w in words if w.startswith(word)]
    return found[0] if len(found) == 1 else None


@dataclass
class Graph:
    """What ph_gimfit and its neighbours draw."""
    title: list[str]
    xlabel: str
    ylabel: str
    window: tuple[float, float, float, float]    #: x0, x1, y0, y1; y0 > y1 for apcor
    kept: tuple[np.ndarray, np.ndarray]          #: the points in the fit, as +
    dropped: tuple[np.ndarray, np.ndarray]       #: the deleted ones, as x
    lines: list[tuple] = field(default_factory=list)    #: (x, y, dashed)


class ApSession:
    """ph_aigrow, up to the output: one image in view, and the fit so far."""

    def __init__(self, images: list[Image], naperts: int,
                 params: tuple = (1.2, 0.1, 0.5, 0.9, 0.0), nparams: int = 3,
                 smallap: int = 1, largeap: int = 0) -> None:
        self.images = [image for image in images if image is not None]
        if not self.images:
            raise ValueError("Error: There is no input data to fit")
        self.naperts = naperts
        self.smallap, self.largeap = limits(naperts, smallap, largeap)
        self.params = [float(f32(v)) for v in params]
        self.lterms = max(1, min(MAX_MTERMS, nparams))
        #: what each star measured; star.nap is IRAF's inap, what the fit may use
        self.measured = {id(s): s.nap for image in self.images for s in image.stars}
        self.fit = fit_model(self.images, self.params, self.lterms)
        self.newfit = False
        self.index = 0
        self.graphtype = "m"
        self.curve: Curve = evaluate(self.image, self.fit.params, naperts)
        self.finished = False
        self.printed = self.status()

    @property
    def image(self) -> Image:
        return self.images[self.index]

    @property
    def fitok(self) -> bool:
        return self.fit.ok

    # ---------------------------------------------------------------- keys

    def key(self, key: str, x: float = 0.0, y: float = 0.0,
            scale: tuple[float, float] | None = None, rest: str = "") -> list[str]:
        """One key at the cursor (x, y): what IRAF prints. `scale` is the
        screen's units per data unit on each axis, for nearest to mean nearest
        on the screen; by default the graph's window is the unit square, as in
        IRAF. After the run has finished the graphs can still be looked
        through, and what would change the results says so."""
        if self.finished and key in ("d", "u", "f", ":") and not (
                key == ":" and rest.split()[:1] == ["show"]):
            return ["The results are written: run the task again to change them."]
        newimage = newgraph = False
        out: list[str] = []
        if key == "?":
            out = [f"{k}\t{what}" for k, what in HELP]
        elif key == ":":
            out, newimage, newgraph = self._colon(rest)
        elif key == "f":
            self.fit = fit_model(self.images, self.params, self.lterms, self.fit.old)
            self.newfit = False
            newimage = newgraph = True
            self.index = 0
        elif key == "c":
            out = self._print_nearest(x, y, scale)
        elif key == "w":
            out = self.status()
        elif key in ("d", "u"):
            if self._delete(x, y, scale, key == "d"):
                self.newfit = True
        elif key == "g":
            newgraph = True
        elif key == "m":
            if self.graphtype != "m":
                self.graphtype, newgraph = "m", True
        elif key in _GRAPHS:
            if (self.fitok and not np.isnan(self.image.ro)
                    and self.graphtype != key):
                self.graphtype, newgraph = key, True
        elif key == "p":
            if self.index == 0:
                out = ["Already at beginning of image list"]
            else:
                self.index -= 1
                newimage = newgraph = True
        elif key == "n":
            if self.index == len(self.images) - 1:
                out = ["Already at end of image list"]
            else:
                self.index += 1
                newimage = newgraph = True
        if newimage:
            self.curve = evaluate(self.image, self.fit.params, self.naperts)
        if newgraph:
            out += self.status()
        return out

    def status(self) -> list[str]:
        """ph_aeprint."""
        if not self.fitok:
            return ["Error: The cog model fit did not converge"]
        if self.newfit:
            return ["Warning: The cog model fit is out-of-date"]
        if np.isnan(self.image.ro):
            return [f"Warning: Unable to fit RO for image {self.image.name}"]
        cum, wcum = correction(self.image, self.curve, self.smallap, self.largeap)
        rap = self._rap()
        return [f"Image: {self.image.name} rin={_f(rap[self.smallap - 1], 0, 2)} "
                f"rout={_f(rap[self.largeap - 1], 0, 2)} "
                f"apercor={_f(cum, 0, 3)} +/- {_f(wcum, 0, 4)}"]

    def _rap(self) -> np.ndarray:
        stars = self.image.stars
        return stars[0].rap if stars else np.full(self.naperts, np.nan, dtype=f32)

    # ------------------------------------------------------------ the points

    def _points(self, star: Star) -> tuple[np.ndarray, np.ndarray]:
        """Every point of a star on the current graph, aperture 2 first."""
        n = self.measured[id(star)] - 1
        c = self.curve
        if self.graphtype == "m":
            return c.rbar[1:n + 1], star.mag[1:n + 1]
        if self.graphtype == "a":
            return star.rap[:n], self._cumulative(star)[1:n + 1]
        y = (star.mag[1:n + 1] - c.adopt[1:n + 1]).astype(f32)
        if self.graphtype == "r":
            return c.rbar[1:n + 1], y
        x = {"b": star.mag[0], "x": star.x, "y": star.y}[self.graphtype]
        return np.full(n, x, dtype=f32), y

    def _cumulative(self, star: Star) -> np.ndarray:
        """The sum of the differences from each aperture out to the large one,
        the adopted curve standing in where the star has none."""
        total = np.zeros(self.naperts + 1, dtype=f32)
        measured = self.measured[id(star)]
        for j in range(self.largeap - 1, 0, -1):
            value = self.curve.adopt[j] if j + 1 > measured else star.mag[j]
            total[j] = f32(value + total[j + 1])
        return total

    def _nearest(self, x: float, y: float, scale, part: str):
        """(star, position among its points), or None. `part`: all of them for
        c, the ones in the fit for d, the deleted ones for u; the first of
        several at one distance, as IRAF's strict comparison leaves it."""
        if scale is None:
            x0, x1, y0, y1 = self.graph().window
            scale = 1.0 / ((x1 - x0) or 1.0), 1.0 / ((y1 - y0) or 1.0)
        sx, sy = scale
        best, found = MAX_REAL, None
        for star in self.image.stars:
            px, py = self._points(star)
            good = star.nap - 1
            first, last = {"all": (0, px.size), "kept": (0, good),
                           "dropped": (good, px.size)}[part]
            for k in range(first, last):
                r2 = float(((px[k] - x) * sx) ** 2 + ((py[k] - y) * sy) ** 2)
                if r2 < best:
                    best, found = r2, (star, k)
        return found

    def _delete(self, x: float, y: float, scale, delete: bool) -> bool:
        """ph_audelete. A count, so one point takes those beyond it along."""
        found = self._nearest(x, y, scale, "kept" if delete else "dropped")
        if found is None:
            return False
        star, k = found
        star.nap = k + 1 if delete else k + 2
        return True

    def _print_nearest(self, x: float, y: float, scale) -> list[str]:
        """ph_pnearest."""
        found = self._nearest(x, y, scale, "all")
        if found is None:
            return []
        star, k = found
        px, py = self._points(star)
        if self.graphtype == "a":
            # ph_pnearest reads the radius and the sum one aperture on from
            # the point it found, its index counting from aperture 2 by then.
            px, py = star.rap[1:], self._cumulative(star)[1:]
        kind = {"m": "delta mag", "a": "apercor"}.get(self.graphtype, "residual")
        return [f"Star: {star.id} x: {_f(star.x, 0, 3)} y: {_f(star.y, 0, 3)} "
                f"{_LABELS[self.graphtype]}: {_f(px[k], 0, 3)} {kind}: {_f(py[k], 0, 3)}"]

    # ------------------------------------------------------- colon commands

    def _colon(self, text: str) -> tuple[list[str], bool, bool]:
        """ph_acolon: what it prints, and whether the image and graph changed."""
        words = text.split()
        if not words:
            return [], False, False
        command = _strdic(words[0], _COMMANDS)
        value = words[1] if len(words) > 1 else None
        if command == "show":
            return self._show(value), False, False
        if command == "image":
            return self._to_image(value)
        if command in ("nparams", "smallap", "largeap"):
            return self._integer(command, value), False, False
        if command in ("swings", "pwings", "pgauss", "rgescale", "xwings"):
            return self._real(command, value), False, False
        return ["Ambiguous or undefined command"], False, False

    def _show(self, what: str | None) -> list[str]:
        which = _strdic(what, _SHOW) if what else None
        if which == "parameters":
            return show_initial(self.params, self.lterms)
        if not self.fitok:
            return ["Error: The cog model fit did not converge"]
        stale = ["Warning: The cog model fit is out-of-date"] if self.newfit else []
        if which == "seeing":
            return stale + show_seeing(self.images)
        return stale + show_model(self.fit, self.lterms)

    def _to_image(self, name: str | None) -> tuple[list[str], bool, bool]:
        if name is None:
            return [f"Image: {self.image.name}  R0: {_f(self.image.ro, 7, 3)}  "
                    f"X: {_f(self.image.xairmass, 5, 3)}"], False, False
        for index, image in enumerate(self.images):
            if image.name == name:
                self.index = index
                return [], True, True
        return [f"Image: {name} not found in data"], False, False

    def _integer(self, command: str, value: str | None) -> list[str]:
        current = {"nparams": self.lterms, "smallap": self.smallap,
                   "largeap": self.largeap}[command]
        try:
            number = int(value) if value is not None else None
        except ValueError:
            number = None
        if number is None:
            return [f"{command} = {current}"]
        if command == "nparams":
            self.lterms = max(1, min(number, MAX_MTERMS))
            self.newfit = True
        elif command == "smallap":
            self.smallap = max(1, min(number, self.naperts))
        else:
            self.largeap = max(1, min(number, self.naperts))
        return []

    def _real(self, command: str, value: str | None) -> list[str]:
        index = ("swings", "pwings", "pgauss", "rgescale", "xwings").index(command)
        try:
            number = float(f32(value)) if value is not None else None
        except ValueError:
            number = None
        if number is None:
            return [f"{command} = {iraffmt.g(float(f32(self.params[index])))}"]
        self.params[index] = {"swings": max(1.0, number),
                              "pwings": max(0.0, min(1.0, number)),
                              "pgauss": max(0.0, min(1.0, number)),
                              "rgescale": max(0.0, number),
                              "xwings": number}[command]
        return []

    # ------------------------------------------------------------ the graph

    def graph(self) -> Graph:
        """The current graph: ph_gimfit, ph_gaimres, ph_gbimres, ph_gaximres,
        ph_gayimres or ph_gacum."""
        kind = self.graphtype
        kept, dropped, all_y = ([], []), ([], []), []
        for star in self.image.stars:
            if kind == "a":
                in_fit, out, py = self._apcor_marks(star)
            else:
                px, py = self._points(star)
                good = star.nap - 1
                in_fit, out = (px[:good], py[:good]), (px[good:], py[good:])
            kept[0].append(in_fit[0])
            kept[1].append(in_fit[1])
            dropped[0].append(out[0])
            dropped[1].append(out[1])
            all_y.append(py)
        ys = np.concatenate(all_y) if all_y else np.zeros(0)
        ys = ys[np.isfinite(ys)]
        window = self._window(kind, ys)
        c = self.curve
        has_fit = self.fitok and not np.isnan(self.image.ro)
        lines: list[tuple] = []
        if kind == "m" and has_fit:
            lines = [(c.rbar[1:], c.theo[1:], True), (c.rbar[1:], c.adopt[1:], False)]
        elif kind in ("r", "b", "x", "y"):
            lines = [(np.array(window[:2]), np.zeros(2), False)]
            if kind == "r":
                lines.append((c.rbar[1:], c.ave[1:], False))
        elif kind == "a":
            lines = self._apcor_lines(window)
        title = [f"Image: {self.image.name}  Ro: {_f(self.image.ro, 0, 3)}  "
                 f"X: {_f(self.image.xairmass, 0, 3)}  Rms: {_three_digits(self._rms())}"]
        if kind == "m" and has_fit:
            title.append("Dashed line: theoretical cog   Solid line: adopted cog")
        if kind == "a":
            title.append("Dashed line: theoretical apcor  Solid line: adopted apcor")
        xlabel = {"b": "Magnitude", "x": "X Coordinate", "y": "Y Coordinate"}.get(
            kind, "Aperture Radius")
        ylabel = {"m": "Curve of Growth", "a": "Aperture Correction"}.get(kind, "Residuals")
        pack = lambda part: tuple(  # noqa: E731
            np.concatenate(p) if p else np.zeros(0) for p in part)
        return Graph(title, xlabel, ylabel, (*widened(*window[:2]), *widened(*window[2:])),
                     pack(kept), pack(dropped), lines)

    def _window(self, kind: str, ys: np.ndarray) -> tuple[float, float, float, float]:
        stars = self.image.stars
        if kind in ("b", "x", "y"):
            xs = np.array([{"b": s.mag[0], "x": s.x, "y": s.y}[kind] for s in stars],
                          dtype=float)
        else:
            xs = np.concatenate([s.rap for s in stars]) if stars else np.zeros(1)
        xs = xs[np.isfinite(xs)]
        lo, hi = (float(xs.min()), float(xs.max())) if xs.size else (0.0, 1.0)
        dx = hi - lo
        if ys.size:
            y0, y1 = float(ys.min()), float(ys.max())
            dy = y1 - y0
            y0, y1 = y0 - 0.1 * dy, y1 + 0.1 * dy
        else:
            y0, y1 = -0.1, 0.1
        if kind == "a":
            y0, y1 = y1, y0                     # brighter is up, as IRAF draws it
        return lo - 0.1 * dx, hi + 0.1 * dx, y0, y1

    def _apcor_marks(self, star: Star) -> tuple[tuple, tuple, np.ndarray]:
        """What ph_gacum marks for a star, and all its window looks at. It goes by
        largeap, not by what was measured: the apertures from the star's last good
        one out to largeap are crosses, deleted or not, those beyond are not drawn,
        and the window takes in every sum, the zeros past largeap too."""
        total = self._cumulative(star)
        rap = star.rap
        good = min(self.largeap - 1, star.nap - 1)
        out = (rap[star.nap - 1:self.largeap - 1], total[star.nap:self.largeap])
        return (rap[:good], total[1:good + 1]), out, total

    def _apcor_lines(self, window) -> list[tuple]:
        """ph_gacum's model curves, the zero line and the aperture limits."""
        c, rap = self.curve, self._rap()
        n, small, large = self.naperts, self.smallap, self.largeap
        theo = np.zeros(n + 1, dtype=f32)
        adopt = np.zeros(n + 1, dtype=f32)
        for j in range(large - 1, 0, -1):
            theo[j] = f32(c.theo[j] + theo[j + 1])
            adopt[j] = f32(c.adopt[j] + adopt[j + 1])
        y = (window[2], window[3])
        lines = [(np.array(window[:2]), np.zeros(2), False),
                 (np.array([rap[small - 1]] * 2), np.array(y), False),
                 (np.array([rap[large - 1]] * 2), np.array(y), False),
                 (rap, theo[1:n + 1], True), (rap, adopt[1:n + 1], False)]
        if small > rap[0]:
            lines.append((np.array([rap[0]] * 2), np.array(y), True))
        if rap[n - 1] > large:
            lines.append((np.array([rap[n - 1]] * 2), np.array(y), True))
        return lines

    def _rms(self) -> float:
        """sqrt (ph_achi)."""
        c = self.curve
        if not (self.fitok and not np.isnan(self.image.ro)):
            return float("nan")
        sumwr = float(np.sum(c.wr[:self.naperts], dtype=f32))
        if sumwr <= 0:
            return float("nan")
        chi = (float(np.sum(c.ressq[:self.naperts], dtype=f32)) / sumwr
               - (float(np.sum(c.resid[:self.naperts], dtype=f32)) / sumwr) ** 2)
        return float(np.sqrt(chi)) if 0.0 < chi <= MAX_REAL else float("nan")

    # ------------------------------------------------------------ a whole run

    def run_commands(self, text: str) -> list[str]:
        """A cursor file, `x y wcs key [command]` a line, up to q: all that was
        printed, the first status first."""
        out = list(self.printed)
        for line in text.splitlines():
            words = line.split(None, 4)
            if len(words) < 4 or line.lstrip().startswith("#"):
                continue
            x, y, key = float(words[0]), float(words[1]), words[3]
            if key == "q":
                break
            if key.startswith(":"):
                rest = (key[1:] + " " + (words[4] if len(words) > 4 else "")).strip()
                out += self.key(":", rest=rest)
            else:
                out += self.key(key, x, y)
        return out

    def finish(self, apercors: str, magfile: str = "", logfile: str = "",
               append: bool = False, names: bool = False) -> tuple[Result, list[str]]:
        """q: write the results; what IRAF prints on the way out."""
        said = []
        if not self.fitok:
            said = ["Error: The cog model fit did not converge"]
        elif self.newfit:
            said = ["Warning: The cog model fit is out-of-date"]
        log = log_title(apercors) if logfile else []
        result = write_results(self.images, self.fit, self.naperts, apercors, self.smallap,
                               self.largeap, magfile, logfile, self.params, self.lterms, log,
                               append, names, stale=self.newfit)
        self.finished = True
        return result, said
