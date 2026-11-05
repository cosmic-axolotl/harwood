"""apedit's interactive half: the keys, the graph, the colon commands.

From noao/twodspec/apextract (apedit, apgraph, apgmark, apcolon, apids, apsort, apupdate,
apnearest, apfindnew). The arithmetic is core.apextract's. Nothing here knows about a
screen: a key is a method call, what IRAF would print goes to `stdout`, and `graph` gives
back what IRAF would draw, so the panel and a file of cursor commands are one session.
b hands the keys to an icfit session on the profile, as IRAF does, until its q.

The quirks kept, because they decide what is extracted. All, a, is a switch that stays
on, for every key after it that looks at it, until a again. A new aperture takes its
limits from the current one. l and u set a limit at the cursor, and a low above the high
is swapped. Two apertures within .01 of as near the cursor ask which is meant.

Not ported: w and the /commands, which set a window (the mouse does that here); e and t,
extraction and tracing being the steps the task goes on to; and :parameters, :database,
:logfile, :plotfile, :read, :write and :image, the database being the task's.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import apextract, icfit, icfitsession, iraffmt
from .apextract import Aperture
from .center1d import center1d
from .gio import with_room
from .gtools import GTools

f32 = np.float32
MAX_REAL = float(np.finfo(f32).max)
DELTA = 0.01

#: The keys, as the panel and `?` list them (apedit.key).
HELP = [
    ("m", "a new aperture, centred on the line at the cursor"),
    ("n", "a new aperture, at the cursor"),
    ("f", "find apertures, as many as nfind"),
    ("d", "delete the nearest"),
    ("c", "centre the nearest"),
    ("g", "recentre: the recenter algorithm"),
    ("s", "move it to the cursor"),
    ("l", "the low limit at the cursor"),
    ("u", "the high limit at the cursor"),
    ("y", "the limits where the profile crosses the cursor"),
    ("z", "resize: the resize algorithm"),
    ("b", "the background: icfit's keys on the profile, until its q"),
    ("i", "the aperture's number"),
    ("j", "the aperture's beam"),
    ("o", "number them all from this one"),
    (".", "go to the nearest"),
    ("+", "the next"),
    ("-", "the previous"),
    ("a", "all, or not: a switch"),
    ("r", "redraw"),
    ("?", "these keys"),
    (":", "show, line, nsum, center, lower, upper, title, and apall's parameters"),
    ("q", "finish"),
]

_NEAREST = set(".bcdegijotyz")
_CMDS = ("show", "parameters", "database", "logfile", "plotfile", "read", "write", "image",
         "line", "nsum", "center", "lower", "upper", "title", "extras", "apidtable",
         "b_function", "b_order", "b_sample", "b_naverage", "b_niterate", "b_low_reject",
         "b_high_reject", "b_grow", "minsep", "maxsep", "order", "apertures", "npeaks",
         "shift",
         "llimit", "ulimit", "ylevel", "peak", "bkg", "r_grow", "avglimits", "width", "radius",
         "threshold", "t_nsum", "t_step", "t_width", "t_function", "t_order", "t_sample",
         "t_naverage", "t_niterate", "t_low_reject", "t_high_reject", "t_grow", "nsubaps",
         "background", "skybox", "clean", "saturation", "weights", "readnoise", "gain",
         "lsigma", "usigma", "t_nlost")
_BOOLEANS = {"extras", "shift", "peak", "bkg", "avglimits", "clean"}
_INTEGERS = {"b_order", "b_naverage", "b_niterate", "t_nsum", "t_step", "t_order",
             "t_naverage", "t_niterate", "nsubaps", "skybox", "t_nlost"}
_STRINGS = {"apidtable", "b_function", "b_sample", "order", "apertures", "t_function",
            "t_sample", "background", "weights", "readnoise", "gain"}

#: The defaults of apall.par, for what no one has set.
SETTINGS = {
    "extras": True, "apidtable": "", "b_function": "chebyshev", "b_order": 1,
    "b_sample": "-10:-6,6:10", "b_naverage": -3, "b_niterate": 0, "b_low_reject": 3.0,
    "b_high_reject": 3.0, "b_grow": 0.0, "minsep": 5.0, "maxsep": 1000.0,
    "order": "increasing", "apertures": "", "npeaks": None, "shift": True, "llimit": None,
    "ulimit": None, "ylevel": 0.1, "peak": True, "bkg": True, "r_grow": 0.0,
    "avglimits": False, "width": 5.0, "radius": 10.0, "threshold": 0.0, "t_nsum": 10,
    "t_step": 10, "t_width": 5.0, "t_function": "legendre", "t_order": 2, "t_sample": "*",
    "t_naverage": 1, "t_niterate": 0, "t_low_reject": 3.0, "t_high_reject": 3.0, "t_grow": 0.0,
    "nsubaps": 1, "background": "none", "skybox": 1, "clean": False, "saturation": None,
    "weights": "none", "readnoise": "0.", "gain": "1.", "lsigma": 4.0, "usigma": 4.0,
    "t_nlost": 3, "lower": -5.0, "upper": 5.0, "nfind": 1}


def _strdic(word: str, words) -> str | None:
    if word in words:
        return word
    found = [w for w in words if w.startswith(word)]
    return found[0] if len(found) == 1 else None


def _g(value) -> str:
    if value is None:
        return "INDEF"
    return iraffmt.g(float(f32(value)))


def _f(value) -> str:
    return iraffmt.printf(float(f32(value)), "f", 2)


@dataclass
class Mark:
    """ap_gmark: an aperture along the top of the plot."""
    number: int
    centre: float
    low: float
    high: float


@dataclass
class Graph:
    """What ap_graph draws."""
    title: str
    params: str
    window: tuple[float, float, float, float]
    y: np.ndarray
    marks: list[Mark] = field(default_factory=list)
    sample: list[tuple[float, float]] = field(default_factory=list)


class ApeditSession:
    """ap_edit on one frame: `frame` is [dispersion, aperture], as apextract has it."""

    def __init__(self, frame: np.ndarray, apertures: list[Aperture], line: int | None = None,
                 nsum: int = 10, image: str = "", title: str = "",
                 settings: dict | None = None, axis: str = "Column") -> None:
        self.frame = np.asarray(frame, dtype=f32)
        self.apertures = list(apertures)
        self.nsum = nsum
        self.profile, self.line = apextract.getdata(self.frame, line, nsum)
        self.npts = self.profile.size
        self.image = image
        self.params = title
        self.axis = axis
        self.settings = {**SETTINGS, **(settings or {})}
        self.current = min(1, len(self.apertures))
        self.all = False
        self.newgraph = True
        self.stdout = self.stderr = ""
        self._new = ""
        self.state = "keys"
        self.prompt = ""
        self.done = False
        self.sub: icfitsession.IcfitSession | None = None
        self._then = None
        self._cursor = (0.0, 0.0)
        self._graph_keys: dict = {}     #: each aperture's icfit keeps its graph key
        self._ic_gt: GTools | None = None   #: apedit's ic_gt, made at the first b
        self._status()

    # ------------------------------------------------------------- the output

    def _print(self, text: str) -> None:
        self.stdout += text
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

    def _status(self) -> None:
        """ap_print."""
        if self.current < 1:
            return
        ap = self.apertures[self.current - 1]
        self._print(("ALL: " if self.all else "     ")
                    + f"aperture = {ap.number}  beam = {self._beam(ap)}  "
                      f"center = {_f(self._centre(ap))}  low = {_f(ap.low)}  "
                      f"upper = {_f(ap.high)}\n")

    def _beam(self, ap: Aperture) -> int:
        return ap.number if ap.beam is None else ap.beam

    def _centre(self, ap: Aperture) -> float:
        return float(f32(f32(ap.centre) + f32(ap.trace(np.array([float(self.line)]))[0])))

    def _center1d(self, x: float) -> float | None:
        return center1d(float(x), self.profile, self.settings["width"], "emission",
                        self.settings["radius"], self.settings["threshold"])

    # --------------------------------------------------------------- the keys

    def key(self, key: str, x: float = 0.0, y: float = 0.0, rest: str = "") -> None:
        """One key at the cursor."""
        if self.sub is not None:
            self.sub.key(key, x, y, rest=rest)
            if self.sub.done:
                self.fit_finished()
            return
        if self.state != "keys":
            return
        self._cursor = (x, y)
        if key in _NEAREST and not self._nearest(x, key):
            return
        self._act(key, x, y, rest)

    def _nearest(self, x: float, key: str) -> bool:
        """ap_nearest. False if it has asked which is meant."""
        if not self.apertures:
            return True
        best, count, delta = 0, 0, MAX_REAL
        for i, ap in enumerate(self.apertures, 1):
            d = abs(self._centre(ap) - x)
            if d < delta - DELTA:
                best, count, delta = i, 1, d
            elif d < delta + DELTA:
                count += 1
        self.current = best
        if count <= 1:
            return True
        near = " ".join(str(ap.number) for ap in self.apertures
                        if abs(self._centre(ap) - x) < delta + DELTA)
        self._print(f"Apertures {near} are equally near the cursor.\n")
        self._then = (key, x)
        self._ask("choose", f"Choose an aperture ({self.apertures[best - 1].number}): ")
        return False

    def _act(self, key: str, x: float, y: float, rest: str = "") -> None:
        """The switch of ap_edit, and what follows it."""
        status = True
        ap = self.apertures[self.current - 1] if self.current > 0 else None
        one = len(self.apertures) == 1 or not self.all
        if key == "?":
            self._print("".join(f"{k}\t{what}\n" for k, what in HELP))
        elif key == ":":
            status = self._colon(rest)
        elif key == ".":
            pass
        elif key == "+":
            self.current = min(len(self.apertures), self.current + 1)
        elif key == "-":
            self.current = min(len(self.apertures), max(1, self.current - 1))
        elif key == "a":
            self.all = not self.all
        elif key == "f":
            self._find_new()
        elif key in "mn":
            self._new_aperture(x, key == "m")
        elif key == "r":
            self.newgraph = True
        elif key == "w":
            pass
        elif key in "bcdeg ijlostuyz" and key != " " and ap is None:
            status = self._beep()
        elif key == "b":
            self._background(ap)
            return
        elif key == "c":
            for each in ([ap] if one else self.apertures):
                centre = self._center1d(self._centre(each))
                if centre is not None:
                    self._set_centre(each, centre)
        elif key == "d":
            if one:
                del self.apertures[self.current - 1]
                self.current = min(self.current, len(self.apertures))
            else:
                self.apertures = []
                self.current = 0
        elif key == "g":
            apertures = [ap] if not self.all else self.apertures
            s = self.settings
            apextract.recenter(apertures, self.profile, self.line, s["width"], s["radius"],
                               s["threshold"], s["apertures"], s["npeaks"], s["shift"])
        elif key == "i":
            self._ask("number", f"Aperture ({ap.number}) = ")
            return
        elif key == "j":
            self._ask("beam", f"Beam ({self._beam(ap)}) = ")
            return
        elif key == "l":
            for each in ([ap] if one else self.apertures):
                self._limits(each, float(f32(f32(x) - f32(self._centre(ap)))), each.high)
        elif key == "o":
            for each in self.apertures:
                if each is not ap:
                    each.number = None
            self._ask("order", f"Aperture ({ap.number}) = ")
            return
        elif key == "s":
            self._ask("shift", f"Center aperture {ap.number} (no)? ")
            return
        elif key == "u":
            for each in ([ap] if one else self.apertures):
                self._limits(each, each.low, float(f32(f32(x) - f32(self._centre(ap)))))
        elif key == "y":
            for each in ([ap] if one else self.apertures):
                low, high = apextract.ylevel(self.profile, self._centre(each), y, False,
                                             False, 0.0, float(-self.npts), float(self.npts))
                self._limits(each, low, high)
        elif key == "z":
            s = self.settings
            apextract.resize([ap] if not self.all else self.apertures, self.profile, self.line,
                             s["llimit"], s["ulimit"], s["ylevel"], s["peak"], s["bkg"],
                             s["r_grow"], s["avglimits"])
        elif key in "et":
            self._print(f"{key} is not ported here: the task traces and extracts after "
                        "this.\n")
            status = False
        elif key == "q":
            self.done = True
            return
        else:
            status = self._beep()
        self._after(status)

    def _beep(self) -> bool:
        self._print("Invalid or unrecognized command\a")
        return False

    def _after(self, status: bool) -> None:
        if self.newgraph and self._batch:
            self.graph()
        if status:
            self._status()

    _batch = False

    def answer(self, text: str | None) -> None:
        """What a prompt read: None at the end of the input."""
        if self.sub is not None:
            self.sub.answer(text)
            return
        state, self.state, self.prompt = self.state, "keys", ""
        words = (text or "").split()
        ap = self.apertures[self.current - 1] if self.current > 0 else None
        number = _integer(words[0]) if words else None
        if state == "choose":
            key, x = self._then
            if text is None or not words:
                self._act(key, x, self._cursor[1])
                return
            for i, each in enumerate(self.apertures, 1):
                if each.number == number:
                    self.current = i
                    self._act(key, x, self._cursor[1])
                    return
            self._ask("choose", f"Choose an aperture ({ap.number}): ")
            return
        if state == "number":
            if text is not None and number is not None:
                if number < 1:
                    self._print("Aperture numbers < 1 are not allowed: ")
                    self._ask("number", f"Aperture ({ap.number}) = ")
                    return
                if any(each is not ap and each.number == number for each in self.apertures):
                    self._print(f"Aperture {number} already used: ")
                    self._ask("number", f"Aperture ({ap.number}) = ")
                    return
                ap.number = number
                ap.beam = number
        elif state == "beam":
            if text is not None and number is not None:
                for each in self.apertures if self.all else [ap]:
                    each.beam = number
        elif state == "order":
            if text is not None and number is not None:
                ap.number = ap.beam = number
            self._sort()
            self._number()
            self.newgraph = True
        elif state == "shift":
            wx = self._cursor[0]
            if words and words[0].lower().startswith(("y", "t")):
                centre = self._center1d(wx)
                if centre is not None:
                    wx = centre
            one = len(self.apertures) == 1 or not self.all
            if one:
                self._set_centre(ap, wx)
            else:
                offset = float(f32(f32(wx) - f32(self._centre(ap))))
                for each in self.apertures:
                    self._set_centre(each, float(f32(f32(self._centre(each)) + f32(offset))))
        self._after(True)

    def _set_centre(self, ap: Aperture, centre: float) -> None:
        ap.centre = float(f32(f32(centre) - f32(ap.trace(np.array([float(self.line)]))[0])))

    def _limits(self, ap: Aperture, low: float, high: float) -> None:
        """ap_update."""
        ap.low, ap.high = min(low, high), max(low, high)

    def _new_aperture(self, x: float, centred: bool) -> None:
        wx = self._center1d(x) if centred else x
        if wx is None:
            return
        settings = self.settings
        if self.apertures and self.current > 0:
            model = self.apertures[self.current - 1]
            low, high, background = model.low, model.high, model.background
        else:
            low, high = settings["lower"], settings["upper"]
            background = self._default_background()
        new = Aperture(None, float(f32(wx)), low, high, float(self.line),
                       background=background)
        self.apertures.append(new)
        self.current = len(self.apertures)
        self._sort()
        self._number()

    def _default_background(self) -> icfit.FitSpec:
        s = self.settings
        return icfit.FitSpec(function=s["b_function"], order=s["b_order"],
                             sample=s["b_sample"], naverage=s["b_naverage"],
                             niterate=s["b_niterate"],
                             low_reject=s["b_low_reject"], high_reject=s["b_high_reject"],
                             grow=s["b_grow"])

    def _sort(self) -> None:
        """ap_sort, which keeps track of the current one."""
        if not self.apertures:
            return
        current = self.apertures[self.current - 1] if self.current > 0 else None
        self.apertures.sort(key=lambda ap: ap.centre,
                            reverse=self.settings["order"] == "decreasing")
        if current is not None:
            self.current = self.apertures.index(current) + 1

    def _number(self) -> None:
        """ap_ids, with no table: from the first numbered one, by the gaps of maxsep."""
        aps = self.apertures
        if not aps:
            return
        used = [ap.number for ap in aps if ap.number is not None]
        maxsep = f32(self.settings["maxsep"])
        first = next((i for i, ap in enumerate(aps) if ap.number is not None), None)
        if first is None:
            first = 0
            aps[0].number = aps[0].beam = 1
            used.append(1)
        elif aps[first].beam is None:
            aps[first].beam = aps[first].number

        def free(start: int, step: int) -> int:
            number = start
            while True:
                number += step
                if number != 0 and abs(number) not in used:
                    return number

        for j in range(first - 1, -1, -1):
            skip = int(abs(aps[j].centre - aps[j + 1].centre) / maxsep)
            number = free(aps[j + 1].number - skip, -1)
            aps[j].number = abs(number)
            aps[j].beam = abs(number)
            used.append(abs(number))
        for i in range(first + 1, len(aps)):
            if aps[i].number is None:
                skip = int(abs(aps[i].centre - aps[i - 1].centre) / maxsep)
                number = free(aps[i - 1].number + skip, 1)
                aps[i].number = abs(number)
                aps[i].beam = abs(number)
                used.append(abs(number))

    def _find_new(self) -> None:
        """ap_findnew: to nfind, the strongest not within minsep of what there is."""
        s = self.settings
        nfind = s["nfind"]
        if nfind <= len(self.apertures):
            return
        taken = [self._centre(ap) for ap in self.apertures]
        minsep = s["minsep"]
        peaks = apextract.find_peaks(self.profile, 0.0, 1, self.npts, minsep, -MAX_REAL)
        found = []
        for peak in peaks:
            if len(taken) + len(found) >= nfind:
                break
            if all(abs(peak - x) >= minsep for x in taken + found):
                found.append(float(peak))
        if self.current > 0:
            model = self.apertures[self.current - 1]
            low, high, background = model.low, model.high, model.background
        else:
            low, high = s["lower"], s["upper"]
            background = self._default_background()
        for peak in sorted(found):
            centre = self._center1d(peak)
            if centre is not None:
                self.apertures.append(Aperture(None, float(f32(centre)), low, high,
                                               float(self.line), background=background))
        self.current = 0
        self._sort()
        self._number()
        self.current = len(self.apertures)

    # --------------------------------------------------------- the background

    def _background(self, ap: Aperture) -> None:
        """b: icfit on the profile, about the aperture."""
        self._fitting = ap
        centre = self._centre(ap)
        x = np.arange(1, self.npts + 1, dtype=f32) - f32(centre)
        if self._ic_gt is None:
            # One GTOOLS for every b, its window that of the first aperture.
            wide = max(10.0, ap.high - ap.low)
            self._ic_gt = GTools()
            self._ic_gt.set_type("line")
            self._ic_gt.setr("xmin", ap.low - 2 * wide)
            self._ic_gt.setr("xmax", ap.high + 2 * wide)
        title = f"Set Background Subtraction for Aperture {ap.number}"
        self.sub = icfitsession.IcfitSession(
            x, self.profile, ap.background, None, title, self.axis, "", "line",
            ap.background_limits(self.npts), axes=self._graph_keys.get(id(ap), (None, "h"))[0],
            gkey=self._graph_keys.get(id(ap), (None, "h"))[1], gt=self._ic_gt)

    def fit_finished(self) -> None:
        """Icfit's q: the parameters stay, the fit does not."""
        spec = self.sub.spec
        self._graph_keys[id(self._fitting)] = (self.sub.axes, self.sub.gkey)
        self._print(self.sub.take())
        self.sub = None
        self._fitting.background = spec
        if self.all and len(self.apertures) > 1:
            for each in self.apertures:
                each.background = replace(spec)
        self.newgraph = True
        self._after(True)

    # ------------------------------------------------------- the colon commands

    def _colon(self, text: str) -> bool:
        """ap_colon: whether the status line is still to be printed."""
        words = text.split()
        if not words:
            return True
        if words[0].startswith("/"):
            self._print("the /commands are not ported: the mouse sets the window here\n")
            return False
        command = _strdic(words[0], _CMDS)
        value = words[1] if len(words) > 1 else None
        if command is None:
            self._print("Unrecognized or ambiguous command\a")
            return False
        if command == "show":
            self._show(value)
        elif command in ("parameters", "database", "logfile", "plotfile", "read", "write"):
            self._print(f":{command} is not ported: the task keeps the database and the log\n")
            return False
        elif command == "image":
            if value is None:
                self._print(f"image {self.image}")
                return False
            self._print(":image is not ported: one frame to a run\n")
            return False
        elif command in ("line", "nsum"):
            number = _integer(value)
            if number is None:
                self._print(f"{command} {getattr(self, command)}")
                return False
            if number != getattr(self, command):
                setattr(self, command, number)
                self.profile, self.line = apextract.getdata(self.frame, self.line, self.nsum)
                self.newgraph = True
        elif command in ("center", "lower", "upper"):
            return self._set_aperture(command, value)
        elif command == "title":
            if self.current > 0:
                ap = self.apertures[self.current - 1]
                if value is None:
                    self._print(f"title {ap.title}")
                    return False
                ap.title = text.split(None, 1)[1]
        else:
            return self._setting(command, value)
        return True

    def _set_aperture(self, command: str, value: str | None) -> bool:
        if self.current == 0:
            return True
        ap = self.apertures[self.current - 1]
        number = _float(value)
        if number is None:
            shown = {"center": ("center", self._centre(ap)), "lower": ("low", ap.low),
                     "upper": ("high", ap.high)}[command]
            self._print(f"{shown[0]} {_g(shown[1])}")
            return False
        if command == "center":
            if not self.all:
                self._set_centre(ap, number)
            else:
                offset = float(f32(f32(number) - f32(self._centre(ap))))
                for each in self.apertures:
                    self._set_centre(each, float(f32(f32(self._centre(each)) + f32(offset))))
        else:
            for each in self.apertures if self.all else [ap]:
                if command == "lower":
                    self._limits(each, number, each.high)
                else:
                    self._limits(each, each.low, number)
            if command == "lower" and self.all:
                return False                 # IRAF's :lower says nothing, with all on
        return True

    def _setting(self, name: str, value: str | None) -> bool:
        settings = self.settings
        if name in _BOOLEANS:
            if value is None:
                self._print(f"{name} {'yes' if settings[name] else 'no'}")
                return False
            settings[name] = value.lower().startswith(("y", "t")) or value == "+"
        elif name in _INTEGERS:
            number = _integer(value)
            if number is None:
                self._print(f"{name} {settings[name]}")
                return False
            settings[name] = number
        elif name in _STRINGS:
            if value is None:
                self._print(f"{name} {settings[name]}")
                return False
            settings[name] = value
        else:
            number = _float(value)
            if number is None:
                self._print(f"{name} {_g(settings[name])}")
                return False
            settings[name] = number
        return True

    def _show(self, file: str | None) -> None:
        """ap_show."""
        if not self.apertures:
            return
        lines = ["# APERTURES", "",
                 f"{'##ID':>4} {'BEAM':>4} {'CENTER':>7} {'LOW':>7} {'HIGH':>7} TITLE"]
        for ap in self.apertures:
            row = (f"{ap.number:4d} {self._beam(ap):4d} {_seven(ap.centre)} {_seven(ap.low)} "
                   f"{_seven(ap.high)}")
            lines.append(row + (f" {ap.title}" if ap.title else ""))
        text = "\n".join(lines) + "\n"
        if file is None:
            self._print(text)
        else:
            with open(file, "a", encoding="utf-8") as out:
                out.write(text)

    # -------------------------------------------------------------- the graph

    def graph(self) -> Graph:
        """ap_graph, and it is drawn."""
        window = (*with_room(np.array([1.0, float(self.npts)])), *with_room(self.profile))
        marks = [Mark(ap.number, self._centre(ap),
                      self._centre(ap) + ap.low, self._centre(ap) + ap.high)
                 for ap in self.apertures]
        sample = []
        if len(self.apertures) == 1:
            ap = self.apertures[0]
            centre = self._centre(ap)
            x1, x2 = window[:2]
            count = int(x2 - x1 + 2)
            x = x1 + np.arange(count) - centre
            try:
                inside = icfit.parse_sample(ap.background.sample, x)
            except ValueError:
                inside = np.zeros(x.shape, dtype=bool)
            for run in icfit._runs(inside):
                sample.append((float(x[run[0]] + centre), float(x[run[-1]] + centre)))
        self.newgraph = False
        return Graph("Define and Edit Apertures", self.params, window, self.profile, marks,
                     sample)

    # ------------------------------------------------------------- a whole run

    def run_commands(self, cursor: str, answers: str = "") -> None:
        """A cursor file, `x y wcs key [command]` a line, and the lines the prompts
        read, as IRAF takes them from the terminal. Both go to icfit while it is on."""
        lines = iter(cursor.splitlines())
        replies = iter(answers.splitlines())

        def next_cursor():
            for line in lines:
                words = line.split(None, 4)
                if len(words) >= 4 and not line.lstrip().startswith("#"):
                    return words
            return None

        self._batch = True
        self.graph()
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
                self.key("q")
                continue
            x, y, key = float(words[0]), float(words[1]), words[3]
            if key.startswith(":"):
                rest = key[1:] + " " + (words[4] if len(words) > 4 else "")
                self.key(":", rest=rest.strip())
            else:
                self.key(key, x, y)


def _integer(text: str | None) -> int | None:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _float(text: str | None) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _seven(value) -> str:
    return iraffmt.printf(float(f32(value)), "f", 2, width=7).rjust(7)
