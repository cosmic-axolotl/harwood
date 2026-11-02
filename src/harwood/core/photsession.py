"""phot's and center's interactive mode, without the window.

apphot/phot/apphot.x answers keys at the image cursor: centre, sky and
magnitudes where the cursor is or for the next star of the list, kept or
written as asked. It remembers what it has fitted and refits only what the
cursor moved off, and that bookkeeping decides what each key writes, so it
is kept here as IRAF keeps it. The window (core.display) and a cursor file
(IRAF's icommands) drive the same session:

    space  f    centre, sky, magnitudes at the cursor; space writes them
    c t s       centre at the cursor; sky at the cursor; sky at the centre
    a           average the sky over several positions (t each, q ends)
    p o         magnitudes with the sky as it is; o writes them
    m n l       move to, measure the next star of the list; all the rest
    r e q       rewind the list; the last errors; quit
    :m N :n N   move to, measure star N
    :name value set a parameter (:annulus 12, :apertures 3,5, :calgorithm gauss)
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from . import apphot

#: The keys, as the window's Keys panel lists them (the docstring above, key
#: by key).
HELP = [("space", "centre, sky and magnitudes at the cursor, written"),
        ("f", "the same, not written"), ("c", "centre at the cursor"),
        ("t", "sky at the cursor"), ("s", "sky at the centre"),
        ("a", "average the sky over several positions: t each, q ends"),
        ("p", "magnitudes with the sky as it is"), ("o", "the same, written"),
        ("m", "move to the next star of the list"), ("n", "measure the next star"),
        ("l", "measure all the rest"), ("r", "rewind the list"),
        ("e", "the last errors"), ("q", "quit"),
        (":m N  :n N", "move to, measure star N"),
        (":name value", "set a parameter: :annulus 12, :apertures 3,5")]


@dataclass
class Row:
    """One written measurement, with IRAF's two numbers: the running id
    and the star's place in the list (0 for one the cursor chose)."""

    id: int
    lid: int
    xinit: float
    yinit: float
    center: apphot.Center
    sky: apphot.Sky
    mags: apphot.Apertures | None
    radii: list[float]
    profile: object = None
    #: radprof: what the fit was asked for, and the #K cards a change since the last row wrote.
    settings: dict | None = None
    cards: list = field(default_factory=list)


@dataclass
class Session:
    data: object
    dpars: apphot.DataPars
    cpars: apphot.CenterPars
    spars: apphot.SkyPars
    radii: list[float] = field(default_factory=lambda: [3.0])
    zmag: float = 25.0
    stars: list[tuple[float, float]] = field(default_factory=list)
    task: str = "phot"
    cursor: object = None              #: histplot/radplot's graphics cursor
    skyfile: object = None
    rows: list[Row] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    done: bool = False
    fitter: object = None              #: radprof's

    def __post_init__(self) -> None:
        self.new_center = self.new_sky = self.new_mag = True
        self.newlist = False
        self.ltid = 0
        self.wx = self.wy = None              # the previous cursor position
        self.xlist = self.ylist = 0.0
        self.center = apphot.Center(0.0, 0.0)
        self.sky = apphot.Sky()
        self.mags: apphot.Apertures | None = None
        self.sky_at = self.mag_at = (None, None)
        self.stid = 1
        self.pending_average: list[apphot.Sky] | None = None
        self.profile = None
        self.pending_cards: list = []

    # ------------------------------------------------------------ fitting

    def _fit_center(self, x: float, y: float) -> None:
        self.center = apphot.center(self.data, x, y, self.cpars, self.dpars)

    def _fit_sky(self, x: float, y: float) -> None:
        self.sky = apphot.fitsky(self.data, x, y, self.spars, self.dpars, self.cursor,
                                 self.skyfile)
        self.sky_at = (x, y)

    def _measure(self) -> None:
        if self.task == "center":
            return
        if self.task == "radprof":
            self.profile = self.fitter.fit(self.data, self.center.x, self.center.y, self.sky,
                                           sorted(self.radii), self.zmag, self.dpars)
            self.mags = self.profile.measured
            self.mag_at = (self.center.x, self.center.y)
            return
        self.mags = apphot.apertures(self.data, self.center.x, self.center.y,
                                     sorted(self.radii), self.sky, self.zmag, self.dpars)
        self.mag_at = (self.center.x, self.center.y)

    def _write(self, lid: int) -> None:
        settings = None
        if self.fitter is not None:
            settings = {name: getattr(self.fitter, name)
                        for name in ("radius", "step", "order", "kreject", "nreject")}
        self.rows.append(Row(self.stid, lid, self._init[0], self._init[1], self.center,
                             self.sky, self.mags, sorted(self.radii), self.profile, settings,
                             self.pending_cards))
        self.pending_cards = []
        self.stid += 1

    # ------------------------------------------------------------ the keys

    def _moved(self, wx: float, wy: float) -> tuple[bool, float, float]:
        """apnew: has the cursor left the object being measured? Near the
        list star just fetched it snaps onto it."""
        delta = self.dpars.fwhmpsf * self.dpars.scale
        if not self.newlist:
            moved = self.wx is None or wx != self.wx or wy != self.wy
        elif ((abs(self.xlist - wx) <= delta and abs(self.ylist - wy) <= delta)
              or (wx == self.wx and wy == self.wy)):
            wx, wy, moved = self.xlist, self.ylist, False
        else:
            self.newlist, moved = False, True
        return moved, wx, wy

    def key(self, key: str, wx: float = 0.0, wy: float = 0.0) -> None:
        if self.pending_average is not None:
            self._average(key, wx, wy)
            return
        moved, wx, wy = self._moved(wx, wy)
        if moved:
            self.new_center = self.new_sky = self.new_mag = True
        self.wx, self.wy = wx, wy
        if key == "q":
            self.done = True
        elif key == "e":
            self.messages.append(self._errors())
        elif key == "r":
            self.ltid = 0
        elif key in ("m", "n"):
            self._next(self.ltid + 1, measure=key == "n")
        elif key == "l":
            for number in range(self.ltid + 1, len(self.stars) + 1):
                x, y = self.stars[number - 1]
                self._init = (x, y)
                self._fit_center(x, y)
                self._fit_sky(self.center.x, self.center.y)
                self._measure()
                self._write(number)
            self.ltid = len(self.stars) + 1
        elif key == "c":
            if self.new_center:
                self._init = (wx, wy)
                self._fit_center(wx, wy)
            self.new_center = False
        elif key == "t":
            if self.new_sky or (wx, wy) != self.sky_at:
                self._fit_sky(wx, wy)
            self.new_sky = False
        elif key == "a":
            self.pending_average = []
            self.messages.append("Measure sky around several cursor positions "
                                 "(t=measure, q=quit)")
        elif key == "s":
            if self.new_sky or (self.center.x, self.center.y) != self.sky_at:
                self._fit_sky(self.center.x, self.center.y)
            self.new_sky = False
        elif key in ("p", "o"):
            if self.new_center:
                self._init = (wx, wy)
                self._fit_center(wx, wy)
            if self.new_mag or (self.center.x, self.center.y) != self.mag_at:
                self._measure()
            self.new_center = self.new_mag = False
            if key == "o":
                self._write(self.ltid if self.newlist else 0)
        elif key in ("f", " "):
            if self.new_center:
                self._init = (wx, wy)
                self._fit_center(wx, wy)
            if self.new_sky or (self.center.x, self.center.y) != self.sky_at:
                self._fit_sky(self.center.x, self.center.y)
            if self.new_mag or (self.center.x, self.center.y) != self.mag_at:
                self._measure()
            self.new_center = self.new_sky = self.new_mag = False
            if key == " ":
                self._write(self.ltid if self.newlist else 0)
        else:
            self.messages.append("Unknown or ambiguous keystroke command")

    def _next(self, number: int, measure: bool) -> None:
        """apgscur and what m, n, :m and :n do with it."""
        if not self.stars:
            self.messages.append("No coordinate list")
            return
        if number > len(self.stars) or number < 1:
            self.messages.append("End of coordinate list, use r key to rewind")
            return
        self.ltid = number
        self.xlist, self.ylist = self.stars[number - 1]
        self.newlist = True
        if not measure:
            self.new_center = self.new_sky = self.new_mag = True
            return
        self._init = (self.xlist, self.ylist)
        self._fit_center(self.xlist, self.ylist)
        self._fit_sky(self.center.x, self.center.y)
        self._measure()
        self._write(self.ltid)
        self.new_center = self.new_sky = self.new_mag = False

    def _average(self, key: str, wx: float, wy: float) -> None:
        """ap_avsky: t measures the sky at the cursor, q averages them."""
        import math

        if key == "t":
            self._fit_sky(wx, wy)
            self.pending_average.append(self.sky)
            return
        if key != "q":
            return
        skies, self.pending_average = self.pending_average, None

        def mean(values):
            good = [v for v in values if v is not None and not math.isnan(v)]
            return sum(good) / len(good) if good else float("nan")

        ier = next((s.ier for s in skies if s.ier), apphot.SKY_OK)
        self.sky = apphot.Sky(mean([s.value for s in skies]), mean([s.sigma for s in skies]),
                              mean([s.skew for s in skies]), sum(s.nsky for s in skies),
                              sum(s.nreject for s in skies), ier)
        self.new_sky = False

    def _errors(self) -> str:
        parts = [apphot.MESSAGES.get(self.center.ier, "") if self.center.ier else "",
                 apphot.MESSAGES.get(self.sky.ier, "") if self.sky.ier else ""]
        if self.mags is not None and self.mags.ier:
            parts.append(apphot.MESSAGES.get(self.mags.ier, ""))
        return "; ".join(p for p in parts if p) or "No errors"

    # ------------------------------------------------------------ colon

    def colon(self, text: str) -> None:
        command = text.strip().lstrip(":").strip()
        if not command:
            return
        name, _, rest = command.partition(" ")
        rest = rest.strip()
        if name in ("m", "n"):
            number = int(rest) if rest else self.ltid + 1
            # radprof's :m goes on into the measuring that follows in apradprof.x, as the
            # key m does not: it writes the star, and :m is :n.
            self._next(number, measure=name == "n" or self.task == "radprof")
            return
        if name == "show":
            self.messages.append(self._show())
            return
        if name == "apertures":
            if rest:
                self.radii = [float(v) for v in rest.replace(",", " ").split()]
                self.new_mag = True
            self.messages.append(f"apertures = {','.join(f'{r:g}' for r in self.radii)}")
            return
        if name == "zmag":
            if rest:
                self.zmag = float(rest)
                self.new_mag = True
            self.messages.append(f"zmag = {self.zmag:g}")
            return
        if self.fitter is not None and name in ("radius", "step", "stepsize", "order",
                                                "kreject", "nreject"):
            self.messages.append(self._profile_parameter(name, rest))
            return
        for pars, refit in ((self.dpars, "all"), (self.cpars, "center"),
                            (self.spars, "sky")):
            names = {f.name for f in fields(pars)}
            if name in names:
                if rest:
                    current = getattr(pars, name)
                    setattr(pars, name, _typed(rest, current))
                    if refit in ("all", "center"):
                        self.new_center = True
                    self.new_sky = self.new_mag = True
                self.messages.append(f"{name} = {getattr(pars, name)}")
                return
        self.messages.append(f"Unknown or ambiguous colon command ':{name}'")

    def _profile_parameter(self, name: str, rest: str) -> str:
        """radprof's colon commands: a new radius or step is a new profile's length."""
        fitter = self.fitter
        field_name = "step" if name == "stepsize" else name
        if rest:
            if field_name in ("order", "nreject"):
                setattr(fitter, field_name, int(float(rest)))
            else:
                setattr(fitter, field_name, np.float32(float(rest)))
            fitter.npts = int(np.float32(fitter.radius / fitter.step)) + 1
            self.new_mag = True
            if self.stid > 1:
                self.pending_cards.append((field_name, getattr(fitter, field_name)))
        return f"{name} = {getattr(fitter, field_name):g}"

    def _show(self) -> str:
        values = {**{f.name: getattr(self.dpars, f.name) for f in fields(self.dpars)},
                  **{f.name: getattr(self.cpars, f.name) for f in fields(self.cpars)},
                  **{f.name: getattr(self.spars, f.name) for f in fields(self.spars)}}
        values["apertures"] = ",".join(f"{r:g}" for r in self.radii)
        values["zmag"] = self.zmag
        return "  ".join(f"{k}={v}" for k, v in values.items())


def _typed(text: str, current):
    if isinstance(current, bool):
        return text.lower() in ("yes", "y", "true", "1", "+")
    if isinstance(current, int):
        return int(float(text))
    if isinstance(current, float) or current is None:
        if text.upper() == "INDEF":
            return None
        return float(text)
    return text


def read_commands(path) -> list[tuple[float, float, str]]:
    """IRAF's image cursor file: 'x y wcs key' or 'x y wcs :command'."""
    from pathlib import Path

    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        words = line.split(None, 3)
        if len(words) < 4 or line.lstrip().startswith("#"):
            continue
        key = words[3].strip()
        key = {"\\040": " ", "\\40": " "}.get(key, key)
        out.append((float(words[0]), float(words[1]), key if key.startswith(":")
                    else key[:1]))
    return out
