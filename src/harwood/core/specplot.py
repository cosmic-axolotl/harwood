"""onedspec's specplot without its window: noao/onedspec/t_specplot.x, as IRAF 2.18
runs it.

Several spectra drawn together, each above the one before: every aperture of every
image named, each scaled and offset (by default to the first one's mean) and
stepped up by a separation that just clears one spectrum from the next. Keys and
colon commands move, shift, scale and relabel them; the log is a table of each
one's coordinates, scale and offset.

What follows from the code, kept:

* the layout scales each spectrum to the first's mean (or offsets it, when either
  mean is not positive), then the step is the largest maximum of one less the
  minimum of the next, times `fraction`;
* a spectrum's index is its position up the graph: deleting or adding one moves
  the indices of those above, and the step multiplies the index less one;
* everything is in real, as IRAF keeps the spectra (altmr: y * scale + offset).
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import iraffmt, shdr, splot
from . import units as unitsmod
from .gtools import GTools

f32 = np.float32
MAX_REAL = float(np.finfo(np.float32).max)
LABELS = ("none", "imname", "imtitle", "index", "user")
CMDS = ("show", "vshow", "step", "fraction", "move", "shift", "w0", "wpc", "velocity",
        "redshift", "offset", "scale", "xlpos", "ylpos", "label", "ulabel", "ptype",
        "units", "color")

#: specplot.key, for a panel's list of keys.
HELP = [("a  i", "add a spectrum after, before"), ("d  e", "delete, undelete"),
        ("f", "pixels / world"), ("l", "label at the cursor"), ("p", "move the label"),
        ("o", "renumber the gaps"), ("s", "shift: s t x y z, r redraw, q"),
        ("t", "two coordinates make a scale"), ("u", "a coordinate here"),
        ("v", "velocity about the cursor"), ("w", "window (gtools)"),
        ("x", "no layout"), ("y", "offset to a common mean"),
        ("z", "scale to a common mean"), ("r", "redraw"), ("q", "quit"),
        (":", "show vshow step fraction move shift w0 wpc velocity redshift offset "
              "scale xlpos ylpos label ulabel ptype units color")]


def _g(value) -> str:
    return iraffmt.g(float(f32(value)))


def strdic(word: str, words) -> int:
    """strdic: the unique abbreviation's index, 1-based; 0 for none."""
    if word in words:
        return words.index(word) + 1
    found = [i for i, w in enumerate(words, 1) if w.startswith(word)] if word else []
    return found[0] if len(found) == 1 else 0


class SpecplotError(RuntimeError):
    """What stops specplot altogether, as IRAF's task stops."""


@dataclass
class Spec:
    """specplot.h's structure: one spectrum, its layout and its label."""
    sh: shdr.Shdr
    index: int = 1
    w0: float = 0.0
    wpc: float = 0.0
    omean: f32 = f32(0)
    omin: f32 = f32(0)
    omax: f32 = f32(0)
    xscale: f32 = f32(1)
    xoffset: f32 = f32(0)
    scale: f32 = f32(1)
    offset: f32 = f32(0)
    mean: f32 = f32(0)
    min: f32 = f32(0)
    max: f32 = f32(0)
    x: np.ndarray = field(default=None, repr=False)
    y: np.ndarray = field(default=None, repr=False)
    xlpos: f32 = f32(1.02)
    ylpos: f32 = f32(0)
    color: int = 1
    imname: str = ""
    imtitle: str = ""
    ptype: str = "1"
    ulabel: str = ""
    label: str = ""

    @property
    def npts(self) -> int:
        return self.sh.sn


def scale_spectra(sps: list[Spec], step) -> None:
    """sp_scale: x = SX * xscale + xoffset, y = SY * scale + offset + (index-1) step."""
    step = f32(step)
    for sp in sps:
        sh = sp.sh
        sp.x = (sh.x * f32(sp.xscale) + f32(sp.xoffset)).astype(f32)
        offset = f32(f32(sp.offset) + f32(f32(sp.index - 1) * step))
        sp.y = (sh.y * f32(sp.scale) + offset).astype(f32)
        sp.mean = f32(f32(sp.omean * f32(sp.scale)) + offset)
        sp.min = f32(f32(sp.omin * f32(sp.scale)) + offset)
        sp.max = f32(f32(sp.omax * f32(sp.scale)) + offset)


def autolayout(sps: list[Spec], autoscale: bool, fraction, step):
    """sp_autolayout: each to the first's mean, and the step that separates them."""
    if len(sps) < 2:
        return step
    first = sps[0]
    scale, offset, a = f32(first.scale), f32(first.offset), f32(first.omean)
    for sp in sps[1:]:
        if autoscale and f32(a * sp.omean) > 0:
            sp.scale = f32(f32(a / sp.omean) * scale)
            sp.offset = offset
        else:
            sp.scale = scale
            sp.offset = f32(f32(f32(a - sp.omean) * scale) + offset)
    step = f32(-MAX_REAL)
    for prev, sp in zip(sps[:-1], sps[1:], strict=True):
        top = f32(f32(prev.omax * prev.scale) + prev.offset)
        bottom = f32(f32(sp.omin * sp.scale) + sp.offset)
        step = max(step, f32(top - bottom))
    return f32(f32(fraction) * step)


def set_labels(sps: list[Spec], labels: int) -> None:
    for sp in sps:
        sp.label = {1: "", 2: sp.imname, 3: sp.imtitle,
                    4: f"{sp.index:<4d}", 5: sp.ulabel}[labels]


@dataclass
class Specplot:
    """t_specplot's loop. `reads` the cursor's reads (x, y, wcs, key, command);
    `answers` what the prompts read (scan, from the terminal)."""

    params: dict
    answers: object = field(default_factory=lambda: iter(()))
    out: str = ""
    err: str = ""
    sps: list = field(default_factory=list)
    graphs: list = field(default_factory=list)
    drawn: list = field(default_factory=list)

    def __post_init__(self) -> None:
        p = self.params
        units = "".join(str(p.get("units", "")).split()) or "display"
        self.units = units
        self.step = f32(p.get("step", 0.0))
        self.fraction = f32(p.get("fraction", 1.0))
        self.labels = strdic(str(p.get("labels", "user")), LABELS) or 5
        self.gt = GTools()
        self.gt.type = 2
        self.gt.sysid = bool(p.get("sysid", True))
        self.gt.title = str(p.get("title", ""))
        self.wscale = True
        self.window = (0.0, 1.0, 0.0, 1.0)
        self.saved = None
        self._first = True
        self.fatal = ""

    def say(self, text: str) -> None:
        self.out += text

    def read(self):
        return next(self.reads, None)

    def scan(self):
        """scan(): a line the terminal gives, None at its end."""
        try:
            return next(self.answers)
        except StopIteration:
            return None

    # ------------------------------------------------------------- data

    def gdata(self, image: str, current: int) -> int:
        """sp_gdata: each aperture (and band) of an image as a spectrum, inserted
        after `current`; the new current."""
        from . import ranges

        p = self.params
        img = splot.open_image(image, int(p.get("dispaxis", 1)),
                               tuple(p.get("nsum", (1, 1))))
        smw = img.smw
        if self._first:
            self.xlpos = f32(p.get("xlpos", 1.02))
            self.ylpos = f32(p.get("ylpos", 0.0))
            self.ptype = str(p.get("ptype", "1"))
            self.trans = str(p.get("transform", "none"))
            self._first = False
        scale = self._value("scale", img, "Error in scale parameter")
        offset = self._value("offset", img, "Error in offset parameter")
        aps = ranges.parse(str(p.get("apertures", "")), -ranges.RNG_MAXINT, ranges.RNG_MAXINT)
        bands = ranges.parse(str(p.get("bands", "1")), -ranges.RNG_MAXINT, ranges.RNG_MAXINT)
        for j in range(1, smw.nbands + 1):
            if smw.format != shdr.ND and j not in bands:
                continue
            for i in range(1, smw.nspec + 1):
                if smw.format == shdr.ND:
                    k = (i - 1) % smw.llen[1] + 1
                    line = (i - 1) // smw.llen[1] + j
                    if k not in aps or line not in bands:
                        continue
                sh = shdr.shdr_open(smw, img.data, image, i, j, None)
                if smw.format != shdr.ND and sh.ap not in aps:
                    continue
                with contextlib.suppress(shdr.ShdrError):
                    sh.set_units(self.units)
                sp = Spec(sh)
                sp.w0 = float(sh.x[0])
                sp.wpc = float(f32(f32(sh.x[sh.sn - 1] - sh.x[0]) / f32(sh.sn - 1)))
                y = sh.y
                if self.trans == "log":
                    pos = y[y > 0]
                    if pos.size:
                        lo = f32(pos.min())
                        y = np.log10(np.maximum(y, lo)).astype(f32)
                        sh.y = y
                total = f32(0)
                for v in sh.y:
                    total = f32(total + v)
                sp.omean = f32(total / f32(sh.sn))
                sp.omin, sp.omax = f32(sh.y.min()), f32(sh.y.max())
                sp.scale, sp.offset = scale, offset
                sp.xlpos, sp.ylpos = self.xlpos, self.ylpos
                sp.imname = f"{sh.image}{sh.imsec}({sh.ap})"
                sp.imtitle = sh.title
                sp.ptype = self.ptype
                current = self.add(sp, current)
        return current

    def _value(self, name: str, img, error: str) -> f32:
        text = "".join(str(self.params.get(name, "1." if name == "scale" else "0."))
                       .split())
        if not text:
            raise ValueError(error)
        if text.startswith("@"):
            lines = Path(text[1:]).read_text(encoding="utf-8").splitlines()
            if len(lines) < len(self.sps) + 1:
                raise ValueError(f"Error reading {name} file")
            return f32(float(lines[len(self.sps)].split()[0]))
        if text[0].isalpha():
            return f32(float(img.header[text]))
        try:
            return f32(float(text))
        except ValueError:
            raise ValueError(error) from None

    def add(self, sp: Spec, current: int) -> int:
        """sp_add: after element `current`, the indices above moved up."""
        sps = self.sps
        for s in sps[current:]:
            s.index += 1
        sps.insert(current, sp)
        if len(sps) == 1:
            sp.index = 1
        elif current == 0:
            sp.index = sps[1].index - 1
        else:
            sp.index = sps[current - 1].index + 1
        return current + 1

    def delete(self, current: int) -> None:
        """sp_delete."""
        sps = self.sps
        for s in sps[current:]:
            s.index -= 1
        del sps[current - 1]

    # -------------------------------------------------------------- graph

    def viewport(self) -> tuple[float, float]:
        lines = int(self.gt.sysid) + len(self.gt.title_block())
        return 1.0 - 2 * 11 * 0.0125, 1.0 - 5 * 0.0294 - (lines + 1) * 0.0294

    def plot(self) -> None:
        """sp_plot: the window over all the spectra, gtools' limits over it, and
        each spectrum with its label."""
        xmin, xmax, ymin, ymax = MAX_REAL, -MAX_REAL, MAX_REAL, -MAX_REAL
        n = 0
        for sp in self.sps:
            if self.wscale:
                xmin = min(xmin, float(sp.x[0]), float(sp.x[-1]))
                xmax = max(xmax, float(sp.x[0]), float(sp.x[-1]))
            else:
                n = max(n, sp.npts)
                xmin, xmax = 1.0, float(n)
            ymin = min(ymin, float(sp.min))
            ymax = max(ymax, float(sp.max))
        if xmin > xmax:
            xmin, xmax = 0.0, 1.0
        if ymin > ymax:
            ymin, ymax = 0.0, 1.0
        gt = self.gt
        self.window = gt.swind((xmin, xmax), (ymin, ymax))
        x1, x2, y1, y2 = self.window
        dx, dy = f32(f32(x2) - f32(x1)), f32(f32(y2) - f32(y1))
        frame = {"window": self.window, "title": gt.title, "params": gt.params,
                 "xlabel": gt.label("x"), "spectra": []}
        for sp in self.sps:
            xs = sp.x if self.wscale else np.arange(1, sp.npts + 1, dtype=f32)
            lx = f32(f32(sp.xlpos * dx) + f32(x1))
            ly = f32(f32(sp.ylpos * dy) + sp.mean)
            frame["spectra"].append({"x": xs.copy(), "y": sp.y.copy(), "index": sp.index,
                                     "label": sp.label, "at": (lx, ly),
                                     "ptype": sp.ptype, "color": sp.color})
        self.graphs.append(frame)
        self.drawn = []

    def _params_line(self) -> None:
        self.gt.params = f"Separation step = {_g(self.step)}" if self.gt.sysid else ""

    def nearest(self, wx, wy) -> int:
        """sp_nearest: the spectrum with the point nearest the cursor, in NDC."""
        best, found = MAX_REAL, 0
        vp = self.viewport()
        cx, cy = self.gt.ndc([wx], [wy], self.window, vp)
        for i, sp in enumerate(self.sps, start=1):
            px, py = self.gt.ndc(sp.x, sp.y, self.window, vp)
            r2 = ((px - cx[0]) ** 2 + (py - cy[0]) ** 2).astype(f32)
            k = int(np.argmin(r2))
            if r2[k] < best:
                best, found = float(r2[k]), i
        return found

    # --------------------------------------------------------------- run

    def run(self, images: list[str], reads) -> None:
        """t_specplot."""
        self.reads = iter(reads)
        current = 0
        for name in images:
            try:
                current = self.gdata(name, current)
            except (OSError, ValueError, shdr.ShdrError) as error:
                self.err += f"Warning: {error}\n"
        p = self.params
        if p.get("autolayout", True):
            self.step = autolayout(self.sps, bool(p.get("autoscale", True)),
                                   self.fraction, self.step)
        scale_spectra(self.sps, self.step)
        ulabels = str(p.get("ulabels", ""))
        if ulabels:
            try:
                lines = Path(ulabels).read_text(encoding="utf-8").splitlines()
            except OSError:
                lines = None
            if lines is not None:
                for k, sp in enumerate(self.sps):
                    sp.ulabel = lines[k] if k < len(lines) else ""
        set_labels(self.sps, self.labels)
        gt = self.gt
        if p.get("xlabel"):
            gt.xlabel, gt.xunits = str(p["xlabel"]), ""
        elif self.sps:
            sh = self.sps[0].sh
            gt.xlabel, gt.xunits = (sh.un.label, sh.un.units) if sh.un.label else \
                (sh.label, sh.units)
        gt.ylabel = str(p.get("ylabel", ""))
        for name in ("xmin", "xmax", "ymin", "ymax"):
            gt.setr(name, p.get(name))
        key, wx, wy, cmd = "r", 0.0, 0.0, ""
        needs = set("dlposuvtxyz")      # the keys that go to nospec_ without spectra
        while True:
            if key in ("q", "I"):
                break
            if self.sps or key not in needs:
                try:
                    redraw = self.key(key, wx, wy, cmd)
                except SpecplotError as error:
                    self.fatal = str(error)
                    return
                if key == ":" and not self.sps:
                    redraw = False
                if redraw:
                    self._params_line()
                    self.plot()
            if not self.sps:
                self.say("No spectra defined\a")
            r = self.read()
            if r is None:
                break
            wx, wy, _, key, cmd = r
        log = str(p.get("logfile", "")).strip()
        if log:
            self.vshow_lines = self.show_lines(verbose=True)

    def key(self, key, wx, wy, cmd) -> bool:
        sps = self.sps
        if key == ":":
            if cmd.startswith("/"):
                said, redraw = self.gt.colon(cmd, self.window)
                self.say(said)
                return bool(redraw)
            current = self.nearest(wx, wy) if "[" not in cmd else 1
            return self.colon(cmd, current)
        if key in ("a", "i"):
            i = self.nearest(wx, wy) if sps else 0
            if key == "i":
                i = max(0, i - 1)
            self.say("Spectrum: ")
            line = self.scan()
            if line is not None and line.split():
                try:
                    self.gdata(line.split()[0], i)
                    set_labels(sps, self.labels)
                    scale_spectra(sps, self.step)
                    return True
                except (OSError, ValueError, shdr.ShdrError) as error:
                    self.err += f"Warning: {error}\n"
            return False
        if key == "d":
            i = self.nearest(wx, wy)
            sp = sps[i - 1]
            self.delete(i)
            set_labels(sps, self.labels)
            scale_spectra(sps, self.step)
            self.saved = sp
            self.drawn.append(("erase", sp.index))
            return False
        if key == "e":
            if self.saved is not None:
                i = max(0, self.nearest(wx, wy) - 1) if sps else 0
                self.add(self.saved, i)
                set_labels(sps, self.labels)
                scale_spectra(sps, self.step)
                self.saved = None
                return True
            return False
        if key == "f":
            if self.wscale:
                self.gt.xlabel, self.gt.xunits = "Pixels", ""
                self.wscale = False
            else:
                if sps:
                    sh = sps[0].sh
                    self.gt.xlabel, self.gt.xunits = sh.un.label, sh.un.units
                self.wscale = True
            return True
        if key in ("l", "p"):
            i = self.nearest(wx, wy)
            sp = sps[i - 1]
            self.say(f"Spectrum {sp.index}:  Mark position for label ('q' to cancel)")
            r = self.read()
            if r is not None and r[3] != "q":
                x1, x2, y1, y2 = self.window
                sp.xlpos = f32(f32(f32(r[0]) - f32(x1)) / f32(f32(x2) - f32(x1)))
                sp.ylpos = f32(f32(f32(r[1]) - sp.mean) / f32(f32(y2) - f32(y1)))
                if key == "l":
                    self.say(f"Spectrum {sp.index}: Label = ")
                    line = self.scan()
                    if line is not None:
                        sp.ulabel = line
                        sp.label = line
                self.drawn.append(("text", (f32(r[0]), f32(r[1]), sp.label)))
            self.say("\n")
            return False
        if key == "o":
            redraw = False
            for i, sp in enumerate(sps, start=1):
                if sp.index != i:
                    sp.index = i
                    redraw = True
            if redraw:
                set_labels(sps, self.labels)
                scale_spectra(sps, self.step)
            return redraw
        if key == "r":
            return True
        if key == "s":
            return self._shift(wx, wy)
        if key == "t":
            return self._scale_x(wx, wy)
        if key == "u":
            i = self.nearest(wx, wy)
            sp = sps[i - 1]
            self.say(f"X coordinate ({_g(wx)}): ")
            line = self.scan()
            if line is not None:
                try:
                    wx1 = f32(float(line.split()[0]))
                except (ValueError, IndexError):
                    return False
                sp.xoffset = f32(f32(sp.xoffset + wx1) - f32(wx))
                scale_spectra([sp], self.step)
                return True
            return False
        if key == "v":
            return self._velocity(wx)
        if key == "w":
            self.say("window:")
            r = self.read()
            if r is None:
                return False
            if r[3] == "e":
                self.say("again:")
                r2 = self.read()
                return r2 is not None and self.gt.window2((r[0], r[1]), (r2[0], r2[1]),
                                                          self.window)
            said, redraw = self.gt.window1(r[3], r[0], r[1], self.window)
            self.say(said)
            return redraw
        if key == "x":
            for sp in sps:
                sp.scale, sp.offset = f32(1), f32(0)
            scale_spectra(sps, self.step)
            return True
        if key in ("y", "z"):
            self.step = autolayout(sps, key == "z", self.fraction, self.step)
            scale_spectra(sps, self.step)
            return True
        if key == "?":
            return False
        self.say("\a")
        return False

    def _shift(self, wx, wy) -> bool:
        """s: the nearest spectrum shifted and scaled by the cursor until q."""
        sp = self.sps[self.nearest(wx, wy) - 1]
        self.say(f"Shift spectrum {sp.index}: (q, r, s, t, x, y, z)")
        wx, wy = f32(wx), f32(wy)
        vel = sp.sh.un.kind == unitsmod.VEL
        while True:
            r = self.read()
            if r is None:
                break
            wx1, wy1, k = f32(r[0]), f32(r[1]), r[3]
            if k == "q":
                break
            if k in ("s", "t"):
                if wy != sp.offset:
                    if k == "t":
                        if vel:
                            sp.xoffset = f32(f32(sp.xoffset + wx1) - wx)
                        else:
                            sp.xscale = f32(f32(sp.xscale * wx1) / wx)
                        wx = wx1
                    sp.scale = f32(f32(sp.scale * f32(wy1 - sp.offset))
                                   / f32(wy - sp.offset))
                    scale_spectra([sp], self.step)
                    wy = wy1
            elif k in ("x", "z"):
                if vel:
                    sp.xoffset = f32(f32(sp.xoffset + wx1) - wx)
                else:
                    sp.xscale = f32(f32(sp.xscale * wx1) / wx)
                if k == "z":
                    sp.offset = f32(f32(sp.offset + wy1) - wy)
                    wy = wy1
                scale_spectra([sp], self.step)
                wx = wx1
            elif k == "y":
                sp.offset = f32(f32(sp.offset + wy1) - wy)
                scale_spectra([sp], self.step)
                wy = wy1
            elif k == "r":
                self._params_line()
                self.plot()
            self.say(f"Shift spectrum {sp.index}: (q, r, s, t, x, y, z)")
        self.say("\n")
        return False

    def _scale_x(self, wx, wy) -> bool:
        """t: a linear coordinate scale from two positions and what they are."""
        sp = self.sps[self.nearest(wx, wy) - 1]
        wx = f32(wx)
        self.say(f"X coordinate ({_g(wx)}): ")
        line = self.scan()
        wyv = _first(line, wx)
        self.say("Mark another position")
        r = self.read()
        wx1 = f32(r[0]) if r is not None else wx
        self.say(f"X coordinate ({_g(wx1)}): ")
        line = self.scan()
        wy1 = _first(line, wx1)
        if wx == wx1:
            return False
        xs = sp.x           # the coordinates as drawn, with the shift u gave them
        n = sp.npts - 1

        def find(value):
            i = 1
            if sp.wpc > 0:
                while i < n and value < xs[i - 1]:
                    i += 1
            return i

        i, j = find(wx), find(wx1)
        px = f32(f32(i) + f32(f32(wx - xs[i - 1]) / f32(xs[i] - xs[i - 1])))
        px1 = f32(f32(j) + f32(f32(wx1 - xs[j - 1]) / f32(xs[j] - xs[j - 1])))
        sp.wpc = float(f32(f32(wyv - wy1) / f32(px - px1)))
        sp.w0 = float(f32(wyv - f32(f32(sp.wpc) * f32(px - f32(1)))))
        sp.sh.x = (f32(sp.w0) + np.arange(sp.npts, dtype=f32) * f32(sp.wpc)).astype(f32)
        sp.xscale, sp.xoffset = f32(1), f32(0)
        scale_spectra([sp], self.step)
        return True

    def _velocity(self, wx) -> bool:
        """v: every spectrum's coordinates as a velocity about the cursor."""
        try:
            for k, sp in enumerate(self.sps):
                sh = sp.sh
                if k == 0:
                    ref = float(unitsmod.convert([float(f32(wx))], sh.un,
                                                 unitsmod.decode("angstroms"),
                                                 real=True)[0])
                    self.units = f"km/s {_g(ref)} angstroms"
                new = unitsmod.decode(self.units)
                sh.x = unitsmod.convert(sh.x, sh.un, new, real=True).astype(f32)
                sh.un = new
                if k == 0:
                    self.gt.xlabel, self.gt.xunits = new.label, new.units
                sp.w0 = float(sh.x[0])
                sp.wpc = float(f32(f32(sh.x[sh.sn - 1] - sh.x[0]) / f32(sh.sn - 1)))
                sp.xscale, sp.xoffset = f32(1), f32(0)
                scale_spectra([sp], self.step)
            return True
        except ValueError as error:
            self.err += f"Warning: {error}\n"
            return False

    # --------------------------------------------------------- the colon

    def colon(self, cmdstr: str, current: int) -> bool:
        """sp_colon."""
        sps = self.sps
        words = cmdstr.split()
        word = words[0] if words else ""
        rest = words[1:]
        # the spectrum: [n] names it by its index, [] (or a word in it) every one;
        # without brackets it is the one nearest the cursor
        if "[" in word:
            name, _, idx = word.partition("[")
            word = name
            current = 0
            m = re.match(r"\s*(-?\d+)", idx)
            if m:
                index = int(m.group(1))
                current = next((i for i, s in enumerate(sps, 1) if s.index == index),
                               len(sps) + 1)
                if current > len(sps):
                    self.say(f"Spectrum {index} not defined")
                    return False
        ncmd = strdic(word, CMDS)
        value = _number(rest[0]) if rest else None
        one = sps[current - 1] if current > 0 else None
        if ncmd in (1, 2):
            lines = self.show_lines(verbose=ncmd == 2)
            if rest:
                with open(rest[0], "a", encoding="utf-8") as fd:
                    fd.writelines(line + "\n" for line in lines)
            else:
                self.say("\n".join(lines) + "\n")
            return False
        if ncmd == 3:
            if value is None:
                self.say(f"step {_g(self.step)}")
                return False
            self.step = f32(value)
            scale_spectra(sps, self.step)
            return True
        if ncmd == 4:
            if value is None:
                self.say(f"fraction {_g(self.fraction)}")
                return False
            self.fraction = f32(value)
            return False
        if ncmd == 5:
            return self._move(current, rest)
        if ncmd == 6:
            if value is None:
                return False
            j = int(value)
            chosen = (sps[current - 1:] if j > 0 else sps[:current][::-1]) \
                if current > 0 else sps
            if current > 0 and j == 0:
                return False
            for sp in chosen:
                sp.index += j
                scale_spectra([sp], self.step)
                set_labels([sp], self.labels)
            return bool(chosen)
        field_cmds = {7: "w0", 8: "wpc", 11: "offset", 12: "scale", 13: "xlpos",
                      14: "ylpos"}
        if ncmd in field_cmds:
            return self._field(field_cmds[ncmd], value, one)
        if ncmd in (9, 10):
            return self._velocity_or_redshift(ncmd == 9, value, one)
        if ncmd == 15:
            if not rest:
                self.say(f"labels {LABELS[self.labels - 1]}")
                return False
            n = strdic(rest[0], LABELS)
            if n == 0:
                self.say(f"Unknown label type: {rest[0]}")
            else:
                self.labels = n
                set_labels(sps, self.labels)
            return False
        if ncmd == 16:
            # gargstr after the command's word: the rest, its leading blank kept
            text = re.match(r"\s*\S+(.*)$", cmdstr).group(1)
            if one is not None:
                if not rest:
                    self.say(f"ulabel[{one.index}] {one.ulabel}")
                else:
                    one.ulabel = text
                    if self.labels == 5:
                        one.label = text
            else:
                if not rest:
                    self.say("ulabel:" + "".join(f" {s.index}={s.ulabel}" for s in sps))
                else:
                    for s in sps:
                        s.ulabel = text
                        if self.labels == 5:
                            s.label = text
            return False
        if ncmd == 17:
            if one is not None:
                if not rest:
                    self.say(f"ptype[{one.index}] {one.ptype}")
                else:
                    one.ptype = rest[0]
            elif not rest:
                self.say("ptype:" + "".join(f" {s.index}={s.ptype}" for s in sps))
            else:
                for s in sps:
                    s.ptype = rest[0]
            return False
        if ncmd == 18:
            # gargstr: the rest with its leading blank, which un_open does not take
            # off -- " nm" is no abbreviation, and the units are unknown
            text = re.match(r"\s*\S+(.*)$", cmdstr).group(1)
            try:
                new = unitsmod.decode(text)
                if new.kind == 0 and any(s.sh.un.kind != 0 for s in sps):
                    raise SpecplotError("Cannot convert between selected units\a")
                for i, sp in enumerate(sps, start=1):
                    if current > 0 and i != current:
                        continue
                    sh = sp.sh
                    sh.x = unitsmod.convert(sh.x, sh.un, new, real=True).astype(f32)
                    sh.un = new
                    sp.w0 = float(sh.x[0])
                    sp.wpc = float(f32(f32(sh.x[sh.sn - 1] - sh.x[0]) / f32(sh.sn - 1)))
                    sp.xscale, sp.xoffset = f32(1), f32(0)
                    scale_spectra([sp], self.step)
                    if i == 1:
                        self.units = text
                        self.gt.xlabel, self.gt.xunits = new.label, new.units
                return True
            except ValueError as error:
                self.err += f"Warning: {error}\n"
                return False
        if ncmd == 19:
            if one is not None:
                if value is None:
                    self.say(f"color[{one.index}] {one.color}")
                else:
                    one.color = int(value)
            elif value is None:
                self.say("color:" + "".join(f" {s.index}={s.color}" for s in sps))
            else:
                for s in sps:
                    s.color = int(value)
            return False
        self.say(f"Unknown command: {cmdstr}\a")
        return False

    def _field(self, name: str, value, one) -> bool:
        sps = self.sps

        def shown(sp):
            if name == "w0":
                return f32(f32(f32(sp.w0) * sp.xscale) + sp.xoffset)
            if name == "wpc":
                return f32(f32(sp.wpc) * sp.xscale)
            return getattr(sp, name)

        def setv(sp):
            if name == "w0":
                sp.xoffset = f32(f32(value) - f32(f32(sp.w0) * sp.xscale))
            elif name == "wpc":
                sp.wpc = float(f32(value))
                sp.sh.x = (f32(sp.w0) + np.arange(sp.npts, dtype=f32)
                           * f32(sp.wpc)).astype(f32)
                sp.xscale, sp.xoffset = f32(1), f32(0)
            else:
                setattr(sp, name, f32(value))
            if name not in ("xlpos", "ylpos"):
                scale_spectra([sp], self.step)

        if one is not None:
            if value is None:
                self.say(f"{name}[{one.index}] {_g(shown(one))}")
                return False
            setv(one)
            return True
        if value is None:
            self.say(f"{name}:" + "".join(f" {s.index}={_g(shown(s))}" for s in sps))
            return False
        for s in sps:
            setv(s)
        return True

    def _velocity_or_redshift(self, velocity: bool, value, one) -> bool:
        sps = self.sps
        if not sps:
            return False
        un1 = sps[0].sh.un
        name = "velocity" if velocity else "redshift"
        targets = [one] if one is not None else sps

        def shown(sp):
            if un1.kind == unitsmod.VEL:
                return f32(sp.xoffset) if velocity else f32(sp.xoffset / f32(un1.scale))
            if velocity:
                if sp.xscale == 1:
                    return f32(0)
                ref = unitsmod.decode(f"km/s {_g(sp.w0)} {un1.units}")
                return f32(unitsmod.convert([float(f32(f32(sp.w0) * sp.xscale))], un1,
                                            ref, real=True)[0])
            if un1.kind == unitsmod.WAVE:
                return f32(sp.xscale - f32(1))
            return f32(f32(f32(1) / sp.xscale) - f32(1))

        if value is None:
            if one is not None:
                self.say(f"{name}[{one.index}] {_g(shown(one))}")
            else:
                self.say(f"{name}:" + "".join(f" {s.index}={_g(shown(s))}" for s in sps))
            return False
        for sp in targets:
            if un1.kind == unitsmod.VEL:
                sp.xoffset = f32(value) if velocity else f32(f32(value) * f32(un1.scale))
            elif velocity:
                ref = unitsmod.decode(f"km/s {_g(sp.w0)} {un1.units}")
                w = f32(unitsmod.convert([float(f32(value))], ref, un1, real=True)[0])
                sp.xscale = f32(w / f32(sp.w0))
            elif un1.kind == unitsmod.WAVE:
                sp.xscale = f32(f32(1) + f32(value))
            else:
                sp.xscale = f32(f32(1) / f32(f32(1) + f32(value)))
            scale_spectra([sp], self.step)
        return True

    def _move(self, current: int, rest) -> bool:
        if not rest:
            return False
        index = int(float(rest[0]))
        if current <= 0:
            self.say("\a")
            return False
        sps = self.sps
        sp = sps[current - 1]
        if index == sp.index:
            return False
        sp.index = index
        del sps[current - 1]
        for s in sps[current - 1:]:
            s.index -= 1
        i = 0
        while i < len(sps) and index > sps[i].index:
            i += 1
        sps.insert(i, sp)
        for j in range(i, len(sps) - 1):
            if sps[j].index == sps[j + 1].index:
                sps[j + 1].index += 1
        if self.labels == 4:
            set_labels(sps, 4)
        scale_spectra(sps, self.step)
        return True

    def show_lines(self, verbose: bool) -> list[str]:
        """sp_show and sp_vshow: the system's line, the step, and a line a spectrum."""
        out = [sysid(), "", f"Step = {_g(self.step)}", "",
               f"   {'Image Name':>16s} {'W0':>7s} {'WPC':>7s} {'Offset':>7s} "
               f"{'Scale':>7s}   Title"]
        for sp in self.sps:
            w0 = f32(f32(f32(sp.w0) * sp.xscale) + sp.xoffset)
            out.append(f"{sp.index:2d} {sp.imname:>16s} {_g7(w0)} "
                       f"{_g7(f32(f32(sp.wpc) * sp.xscale))} {_g7(sp.offset)} "
                       f"{_g7(sp.scale)}   {sp.imtitle}")
        if verbose:
            out += ["", f"   {'Image Name':>16s} {'Mean':>9s} {'DW0':>9s} {'Z':>9s} "
                        f"{'V(km/s)':>9s}"]
            un = self.sps[0].sh.un if self.sps else None
            for sp in self.sps:
                z = v = f32(0)
                if un is not None and un.kind == unitsmod.VEL:
                    z, v = f32(sp.xoffset / f32(un.scale)), f32(sp.xoffset)
                elif un is not None and sp.xscale != 1 and un.kind in (
                        unitsmod.WAVE, unitsmod.FREQ, unitsmod.ENERGY):
                    z = f32(sp.xscale - f32(1)) if un.kind == unitsmod.WAVE else \
                        f32(f32(f32(1) / sp.xscale) - f32(1))
                    ref = unitsmod.decode(f"km/s {_g(sp.w0)} {un.units}")
                    v = f32(unitsmod.convert([float(f32(f32(sp.w0) * sp.xscale))], un,
                                             ref, real=True)[0])
                out.append(f"{sp.index:2d} {sp.imname:>16s} {_g9(sp.omean)} "
                           f"{_g9(sp.xoffset)} {_g9(z)} {_g9(v)}")
        return out


def sysid() -> str:
    """sysid's line: harwood's version, the user and the machine, the time."""
    import getpass
    import socket

    from .provenance import _version

    return (f"harwood {_version()} {getpass.getuser()}@{socket.gethostname()} "
            f"{iraffmt.cnvtime()}")


def _g7(v) -> str:
    return iraffmt.printf(float(f32(v)), "g", None, 7).rjust(7)


def _g9(v) -> str:
    return iraffmt.printf(float(f32(v)), "g", None, 9).rjust(9)


def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _first(line, default):
    if line is None:
        return f32(default)
    words = line.split()
    if not words:
        return f32(default)
    try:
        return f32(float(words[0]))
    except ValueError:
        return f32(default)
