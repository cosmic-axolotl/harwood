"""plotspec -- RVSAO's graph of a vector to look at and edit by eye, as RVSAO 2.8.5
draws it: xcsao's e (the spectrum, edited before it is correlated again) and p
(each template's correlation, whose peak is chosen with the cursor).

The vector smoothed, on gtools' window (the data's range and 3% around it) under
a title block of the task's log heading and the vector's name. The keys: r redraws,
h hides and shows the heading, w says where the cursor is, g smooths again, z z
zooms between two points, u unzooms, x cancels a zoom or a delete half made, d d
replaces the points between two by a straight line, , replaces one by the cursor's
height, s sets the half width of the peak search, p chooses the highest point
within it of where the cursor is, ? lists the keys, q leaves. From Util/{plotspec,
replot}.x. Kept, because they decide what is drawn and said:

* g and s read their answers into the buffer the title came from: after them, h
  shows the answer as the title (an answer of 3 is the lines "3" and "").
* z's first press says where the graph starts, not where the cursor is.
* , changes the vector but not what is drawn, which is the smoothed copy.
* Any key it does not know says where the cursor is; q is gt_gcur's end of file.

Not kept: d's second press calls fillpix without its last argument and ends in a
segmentation violation, always; here it fills. @ makes a picture.

Translated from RVSAO, Copyright (c) 1991-2022 Smithsonian Astrophysical
Observatory, whose notice is in LICENSES/RVSAO-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import iraffmt
from .rvsao import f32

CH = 0.0294         #: stdgraph's character height, as glb_set_viewport counts it
KEYS = ("@  make hard copy of screen\n"
        "c  show x and y coordinates\n"
        "d  delete data between 1st and 2nd positions\n"
        "g  smooth plotted data\n"
        "h  toggle headings\n"
        "p  select correlation peak\n"
        "q  leave plot\n"
        "r  replot current graph\n"
        "s  reset peak search half-width\n"
        "u  unzoom\n"
        "x  cancel zoom or delete after 1st char\n"
        "z  zoom between 1st and 2nd positions\n"
        ",  replace single pixel with value at cursor\n"
        ">  overwrite spectrum with edited one\n"
        "?  this menu (space works, too)\n")


def _f(spec: str, value) -> str:
    return iraffmt.Format(spec)(value)


@dataclass
class Frame:
    """One replot: gtools' window and viewport, the title block's lines, the label,
    and the vector drawn."""
    window: tuple[float, float, float, float]
    viewport: tuple[float, float, float, float]
    title: list[str]
    xlabel: str
    x: np.ndarray
    y: np.ndarray

    def ndc(self, x, y):
        x1, x2, y1, y2 = self.window
        v1, v2, v3, v4 = self.viewport
        return (v1 + (x - x1) / (x2 - x1) * (v2 - v1),
                v3 + (y - y1) / (y2 - y1) * (v4 - v3))

    def svg(self, width: int = 1000, height: int = 760) -> str:
        def esc(t):
            return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        v1, v2, v3, v4 = self.viewport
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
               f'height="{height}" viewBox="0 0 {width} {height}">',
               f'<rect width="{width}" height="{height}" fill="white"/>',
               f'<rect x="{v1 * width:.1f}" y="{(1 - v4) * height:.1f}" '
               f'width="{(v2 - v1) * width:.1f}" height="{(v4 - v3) * height:.1f}" '
               'fill="none" stroke="black"/>']
        pts = []
        for a, b in zip(self.x, self.y, strict=True):
            nx, ny = self.ndc(float(a), float(b))
            pts.append(f"{nx * width:.1f},{(1 - ny) * height:.1f}")
        out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="black"/>')
        for k, line in enumerate(self.title):
            out.append(f'<text x="{width / 2:.1f}" y="{(k + 1.2) * CH * height:.1f}" '
                       f'font-size="13" text-anchor="middle">{esc(line)}</text>')
        out.append(f'<text x="{(v1 + v2) / 2 * width:.1f}" y="{0.97 * height:.1f}" '
                   f'font-size="13" text-anchor="middle">{esc(self.xlabel)}</text>')
        out.append("</svg>\n")
        return "\n".join(out)


class Plotspec:
    """plotspec on a vector (edited in place): `read_key` gives the next cursor read
    (x, y, wcs, key) or None; `read_line` the next line typed; what it says goes
    through `say`. After `run`, `peak` is the point p chose (z[1], 0 for none) and
    `edited` whether d or , changed the vector."""

    def __init__(self, pix, xvec, name: str, xlabel: str, nsmooth: int, task: str,
                 params: dict, read_key, read_line, say) -> None:
        from .rvsao import loghead

        self.pix = pix                       # the vector itself: d and , edit it
        self.xvec = np.asarray(xvec, f32)
        self.name, self.xlabel = name, xlabel
        self.p = params
        self.read_key, self.read_line, self.say = read_key, read_line, say
        self.linebuf = loghead(task)         # the title's buffer, which g and s reuse
        self.title, self.subtitle = self.linebuf, name
        self.heading = True
        self.ismooth = -(nsmooth + 1) if nsmooth < 0 else nsmooth
        self.wait = nsmooth >= 0
        self.smpix = np.array(pix, f32)
        self._smooth()
        n = self.xvec.size
        self.ipix1, self.ipix2, self.npix = 1, n, n
        self.dfirst = self.zfirst = True
        self.d1 = self.ipx1 = 0
        self.reversed = bool(self.xvec[0] > self.xvec[-1])
        self.peak = 0
        self.edited = False
        self.frames: list[Frame] = []
        self.hardcopies: list[Frame] = []

    def _smooth(self) -> None:
        from .rvsao import smooth

        self.smpix = np.array(self.pix, f32)
        smooth(self.smpix, self.smpix.size, self.ismooth)

    # ------------------------------------------------------------ the graph

    def _title(self) -> list[str]:
        lines = []
        for part in (self.title, self.subtitle):
            if part:
                lines.extend(part.split("\n"))
        return lines

    def draw(self) -> Frame:
        """replot: gascale over what is shown, gt_swind (3% each side), gt_labax."""
        from .gtools import GTools

        a, b = self.ipix1 - 1, self.ipix1 - 1 + self.npix
        x = np.array(self.xvec[a:b], f32)
        y = np.array(self.smpix[a:b], f32)
        x1, x2 = float(x[0]), float(x[-1])
        if x1 > x2:
            x1, x2 = x2, x1
        gt = GTools()
        window = gt.swind((x1, x2), (float(y.min()), float(y.max())))
        title = self._title()
        top = 1.0 - (max(len(title), 1) + 1) * CH
        frame = Frame(window, (9 * 0.0125, 1.0 - 9 * 0.0125, 5 * CH, top), title,
                      self.xlabel, x, y)
        self.frames.append(frame)
        return frame

    # ------------------------------------------------------------ the keys

    def run(self) -> None:
        self.draw()
        if not self.wait:
            return
        while True:
            read = self.read_key()
            if read is None or self.press(read[0], read[1], read[3]):
                break

    def _ix(self, wx: float) -> int:
        """The point the cursor is on, as plotspec finds it (the ends beyond)."""
        xv = self.xvec
        i1, i2 = self.ipix1, self.ipix2
        ix = 0
        if self.reversed:
            if wx >= xv[i1 - 1]:
                return i1
            if wx <= xv[i2 - 1]:
                return i2
            for i in range(i1, i2):
                if wx < xv[i - 1] and wx >= xv[i]:
                    return i + 1 if wx - xv[i] > 0.5 else i
        else:
            if wx <= xv[i1 - 1]:
                return i1
            if wx >= xv[i2 - 1]:
                return i2
            for i in range(i1, i2):
                if xv[i - 1] <= wx < xv[i]:
                    return i if wx - xv[i - 1] > 0.5 else i + 1
        return ix

    def press(self, x: float, y: float, key: str) -> bool:
        """One cursor read: True when plotspec leaves."""
        if key == "q":                          # gt_gcur: q is the end of file
            return True
        wx, wy = f32(x), f32(y)
        ix = self._ix(wx)
        say = self.say
        newgraph = False
        if key == "r":
            newgraph = True
        elif key == "@":
            say(f"Making hard copy on {self.p.get('plotter', 'stdplot')}\n")
            self.hardcopies.append(self.draw())
            self.frames.pop()
        elif key == "h":
            if self.heading:
                self.title = self.subtitle = ""
            else:
                self.title, self.subtitle = self.linebuf, self.name
            self.heading = not self.heading
            newgraph = True
        elif key == "w":
            say(f"Wavelength {_f('%8.2f', float(wx))}, Pixel {ix}: "
                f"{_f('%8.2f', float(wy))}\n")
        elif key == "g":
            say(f"Number of times to smooth spectrum ({self.ismooth}): ")
            line = self.read_line()
            if line is not None:
                self.linebuf = line + "\n"
            if line:
                value = _ctoi(line)
                if value is not None:
                    self.ismooth = value
                say(f"Smoothing spectrum {self.ismooth} times\n")
                self._smooth()
                newgraph = True
        elif key == ",":
            say(f"replacing pixel {ix} with {float(wy):.7f}\n")
            ip = min(max(ix, 1), len(self.pix))
            self.pix[ip - 1] = wy
            self.edited = True
            newgraph = True
        elif key == "d":
            if self.dfirst:
                self.d1 = ix
                say(f"d again to delete data from pixel {self.d1}\n")
                self.dfirst = False
            else:
                from .rvsao import fillpix

                self.dfirst = True
                say(f"filling from pixel {self.d1} to pixel {ix}\n")
                fillpix(self.pix, self.d1, ix)
                self._smooth()
                newgraph = True
                self.edited = True
        elif key == "x":
            if not self.zfirst:
                say("zoom cancelled\n")
                self.zfirst = True
            if not self.dfirst:
                say("delete cancelled\n")
                self.dfirst = True
        elif key == "z":
            if self.zfirst:
                self.ipx1 = ix
                self.zfirst = False
                say(f"z again to zoom from {_f('%5.0f', float(self.xvec[self.ipix1 - 1]))}"
                    "\n")
            else:
                self.zfirst = True
                self.ipix1, self.ipix2 = sorted((self.ipx1, ix))
                self.npix = self.ipix2 - self.ipix1 + 1
                say("zooming from " + self._range() + "\n")
                newgraph = True
        elif key == "u":
            self.ipix1, self.ipix2 = 1, self.xvec.size
            self.npix = self.xvec.size
            say("unzooming back to " + self._range() + "\n")
            newgraph = True
        elif key in ("?", " "):
            say(KEYS)
        elif key in ("s", "S"):
            width = int(self.p.get("pksrch", 25))
            say(f"Half-width in pixels for peak search ({width}) = \n")
            line = self.read_line()
            if line is not None:
                self.linebuf = line + "\n"
                value = _ctoi(line)
                if value is not None:
                    self.p["pksrch"] = value
        elif key in ("p", "P"):
            ix0, wx0 = ix, wx
            width = int(self.p.get("pksrch", 25))
            i1 = max(ix - width + 1, 1)
            i2 = min(ix + width, self.ipix2)
            xmax = self.pix[ix - 1]
            for i in range(i1, i2 + 1):
                if self.pix[i - 1] >= xmax:
                    xmax = self.pix[i - 1]
                    ix = i
            self.peak = ix
            say(f"Max peak at {ix} = {_f('%8.2f', float(self.xvec[ix - 1]))} "
                f"({ix0} = {_f('%8.2f', float(wx0))})\n")
        elif key == "Q":
            return True
        else:
            say(f"x,y: {_f('%10.3f', float(wx))} {_f('%10.4g', float(wy))}\n")
        if newgraph:
            self.draw()
        return False

    def _range(self) -> str:
        a = float(self.xvec[self.ipix1 - 1])
        b = float(self.xvec[self.ipix2 - 1])
        if a > b:
            a, b = b, a
        return (f"{_f('%.0f', a)} to {_f('%.0f', b)} (pix {self.ipix1} - "
                f"{self.ipix2})")


def _ctoi(text: str) -> int | None:
    from .rvsao import Scan

    value = Scan(text).d()
    return None if value is None else int(value)
