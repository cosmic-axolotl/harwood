"""implot without the window: the state and the commands of IRAF's implot.

pkg/plot/t_implot.x keeps a current image, a current line or column, how
many are averaged, a step, overplotting, log scaling and fixed ranges, and
answers keys at a graphics cursor. This module is that, so the window's
panel, a cursor file (`--coords`, IRAF's graphics cursor list) and the
tests drive the same code:

    keys    l c a e j k m n o p r s <space> q
    colon   :a N  :c N [M]  :l N [M]  :i image  :o  :log+ :log-  :step N
            :solid  :w wcs  :f format  :x x1 x2  :y y1 y2

Cursor positions are in the plot's own coordinates, x along the plotted
vector and y in data values; the image line or column a key means (l in a
line plot, c in a column plot) is read off the right-hand axis, which maps
the plot's y range onto 1 .. the other dimension, as imp_tran does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

f32 = np.float32


@dataclass
class Plot:
    """What is drawn: the curves (x, y, label) and the window."""

    curves: list[tuple[np.ndarray, np.ndarray, str]] = field(default_factory=list)
    title: str = ""
    xlabel: str = ""
    xrange: tuple[float, float] | None = None
    yrange: tuple[float, float] | None = None
    logy: bool = False
    marks: list[tuple[float, float, float, float]] = field(default_factory=list)


class Implot:
    def __init__(self, images: list[str], line: int | None = None, wcs: str = "logical",
                 step: int = 0, reader=None) -> None:
        from . import images as imageio

        self.images = list(images)
        self.read = reader or (lambda name: imageio.read(name))
        self.index = 0
        self.wcs = wcs
        self.lineplot = True
        self.navg = self.p_navg = 1
        self.overplot = False
        self.logscale = False
        self.fixed_x: tuple[float, float] | None = None
        self.fixed_y: tuple[float, float] | None = None
        self.format = ""
        self.solid = False
        self.messages: list[str] = []
        self.status: list[str] = []
        self.pending: tuple[str, float, float] | None = None
        self.done = False
        self.plot = Plot()
        self._open(self.images[0])
        self.line = (max(1, min(self.nlines, (self.nlines + 1) // 2)) if line is None
                     else line)
        self.col = (self.ncols + 1) // 2
        self.step = step if step and step >= 1 else max(1, self.nlines // 10)
        self._draw()

    # ---------------------------------------------------------- the image

    def _open(self, name: str) -> None:
        data, header = self.read(name)
        data = np.asarray(data, dtype=f32)
        if data.ndim == 0 or data.size == 0:
            raise ValueError("image has no pixels")
        if data.ndim == 1:
            data = data[None, :]
        elif data.ndim > 2:
            data = data.reshape((-1,) + data.shape[-2:])[0]
        self.name, self.data, self.header = name, data, header
        self.nlines, self.ncols = data.shape
        if self.nlines == 1:
            self.lineplot = True

    def vector(self) -> tuple[np.ndarray, np.ndarray]:
        """imp_getvector: the average of navg lines (or columns) from the
        current one, in single precision, against the chosen coordinates."""
        if self.lineplot:
            y1 = max(1, min(self.nlines, self.line))
            y2 = max(1, min(self.nlines, self.line + self.navg - 1))
            total = np.zeros(self.ncols, dtype=f32)
            for i in range(y1, y2 + 1):
                total = (total + self.data[i - 1]).astype(f32)
            width = y2 - y1 + 1
            if width > 1:
                total = (total * f32(f32(1.0) / f32(width))).astype(f32)
            x = self._coords(1, np.arange(1, self.ncols + 1), (y1 + y2) / 2.0)
            return x, total
        x1 = max(1, min(self.ncols, self.col))
        x2 = max(1, min(self.ncols, self.col + self.navg - 1))
        width = x2 - x1 + 1
        if width > 1:
            norm = f32(1.0) / f32(width)
            block = self.data[:, x1 - 1:x2]
            sums = np.zeros(self.nlines, dtype=f32)
            for k in range(width):
                sums = (sums + block[:, k]).astype(f32)
            values = (sums * norm).astype(f32)
        else:
            values = self.data[:, x1 - 1].astype(f32)
        y = self._coords(2, np.arange(1, self.nlines + 1), (x1 + x2) / 2.0)
        return y, values

    def _coords(self, axis: int, pixels: np.ndarray, other: float) -> np.ndarray:
        """plt_wcs: logical pixels, physical through LTV/LTM, or world."""
        pixels = pixels.astype(float)
        h = self.header
        if self.wcs == "logical" or h is None:
            return pixels.astype(f32)
        ltm = float(h.get(f"LTM{axis}_{axis}", 1.0))
        ltv = float(h.get(f"LTV{axis}", 0.0))
        physical = (pixels - ltv) / ltm
        has_wcs = any(f"{key}{axis}" in h for key in ("CRVAL", "CRPIX", "CDELT", "CTYPE")) \
            or f"CD{axis}_{axis}" in h
        if self.wcs == "physical" or not has_wcs:
            # Without a world system in the header, IRAF's world is the physical.
            return physical.astype(f32)
        crval = float(h.get(f"CRVAL{axis}", 0.0))
        crpix = float(h.get(f"CRPIX{axis}", 0.0))
        cd = float(h.get(f"CD{axis}_{axis}", h.get(f"CDELT{axis}", 1.0)))
        return (crval + (pixels - crpix) * cd).astype(f32)

    # ------------------------------------------------------------ drawing

    def _label(self) -> str:
        if self.nlines <= 1:
            return str(self.header.get("OBJECT", "")).strip() if self.header else ""
        kind, n = ("lines", self.nlines) if self.lineplot else ("columns", self.ncols)
        at = self.line if self.lineplot else self.col
        if self.navg > 1:
            i1 = max(1, min(n, at))
            i2 = max(1, min(n, at + self.navg - 1))
            return f"Average of {kind} {i1} to {i2} of {self.name}"
        return f"{'Line' if self.lineplot else 'Column'} {at} of {self.name}"

    def _draw(self, erase_last: bool = False) -> None:
        x, y = self.vector()
        label = self._label()
        if self.overplot and self.plot.curves:
            if erase_last:
                self.plot.curves[-1] = (x, y, label)
            else:
                self.plot.curves.append((x, y, label))
            self.overplot = False
        else:
            self.plot = Plot(curves=[(x, y, label)], title=label,
                             xlabel=self.wcs, logy=self.logscale)
        self.plot.xrange = self.fixed_x
        self.plot.yrange = self.fixed_y
        self.x, self.y = x, y

    def window(self) -> tuple[float, float, float, float]:
        """The plot window, as gascale leaves it: the vector's extremes, or
        the fixed ranges."""
        x, y = self.x, self.y
        x1, x2 = self.fixed_x or (float(x.min()), float(x.max()))
        lo, hi = min(x1, x2), max(x1, x2)
        inside = (x >= lo) & (x <= hi)
        values = y[inside] if inside.sum() >= 2 else y
        y1, y2 = self.fixed_y or (float(values.min()), float(values.max()))
        if y1 == y2:
            y1, y2 = 0.99 * y1, 1.01 * y2
        return x1, x2, y1, y2

    def _tran(self, x: float, y: float, other: int) -> tuple[int, int]:
        """imp_tran: the nearest point of the vector to x, and y on the
        right-hand axis's 1 .. other."""
        _, _, y1, y2 = self.window()
        if y1 == y2:
            py = round(other / 2.0)
        else:
            py = int(math.floor(float(f32(f32(other - 1) / f32(y2 - y1) * f32(y - y1)
                                          + f32(1))) + 0.5))
        px = int(np.argmin(np.abs(f32(x) - self.x))) + 1
        return px, py

    # ------------------------------------------------------------ the keys

    def key(self, key: str, x: float = 0.0, y: float = 0.0) -> None:
        if self.pending is not None:
            first, x0, y0 = self.pending
            self.pending = None
            self._second(first, x0, y0, x, y)
            return
        if key in ("a", "e", "p", "s"):
            self.pending = (key, x, y)
            self.status = ["again:"]
            return
        if key == "q":
            self.done = True
        elif key == "l":
            if self.lineplot:
                _, line = self._tran(x, y, self.nlines)
            else:
                line, _ = self._tran(x, y, self.ncols)
            self.navg = self.p_navg
            self.line = max(1, min(self.nlines, line)) - (self.navg - 1) // 2
            self._show_line()
        elif key == "c":
            if self.nlines == 1:
                self.status = ["\a"]
                return
            if self.lineplot:
                col, _ = self._tran(x, y, self.nlines)
            else:
                _, col = self._tran(x, y, self.ncols)
            self.navg = self.p_navg
            self.col = max(1, min(self.ncols, col)) - (self.navg - 1) // 2
            self._show_col()
        elif key in ("j", "k"):
            self.navg = self.p_navg
            self.overplot = True
            shift = self.step if key == "k" else -self.step
            if self.lineplot:
                self.line += shift
                self._show_line(erase=True)
            else:
                self.col += shift
                self._show_col(erase=True)
        elif key in ("m", "n"):
            target = self.index - 1 if key == "m" else self.index + 1
            if not 0 <= target < len(self.images):
                return
            try:
                self._open(self.images[target])
            except Exception as error:        # IRAF warns and keeps the image
                self.messages.append(f"Error opening image '{self.images[target]}': {error}")
                return
            self.index = target
            self._show_line() if self.lineplot else self._show_col()
        elif key == "o":
            self.overplot = True
        elif key == "r":
            self._draw()
        elif key == " ":
            self._coordinates(x, y)
        elif key == "?":
            self.status = ["l line  c column  a average  e expand  j k move  m n image  "
                           "o overplot  p profile  r redraw  s statistics  space value  "
                           "q quit"]
        else:
            self.status = ["\a"]

    def _show_line(self, erase: bool = False) -> None:
        self.lineplot = True
        self.line = max(1, min(self.nlines, self.line))
        self._draw(erase_last=erase)

    def _show_col(self, erase: bool = False) -> None:
        if self.nlines == 1:
            self.status = ["\a"]
            return
        self.lineplot = False
        self.col = max(1, min(self.ncols, self.col))
        self._draw(erase_last=erase)

    def _second(self, key, x1, y1, x2, y2) -> None:
        if key == "a":
            if abs(x2 - x1) > abs(y2 - y1):
                self.navg = int(abs(x2 - x1) + 1)
                if self.lineplot:
                    self.col = round(min(x1, x2))
                    self._show_col()
                else:
                    self.line = round(min(x1, x2))
                    self._show_line()
            else:
                other = self.nlines if self.lineplot else self.ncols
                _, p1 = self._tran(x1, y1, other)
                _, p2 = self._tran(x2, y2, other)
                self.navg = abs(p2 - p1) + 1
                if self.lineplot:
                    self.line = min(p1, p2)
                    self._show_line()
                else:
                    self.col = min(p1, p2)
                    self._show_col()
        elif key == "e":
            self.fixed_x = (x1, x2)
            self.fixed_y = None if abs(y1 - y2) < 0.01 * abs(self.window()[3]
                                                             - self.window()[2]) \
                else (y1, y2)
            self._draw()
            self.fixed_x = self.fixed_y = None
        elif key == "s":
            self._statistics(x1, x2)
        elif key == "p":
            self.status = profile(self.x, self.y, x1, y1, x2, y2)
            self.messages += self.status

    def _statistics(self, x1: float, x2: float) -> None:
        from .iraffmt import printf

        npix = self.y.size
        i1 = max(1, min(npix, round(x1)))
        i2 = max(1, min(npix, round(x2)))
        if i1 > i2:
            i1, i2 = i2, i1
        elif i1 == i2:
            i2 = i1 + 1
        values = self.y[i1 - 1:i2]
        n = values.size
        mean, sigma = _aavg(values)
        median = _amed(values)
        total = f32(np.sum(values.astype(np.float64)))
        line = (f"median={printf(median)}, mean={printf(mean)}, rms={printf(sigma)}, "
                f"sum={printf(float(total))}, npix={n}")
        self.status = [line]
        self.messages.append(line)

    def _coordinates(self, x: float, y: float) -> None:
        """plt_wcscoord: the nearest pixel, its coordinates and its value."""
        from .iraffmt import printf

        px, _ = self._tran(x, y, self.nlines if self.lineplot else self.ncols)
        col, line = (px, self.line) if self.lineplot else (self.col, px)
        value = printf(float(self.y[px - 1]))
        if self.wcs == "logical":
            text = f"pixel=[{col},{line}] value={value}"
        else:
            cx = float(self._coords(1, np.array([col]), line)[0])
            cy = float(self._coords(2, np.array([line]), col)[0])
            if self.wcs == "physical":
                text = (f"pixel=[{col},{line}], physical=[{round(cx)},{round(cy)}], "
                        f"value={value}")
            else:
                text = (f"pixel=[{col},{line}], world=[{printf(cx)},{printf(cy)}] "
                        f"value={value}")
        self.status = [text]
        self.messages.append(text)

    # ------------------------------------------------------------ colon

    def colon(self, text: str) -> None:
        command = text.strip().lstrip(":").strip()
        if command.startswith("o"):
            self.overplot = True
            command = command[1:].strip()
            if not command:
                return
        words = command.split()
        if not words:
            return
        name, args = words[0], words[1:]
        try:
            numbers = [float(a) for a in args]
        except ValueError:
            numbers = []
        if name in ("a",) or (name.startswith("a") and name[1:].isdigit()):
            value = args[0] if args else name[1:]
            self.p_navg = max(1, int(float(value))) if value else 1
        elif name.startswith("log"):
            self.logscale = command.rstrip().endswith("+")
            self._draw()
        elif name in ("l", "c") or (name[:1] in "lc" and name[1:].isdigit()):
            try:
                values = [int(float(v)) for v in ([name[1:]] if name[1:] else [])
                          + args[:2]]
            except ValueError:
                values = []
            if not values:
                self.status = [f"':{name[0]}' takes numbers: :{name[0]} 250, or "
                               f":{name[0]} 250 260 for their average"]
                return
            n = self.nlines if name[0] == "l" else self.ncols
            if len(values) == 1:
                at = max(1, min(n, values[0]))
                self.navg = self.p_navg
                at -= (self.navg - 1) // 2
            else:
                i1, i2 = (max(1, min(n, v)) for v in values)
                at, self.navg = min(i1, i2), max(1, abs(i2 - i1) + 1)
            if name[0] == "l":
                self.line = at
                self._show_line()
            else:
                self.col = at
                self._show_col()
        elif name == "i":
            target = " ".join(args)
            try:
                self._open(target)
            except Exception as error:
                self.messages.append(f"Warning: {error}")
                return
            self.images.insert(self.index + 1, target)
            self.index += 1
        elif name in ("step", "st") or name.startswith("step"):
            self.step = int(numbers[0]) if numbers else 1
        elif name == "solid":
            self.solid = True
        elif name == "w":
            wanted = args[0] if args else ""
            if wanted not in ("logical", "physical", "world"):
                self.messages.append(f"Warning: unknown wcs '{wanted}'")
                return
            self.wcs = wanted
            self._show_line() if self.lineplot else self._show_col()
        elif name == "f":
            self.format = " ".join(args)
        elif name in ("x", "y"):
            pair = (numbers[0], numbers[1]) if len(numbers) >= 2 else None
            if len(numbers) == 1:
                self.status = ["\a"]
                return
            if name == "x":
                self.fixed_x = pair
            else:
                self.fixed_y = pair
            self._draw()
        elif name in ("nxticks", "nyticks"):
            pass                                  # the window draws its own ticks
        else:
            self.status = [f"no command ':{name}' -- :l and :c plot lines and columns, "
                           ":a averages, :i opens an image; ? lists the keys"]


def _aavg(values: np.ndarray) -> tuple[float, float]:
    """aavgr (awvgr without cuts): mean and sigma, n - 1 in the variance."""
    n = values.size
    total = float(np.sum(values.astype(np.float64)))
    if n <= 1:
        return float(f32(total)), float("nan")
    sumsq = float(np.sum((values * values).astype(np.float64)))
    variance = (sumsq - (total / n) * total) / (n - 1)
    return float(f32(total / n)), float(f32(math.sqrt(variance) if variance > 0 else 0.0))


def _amed(values: np.ndarray) -> float:
    """amedr: the (n + 1) / 2 th smallest, three or fewer by IRAF's rule."""
    a = values
    n = a.size
    if n in (1, 2):
        return float(a[0])
    if n == 3:
        if a[0] < a[1]:
            return float(a[1] if a[1] < a[2] else (a[2] if a[0] < a[2] else a[0]))
        return float(a[1] if a[1] > a[2] else (a[0] if a[0] < a[2] else a[2]))
    return float(np.sort(a)[(n + 1) // 2 - 1])


def profile(x: np.ndarray, y: np.ndarray, x1: float, y1: float, x2: float,
            y2: float) -> list[str]:
    """impprofile.x: within x1..x2, over the line from (x1, y1) to (x2,
    y2), the peak, its half-intensity width and the moments, all from
    spline3 interpolants of the coordinates and the data. Returns IRAF's
    four status lines."""
    from .asi import Interpolant

    n = x.size
    xasi, yasi = Interpolant("spline3", x), Interpolant("spline3", y)
    step0, dxacc, niterate = 0.01, 0.001, 10

    def solve(target, start):
        p = start
        for _ in range(niterate):
            value, slope = xasi(p), xasi.derivative(p)
            a = target - value
            p = max(1.0, min(float(n), p + a / slope)) if slope else p
            if abs(a) < dxacc:
                break
        return p

    p1 = solve(x1, 1.0 + n / 2.0)
    p2 = solve(x2, p1)
    if p1 == p2:
        y0, dy, step = (y1 + y2) / 2.0, 1.0, step0
    elif p1 < p2:
        y0, dy = y1, (y2 - y1) / (p2 - p1)
        step = (p2 - p1) / (round(p2) - round(p1) + 1) * step0
    else:
        p1, p2 = p2, p1
        y0, dy = y2, (y1 - y2) / (p2 - p1)
        step = (p2 - p1) / (round(p2) - round(p1) + 1) * step0

    ps = np.arange(p1, p2 + step * 1e-9, step)
    a = np.array([xasi(p) for p in ps])
    base = y0 + (ps - p1) * dy
    c = np.array([yasi(p) for p in ps]) - base
    sumb = (base.sum() - base[0] / 2 - base[-1] / 2) * step
    sum0 = (c.sum() - c[0] / 2 - c[-1] / 2) * step
    sum1 = (np.sum(a * c) - a[0] * c[0] / 2 - a[-1] * c[-1] / 2) * step
    if sum0 == 0:
        centroid = width = asym = float("nan")
    else:
        centroid = sum1 / sum0
        d = a - centroid
        sum2 = (np.sum(d * d * c) - d[0] ** 2 * c[0] / 2 - d[-1] ** 2 * c[-1] / 2) * step
        sum3 = (np.sum(d ** 3 * c) - d[0] ** 3 * c[0] / 2 - d[-1] ** 3 * c[-1] / 2) * step
        sigma = math.sqrt(sum2 / sum0) if sum2 / sum0 > 0 else 0.0
        width = 2.35482 * sigma
        asym = (sum3 / sum0) / sigma ** 3 if sigma > 0 else float("nan")
    excess = np.abs(c)
    k = int(np.argmax(excess)) if excess.size else 0
    pc, half = ps[k], excess[k] / 2
    xc = xasi(pc)
    xl = xr = float("nan")
    previous = excess[k]
    for j in range(k, -1, -1):
        if excess[j] < half:
            pl = ps[j] + (half - excess[j]) / (previous - excess[j]) * step
            xl = xasi(pl)
            break
        previous = excess[j]
    previous = excess[k]
    for j in range(k, len(ps)):
        if ps[j] >= p2:
            break
        if excess[j] < half:
            pr = ps[j] - (half - excess[j]) / (previous - excess[j]) * step
            xr = xasi(pr)
            break
        previous = excess[j]
    if not math.isnan(xl) and xl > xc:
        xl, xr = xr, xl
    fwhm = xr - xl if not (math.isnan(xl) or math.isnan(xr)) else float("nan")
    background = y0 + (pc - p1) * dy
    peak = yasi(pc)
    del sumb

    def g(v, w=8):
        from .iraffmt import printf

        return f"{printf(float(v), 'g', width=w):>{w}}" if not math.isnan(v) else \
            f"{'INDEF':>{w}}"

    return [f"[1/4] Profile: Center={g(xc)}, Width={g(fwhm)}, Peak={g(peak)}, "
            f"Bkg={g(background)}",
            f"[2/4] Moments: Centroid={g(centroid)}, Width={g(width)}, Flux={g(sum0)}, "
            f"Asym={g(asym, 6)}",
            f"[3/4] Half Intensity: Lower={g(xl)}, Upper={g(xr)}, Width={g(fwhm)}",
            f"[4/4] Background: ({g(x1)}, {g(y1)}) - ({g(x2)}, {g(y2)})"]
