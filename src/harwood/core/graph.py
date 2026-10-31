"""A graph drawn without Qt: an SVG of curves on labelled axes.

IRAF's tasks plot to `stdgraph`, which is a window when there is one and a
plotter file when there is not. The window is harwood's own; this is the
file -- a plain SVG, readable in any browser, small, and the same on every
machine, so a task run over ssh or in a script still leaves its plot behind.

    graph = Graph(["Histogram of obj001.fits"], xlabel="value", ylabel="count", logy=True)
    graph.line(x, y)          # a polyline through the points
    graph.steps(edges, y)     # a stepped curve: y[i] from edges[i] to edges[i+1]
    graph.write("hist.svg")

Colours come from CSS variables with fallbacks, so the plot follows a dark
page when it is embedded in one.
"""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

import numpy as np

from .gio import decades, elog

WIDTH, HEIGHT = 760, 480
LEFT, RIGHT, TOP, BOTTOM = 78, 24, 64, 56


def ticks(low: float, high: float, count: int = 6) -> list[float]:
    """Round values for an axis: steps of 1, 2 or 5 times a power of ten."""
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        return [low]
    raw = (high - low) / max(1, count)
    power = 10 ** math.floor(math.log10(raw))
    step = min((m * power for m in (1, 2, 5, 10)), key=lambda s: abs(s - raw))
    first = math.ceil(low / step) * step
    out, value = [], first
    while value <= high + step * 1e-9:
        out.append(round(value, 12))
        value += step
    return out


def _label(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1e5 or abs(value) < 1e-3:
        return f"{value:.3g}"
    return f"{value:.6g}"


class Graph:
    def __init__(self, title: list[str] | str = "", xlabel: str = "", ylabel: str = "",
                 logy: bool = False) -> None:
        self.title = [title] if isinstance(title, str) else list(title)
        self.xlabel, self.ylabel, self.logy = xlabel, ylabel, logy
        self.curves: list[tuple[np.ndarray, np.ndarray]] = []
        self.marks: tuple[np.ndarray, np.ndarray] | None = None    #: plain +, for IRAF's marks
        self.window: tuple[float, float, float, float] | None = None
        self.hint = ""

    def line(self, x, y) -> Graph:
        self.curves.append((np.asarray(x, float), np.asarray(y, float)))
        return self

    def steps(self, edges, y) -> Graph:
        """y[i] across edges[i]..edges[i+1], joined by verticals (hgline)."""
        edges, y = np.asarray(edges, float), np.asarray(y, float)
        xs = np.repeat(edges, 2)[1:-1]
        ys = np.repeat(y, 2)
        # hgline stops where the last bin starts.
        return self.line(xs[:-1], ys[:-1])

    def _yspan(self) -> tuple[float, float]:
        values = np.concatenate([y for _, y in self.curves]) if self.curves else np.zeros(1)
        if self.logy:
            positive = values[values > 0]
            if positive.size == 0:
                return 0.0, 1.0
            return math.floor(math.log10(positive.min())), math.ceil(
                math.log10(positive.max()) + 1e-12)
        low, high = float(values.min()), float(values.max())
        if high == low:
            low, high = low - 1, high + 1
        pad = (high - low) * 0.05
        return low - pad if low < 0 else min(0.0, low), high + pad

    def scale_to(self, x1: float, x2: float) -> Graph:
        """The window IRAF gives a plot: these limits, the data's extremes in y, and on
        a log axis out to whole decades."""
        values = np.concatenate([y for _, y in self.curves])
        low, high = float(values.min()), float(values.max())
        if self.logy:
            low, high = decades(low, high)
        self.window = (x1, x2, low, high)
        return self

    @property
    def elog(self) -> bool:
        """IRAF's scale when a log axis has data at or under zero."""
        return self.logy and self.window is not None and self.window[2] <= 0

    def on_axis(self, y) -> np.ndarray:
        """Values as the y axis has them."""
        return elog(y) if self.elog else np.asarray(y, dtype=float)

    def svg(self) -> str:
        xs = np.concatenate([x for x, _ in self.curves]) if self.curves else np.zeros(2)
        x1, x2 = float(np.nanmin(xs)), float(np.nanmax(xs))
        if x2 == x1:
            x1, x2 = x1 - 1, x2 + 1
        y1, y2 = self._yspan()
        if self.window is not None:
            x1, x2 = self.window[:2]
            if self.elog:
                y1, y2 = float(elog(self.window[2])), float(elog(self.window[3]))
            elif self.logy:
                y1, y2 = math.log10(self.window[2]), math.log10(self.window[3])
            else:
                y1, y2 = self.window[2:]
        w, h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

        def px(x):
            return LEFT + (np.asarray(x, float) - x1) / (x2 - x1) * w

        def py(y):
            y = np.asarray(y, float)
            if self.elog:
                y = elog(y)
            elif self.logy:
                y = np.log10(np.where(y > 0, y, 10.0 ** y1))
            return TOP + h - (y - y1) / (y2 - y1) * h

        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
               f'width="{WIDTH}" height="{HEIGHT}" font-family="sans-serif" font-size="12">',
               '<style>.ink{fill:var(--ink,#222)}.line{stroke:var(--line,#666)}'
               '.curve{stroke:var(--accent,#1f6fb2)}</style>',
               f'<rect width="{WIDTH}" height="{HEIGHT}" fill="var(--ground,#fff)"/>']
        for row, text in enumerate(self.title):
            out.append(f'<text class="ink" x="{WIDTH / 2}" y="{20 + 16 * row}" '
                       f'text-anchor="middle">{escape(text)}</text>')
        out.append(f'<rect class="line" x="{LEFT}" y="{TOP}" width="{w}" height="{h}" '
                   'fill="none"/>')
        for value in ticks(x1, x2):
            x = float(px(value))
            out.append(f'<line class="line" x1="{x:.1f}" y1="{TOP + h}" x2="{x:.1f}" '
                       f'y2="{TOP + h + 5}"/>')
            out.append(f'<text class="ink" x="{x:.1f}" y="{TOP + h + 19}" '
                       f'text-anchor="middle">{_label(value)}</text>')
        yt = ([10.0 ** p for p in range(int(y1), int(y2) + 1)]
              if self.logy and not self.elog else ticks(y1, y2))
        for value in yt:
            y = float(py(value))
            out.append(f'<line class="line" x1="{LEFT - 5}" y1="{y:.1f}" x2="{LEFT}" '
                       f'y2="{y:.1f}"/>')
            out.append(f'<text class="ink" x="{LEFT - 8}" y="{y + 4:.1f}" '
                       f'text-anchor="end">{_label(value)}</text>')
        if self.xlabel:
            out.append(f'<text class="ink" x="{LEFT + w / 2}" y="{HEIGHT - 12}" '
                       f'text-anchor="middle">{escape(self.xlabel)}</text>')
        if self.ylabel:
            out.append(f'<text class="ink" transform="translate(16 {TOP + h / 2}) '
                       f'rotate(-90)" text-anchor="middle">{escape(self.ylabel)}</text>')
        if self.marks is not None:
            for a, b in zip(px(self.marks[0]), py(self.marks[1]), strict=True):
                out.append(f'<path class="curve" d="M{a - 3:.1f} {b:.1f}h6'
                           f'M{a:.1f} {b - 3:.1f}v6" fill="none" stroke-width="1"/>')
        for x, y in self.curves:
            points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(px(x), py(y), strict=True))
            out.append(f'<polyline class="curve" fill="none" stroke-width="1.3" '
                       f'points="{points}"/>')
        out.append("</svg>")
        return "\n".join(out)

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.svg() + "\n", encoding="utf-8")
        return path
