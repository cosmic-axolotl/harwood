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
page when it is embedded in one. IRAF's line types and colours and GIO's marks
(gmark's shapes and sizes, from sys/gio/markers.inc and gumark.x) are drawn as
GIO draws them: translated from IRAF, Copyright (c) 1986 Association of
Universities for Research in Astronomy Inc.; see LICENSES/IRAF-COPYRIGHT.
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


#: IRAF's colours 1-9 (gio's G_PLCOLOR): the foreground, then red, green, blue,
#: cyan, yellow, magenta, purple and dark slate grey. 1 follows the page's accent.
COLORS = ("var(--accent,#1f6fb2)", "#d62728", "#2ca02c", "#1f4fd6", "#17becf",
          "#c9a400", "#c51bb5", "#7b2fbe", "#2f4f4f")
#: G_PLTYPE 1-4: solid, dashed, dotted, dot-dashed; 0 is clear (nothing drawn).
DASHES = {1: "", 2: "6 4", 3: "1.5 3", 4: "7 3 1.5 3"}

#: The unit-square polylines of GIO's marks (sys/gio/markers.inc), which gumark
#: scales to the mark's size about its centre.
_CIRCLE_X = (1.000, 0.996, 0.984, 0.963, 0.936, 0.901, 0.859, 0.812, 0.759, 0.702,
             0.642, 0.580, 0.516, 0.452, 0.389, 0.327, 0.269, 0.214, 0.164, 0.119,
             0.081, 0.050, 0.025, 0.009, 0.001, 0.001, 0.009, 0.025, 0.050, 0.081,
             0.119, 0.164, 0.214, 0.269, 0.327, 0.389, 0.452, 0.516, 0.580, 0.642,
             0.702, 0.759, 0.812, 0.859, 0.901, 0.936, 0.963, 0.984, 0.996, 1.000)
_CIRCLE_Y = (0.500, 0.564, 0.627, 0.688, 0.745, 0.799, 0.848, 0.891, 0.928, 0.957,
             0.979, 0.994, 1.000, 0.998, 0.987, 0.969, 0.943, 0.910, 0.870, 0.824,
             0.773, 0.717, 0.658, 0.596, 0.532, 0.468, 0.404, 0.342, 0.283, 0.227,
             0.176, 0.130, 0.090, 0.057, 0.031, 0.013, 0.002, 0.000, 0.006, 0.021,
             0.043, 0.072, 0.109, 0.152, 0.201, 0.255, 0.312, 0.373, 0.436, 0.500)
SHAPES = {
    "box": ((0, 1, 1, 0, 0), (0, 0, 1, 1, 0)),
    "plus": ((.5, .5, np.nan, 0, 1), (0, 1, np.nan, .5, .5)),
    "cross": ((0, 1, np.nan, 0, 1), (0, 1, np.nan, 1, 0)),
    "diamond": ((.5, 1, .5, 0, .5), (0, .5, 1, .5, 0)),
    "hline": ((0, 1), (.5, .5)),
    "vline": ((.5, .5), (0, 1)),
    "hebar": ((0, 0, np.nan, 0, 1, np.nan, 1, 1), (0, 1, np.nan, .5, .5, np.nan, 1, 0)),
    "vebar": ((0, 1, np.nan, .5, .5, np.nan, 1, 0), (0, 0, np.nan, 0, 1, np.nan, 1, 1)),
    "circle": (_CIRCLE_X, _CIRCLE_Y),
}
#: gumark's standard mark size 1, in NDC: a quarter of a character's height.
STANDARD_SIZE = 0.0294 / 4


def mark_size(size: float, per_ndc: float, per_world: float) -> float:
    """gumark's size on the page: above 0 in NDC (1 and over the standard sizes),
    under 0 in world units. `per_ndc` and `per_world`: page units per NDC and per
    world unit along the axis."""
    if size > 0:
        ndc = STANDARD_SIZE * min(4, int(size)) if size >= 1 else size
        return ndc * per_ndc
    return -size * per_world


class Graph:
    def __init__(self, title: list[str] | str = "", xlabel: str = "", ylabel: str = "",
                 logy: bool = False, logx: bool = False) -> None:
        self.title = [title] if isinstance(title, str) else list(title)
        self.xlabel, self.ylabel, self.logy, self.logx = xlabel, ylabel, logy, logx
        self.curves: list[tuple[np.ndarray, np.ndarray]] = []
        self.styles: list[tuple[int, int]] = []     #: each curve's (line type, colour)
        self.marks: tuple[np.ndarray, np.ndarray] | None = None    #: plain +, for IRAF's marks
        #: GIO's marks: (x, y, kind, xsize, ysize, colour), sizes as gmark takes them;
        #: kind "point" is a dot.
        self.marklist: list[tuple[float, float, str, float, float, int]] = []
        self.window: tuple[float, float, float, float] | None = None
        self.xtran = self.ytran = None      #: "linear", "log" or "elog", when set
        self.hint = ""
        #: Texts at points of the data: (x, y, text, colour).
        self.texts: list[tuple[float, float, str, int]] = []
        #: One unit of x as long as one of y (an image's pixels).
        self.lock_aspect = False
        #: Draw the axes, their labels and the title (IRAF's perimeter).
        self.frame = True
        #: Closed polygons painted in this order, each filled with the page's ground
        #: over what is behind it and outlined: a surface with its hidden lines gone.
        self.polygons: list[tuple[np.ndarray, np.ndarray]] = []
        #: GIO's texts, anywhere on the page (not clipped to the axes): (x, y, text,
        #: angle of the baseline in degrees counterclockwise, anchor in x and y --
        #: 0 left or top, 0.5 centre, 1 right or bottom --, height of a character
        #: in y's units, colour). Monospaced, their spaces kept: wcslab's units line
        #: sits over the number's.
        self.labels: list[tuple[float, float, str, float, float, float, float, int]] = []

    def line(self, x, y, ltype: int = 1, color: int = 1) -> Graph:
        self.curves.append((np.asarray(x, float), np.asarray(y, float)))
        self.styles.append((ltype, color))
        return self

    def mark(self, x: float, y: float, kind: str = "box", xsize: float = 0.01,
             ysize: float | None = None, color: int = 1) -> Graph:
        self.marklist.append((float(x), float(y), kind, float(xsize),
                              float(xsize if ysize is None else ysize), color))
        return self

    def text(self, x: float, y: float, text: str, color: int = 1) -> Graph:
        self.texts.append((float(x), float(y), str(text), color))
        return self

    def label(self, x: float, y: float, text: str, angle: float = 0.0,
              anchor: tuple[float, float] = (0.5, 0.5), size: float = 0.0,
              color: int = 1) -> Graph:
        """A text on the page; size is a character's height in y's units (0: the
        page's own small size)."""
        self.labels.append((float(x), float(y), str(text), float(angle),
                            float(anchor[0]), float(anchor[1]), float(size), color))
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
        if self.ytran is not None:
            return self.ytran == "elog"
        return self.logy and self.window is not None and self.window[2] <= 0

    @property
    def xelog(self) -> bool:
        if self.xtran is not None:
            return self.xtran == "elog"
        return self.logx and self.window is not None and self.window[0] <= 0

    def on_axis(self, y) -> np.ndarray:
        """Values as the y axis has them."""
        return elog(y) if self.elog else np.asarray(y, dtype=float)

    def on_xaxis(self, x) -> np.ndarray:
        """Values as the x axis has them (elog's; a log axis is the panel's to take)."""
        return elog(x) if self.xelog else np.asarray(x, dtype=float)

    def _axis_values(self, values, log: bool, elog_: bool) -> np.ndarray:
        values = np.asarray(values, float)
        if elog_:
            return elog(values)
        if log:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(values > 0, np.log10(np.where(values > 0, values, 1)),
                                np.nan)
        return values

    def axes_window(self) -> tuple[float, float, float, float]:
        """The window as the axes have it: log10 on a log axis, elog's on elog."""
        if self.window is not None:
            x1, x2, y1, y2 = self.window
            xs = self._axis_values([x1, x2], self.logx, self.xelog)
            ys = self._axis_values([y1, y2], self.logy, self.elog)
            return float(xs[0]), float(xs[1]), float(ys[0]), float(ys[1])
        xs = [self._axis_values(x, self.logx, self.xelog) for x, _ in self.curves]
        xs += [self._axis_values([m[0]], self.logx, self.xelog) for m in self.marklist]
        allx = np.concatenate(xs) if xs else np.zeros(2)
        allx = allx[np.isfinite(allx)]
        x1, x2 = (float(allx.min()), float(allx.max())) if allx.size else (0.0, 1.0)
        if x2 == x1:
            x1, x2 = x1 - 1, x2 + 1
        y1, y2 = self._yspan()
        return x1, x2, y1, y2

    def mark_lines(self, per_ndc_x: float, per_ndc_y: float, per_world_x: float,
                   per_world_y: float):
        """Each mark as gumark draws it, in the axes' coordinates: a polyline with
        NaN where the pen lifts, and its colour. `per_ndc_*`: axis units per NDC;
        `per_world_*`: axis units per world unit at the mark (1 on a linear axis)."""
        out = []
        for x, y, kind, xsize, ysize, color in self.marklist:
            if kind == "point" or kind not in SHAPES:
                continue
            ax = float(self._axis_values([x], self.logx, self.xelog)[0])
            ay = float(self._axis_values([y], self.logy, self.elog)[0])
            wx = mark_size(xsize, per_ndc_x, per_world_x)
            wy = mark_size(ysize, per_ndc_y, per_world_y)
            ux, uy = (np.asarray(v, float) for v in SHAPES[kind])
            out.append((ax - wx / 2 + ux * wx, ay - wy / 2 + uy * wy, color))
        return out

    def svg(self) -> str:
        x1, x2, y1, y2 = self.axes_window()
        w, h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM
        left, top = LEFT, TOP
        if self.lock_aspect and x2 != x1 and y2 != y1:
            scale = min(w / abs(x2 - x1), h / abs(y2 - y1))
            left += (w - scale * abs(x2 - x1)) / 2
            top += (h - scale * abs(y2 - y1)) / 2
            w, h = scale * abs(x2 - x1), scale * abs(y2 - y1)

        def px(x):
            return left + (self._axis_values(x, self.logx, self.xelog) - x1) / (x2 - x1) * w

        def py(y):
            y = np.asarray(y, float)
            if self.elog:
                y = elog(y)
            elif self.logy:
                y = np.log10(np.where(y > 0, y, 10.0 ** y1))
            return top + h - (y - y1) / (y2 - y1) * h

        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
               f'width="{WIDTH}" height="{HEIGHT}" font-family="sans-serif" font-size="12">',
               '<style>.ink{fill:var(--ink,#222)}.line{stroke:var(--line,#666)}'
               '.curve{stroke:var(--accent,#1f6fb2)}</style>',
               f'<rect width="{WIDTH}" height="{HEIGHT}" fill="var(--ground,#fff)"/>']
        for row, text in enumerate(self.title):
            out.append(f'<text class="ink" x="{WIDTH / 2}" y="{20 + 16 * row}" '
                       f'text-anchor="middle">{escape(text)}</text>')
        if self.frame:
            out.append(f'<rect class="line" x="{left}" y="{top}" width="{w}" height="{h}" '
                       'fill="none"/>')
            xt = ([10.0 ** p for p in range(math.ceil(x1), math.floor(x2) + 1)]
                  if self.logx and not self.xelog else ticks(*sorted((x1, x2))))
            for value in xt:
                x = float(px(value)) if not self.xelog else \
                    left + (value - x1) / (x2 - x1) * w
                out.append(f'<line class="line" x1="{x:.1f}" y1="{top + h}" x2="{x:.1f}" '
                           f'y2="{top + h + 5}"/>')
                out.append(f'<text class="ink" x="{x:.1f}" y="{top + h + 19}" '
                           f'text-anchor="middle">{_label(value)}</text>')
            decades = range(math.ceil(min(y1, y2)), math.floor(max(y1, y2)) + 1)
            yt = ([10.0 ** p for p in decades]
                  if self.logy and not self.elog else ticks(*sorted((y1, y2))))
            for value in yt:
                y = (float(py(value)) if not self.elog
                     else top + h - (value - y1) / (y2 - y1) * h)
                out.append(f'<line class="line" x1="{left - 5}" y1="{y:.1f}" x2="{left}" '
                           f'y2="{y:.1f}"/>')
                out.append(f'<text class="ink" x="{left - 8}" y="{y + 4:.1f}" '
                           f'text-anchor="end">{_label(value)}</text>')
            if self.xlabel:
                out.append(f'<text class="ink" x="{left + w / 2}" y="{HEIGHT - 12}" '
                           f'text-anchor="middle">{escape(self.xlabel)}</text>')
            if self.ylabel:
                out.append(f'<text class="ink" transform="translate(16 {top + h / 2}) '
                           f'rotate(-90)" text-anchor="middle">{escape(self.ylabel)}</text>')
        if self.frame:
            out.append(f'<clipPath id="plot"><rect x="{left}" y="{top}" width="{w}" '
                       f'height="{h}"/></clipPath><g clip-path="url(#plot)">')
        else:
            out.append("<g>")
        if self.marks is not None:
            for a, b in zip(px(self.marks[0]), py(self.marks[1]), strict=True):
                out.append(f'<path class="curve" d="M{a - 3:.1f} {b:.1f}h6'
                           f'M{a:.1f} {b - 3:.1f}v6" fill="none" stroke-width="1"/>')
        for (x, y), (ltype, color) in zip(self.curves, self.styles, strict=True):
            if ltype == 0:
                continue
            dash = DASHES.get(ltype, "")
            dash = f' stroke-dasharray="{dash}"' if dash else ""
            for piece in _pieces(px(x), py(y)):
                points = " ".join(f"{a:.2f},{b:.2f}" for a, b in piece)
                out.append(f'<polyline fill="none" stroke="{_color(color)}" '
                           f'stroke-width="1.3"{dash} points="{points}"/>')
        for x, y, kind, _, _, color in self.marklist:
            if kind == "point":
                out.append(f'<circle cx="{float(px([x])[0]):.2f}" '
                           f'cy="{float(py([y])[0]):.2f}" r="1.2" fill="{_color(color)}"/>')
        world_x = w / (x2 - x1) if x2 != x1 else 1.0
        world_y = h / (y2 - y1) if y2 != y1 else 1.0
        if self.polygons:
            for x, y in self.polygons:
                points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(px(x), py(y),
                                                                    strict=True))
                out.append(f'<polygon fill="var(--ground,#fff)" stroke="{_color(1)}" '
                           f'stroke-width="0.6" points="{points}"/>')
        for x, y, text, color in self.texts:
            out.append(f'<text x="{float(px([x])[0]):.2f}" y="{float(py([y])[0]) + 4:.2f}" '
                       f'fill="{_color(color)}" font-size="10" text-anchor="middle">'
                       f'{escape(text)}</text>')
        for x, y, color in self._page_marks(px, py, world_x, world_y):
            for piece in _pieces(x, y):
                points = " ".join(f"{a:.2f},{b:.2f}" for a, b in piece)
                out.append(f'<polyline fill="none" stroke="{_color(color)}" '
                           f'stroke-width="1" points="{points}"/>')
        out.append("</g>")
        for x, y, text, angle, ax, ay, size, color in self.labels:
            if not text.strip():
                continue
            anchor = "start" if ax < 0.25 else "end" if ax > 0.75 else "middle"
            base = "hanging" if ay < 0.25 else "alphabetic" if ay > 0.75 else "central"
            a, b = float(px([x])[0]), float(py([y])[0])
            font = abs(size * h / (y2 - y1)) if size and y2 != y1 else 11.0
            out.append(f'<text x="{a:.2f}" y="{b:.2f}" fill="{_color(color)}" '
                       f'font-family="monospace" font-size="{font:.1f}" '
                       f'text-anchor="{anchor}" dominant-baseline="{base}" '
                       f'xml:space="preserve" transform="rotate({-angle:.2f} {a:.2f} '
                       f'{b:.2f})">{escape(text)}</text>')
        out.append("</svg>")
        return "\n".join(out)

    def _page_marks(self, px, py, world_x: float, world_y: float):
        """The marks in page pixels: NDC sizes on the page's width and height, world
        sizes through the axes' scale (on a log axis, at the mark)."""
        out = []
        for x, y, kind, xsize, ysize, color in self.marklist:
            if kind not in SHAPES:
                continue
            cx, cy = float(px([x])[0]), float(py([y])[0])
            sx = mark_size(xsize, WIDTH, abs(float(px([x + 0.5])[0] - px([x - 0.5])[0]))
                           if self.logx else world_x)
            sy = mark_size(ysize, HEIGHT, abs(float(py([y + 0.5])[0] - py([y - 0.5])[0]))
                           if self.logy else world_y)
            ux, uy = (np.asarray(v, float) for v in SHAPES[kind])
            out.append((cx - sx / 2 + ux * sx, cy + sy / 2 - uy * sy, color))
        return out

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.svg() + "\n", encoding="utf-8")
        return path


def _last_path() -> Path:
    import os

    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "harwood" / "stdgraph.pickle"


def remember(graph: Graph) -> None:
    """Keep the graph as stdgraph's last frame, for a task that appends to it
    (wcslab --append over contour); a graph that cannot be kept is not."""
    import pickle

    try:
        path = _last_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(graph))
    except Exception:  # noqa: BLE001 -- the plot is made; only the memory of it fails
        pass


def last() -> Graph | None:
    """stdgraph's last frame, as remember kept it."""
    import pickle

    path = _last_path()
    if not path.exists():
        return None
    try:
        graph = pickle.loads(path.read_bytes())
    except Exception:  # noqa: BLE001 -- an old or broken file is no frame
        return None
    return graph if isinstance(graph, Graph) else None


def _color(index: int) -> str:
    return COLORS[(max(1, int(index)) - 1) % len(COLORS)]


def _pieces(x, y):
    """The runs of a polyline between its breaks (NaN, INDEF in IRAF)."""
    run = []
    for a, b in zip(np.asarray(x, float), np.asarray(y, float), strict=True):
        if math.isfinite(a) and math.isfinite(b):
            run.append((a, b))
        elif run:
            yield run
            run = []
    if run:
        yield run
