"""A graph as gtools leaves it, on a pyqtgraph plot: what every panel whose
session holds a `core.gtools.GTools` shares.

The window `w` and `:/xwindow` set, an axis that runs the other way after `w f`,
a logarithmic axis on IRAF's elog scale with its decades marked, the marks
`:/mark` names; the cursor's position given back in the graph's coordinates,
and the view as zoomed, so that nearest is nearest on the screen. And cursor
mode: `:.` commands, which in IRAF the graphics kernel takes before the task
sees them, are the panel's here (`:.snap file.png` saves the graph).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui

from ..core.gtools import elog, unelog

#: Cursor mode, as far as the panels have it.
CURSOR_MODE = [(":.snap [file]", "the graph to a file: .png, .svg, .pdf or .csv"),
               (":.write [file]", "the same"),
               (":.help", "these")]


def _bar(horizontal: bool, ends: bool) -> QtGui.QPainterPath:
    """gtools' hline, vline, hebar and vebar as pyqtgraph symbols."""
    path = QtGui.QPainterPath()
    a, b = (QtCore.QPointF(-0.5, 0), QtCore.QPointF(0.5, 0)) if horizontal else (
        QtCore.QPointF(0, -0.5), QtCore.QPointF(0, 0.5))
    path.moveTo(a)
    path.lineTo(b)
    if ends:
        for end in (a, b):
            tick = QtCore.QPointF(0, 0.2) if horizontal else QtCore.QPointF(0.2, 0)
            path.moveTo(end - tick)
            path.lineTo(end + tick)
    return path


#: A gtools mark as pyqtgraph draws it; lines and histograms of one point each
#: are nothing in IRAF, and dots here.
SYMBOLS = {"plus": "+", "cross": "x", "box": "s", "diamond": "d", "circle": "o",
           "point": "o", "line": "o", "histogram": "o",
           "hline": _bar(True, False), "vline": _bar(False, False),
           "hebar": _bar(True, True), "vebar": _bar(False, True)}


def on_axes(logs, x, y):
    """Where the values go on the plot: through elog on a logarithmic axis."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return (np.array([elog(v) for v in x]) if logs[0] else x,
            np.array([elog(v) for v in y]) if logs[1] else y)


def world(logs, x: float, y: float) -> tuple[float, float]:
    """A point of the plot as the graph's coordinates: back off elog."""
    return (unelog(x) if logs[0] else x), (unelog(y) if logs[1] else y)


def steps(x, y):
    """gt_plot's histogram: a level at each point, from halfway to the one before
    to halfway to the one after, the first and last as wide as their neighbours'."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 2:
        return x, y
    middle = (x[1:] + x[:-1]) / 2
    edges = np.r_[2 * x[0] - middle[0], middle, 2 * x[-1] - middle[-1]]
    return np.repeat(edges, 2)[1:-1], np.repeat(y, 2)


def set_window(plot, window, logs) -> None:
    """The window gtools gives, the way its axes run; a logarithmic axis drawn
    on elog, with ticks at its decades that say what they are."""
    x1, x2, y1, y2 = window
    box = plot.getViewBox()
    box.invertX(x1 > x2)
    box.invertY(y1 > y2)
    ranges = []
    for name, (a, b), log in (("bottom", (x1, x2), logs[0]), ("left", (y1, y2), logs[1])):
        axis = plot.getAxis(name)
        if log:
            low, high = sorted((elog(a), elog(b)))
            steps = range(int(np.floor(low)), int(np.ceil(high)) + 1)
            axis.setTicks([[(float(k), f"{unelog(k):g}") for k in steps]])
            ranges.append((low, high))
        else:
            axis.setTicks(None)
            ranges.append(tuple(sorted((a, b))))
    box.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)


def scale(plot, logs) -> tuple[tuple, tuple[float, float]]:
    """The view as zoomed, in the graph's coordinates, and its size on the
    screen: what a session's nearest takes as `scale`."""
    box = plot.getViewBox()
    (x0, x1), (y0, y1) = box.viewRange()     # an axis run backwards measures the same
    (x0, y0), (x1, y1) = world(logs, x0, y0), world(logs, x1, y1)
    return (x0, x1, y0, y1), (box.width() or 1.0, box.height() or 1.0)


def cursor_mode(plot, text: str, name: str) -> str:
    """IRAF's cursor mode, the part of it worth having beside a mouse: what it
    says. `name` is the file :.snap writes when it is given none."""
    words = text.split()
    command = words[0] if words else ""
    if command and "help".startswith(command):
        return "".join(f"{what:<16}{why}\n" for what, why in CURSOR_MODE)
    if command and ("snap".startswith(command) or "write".startswith(command)):
        from .window import save_plot

        path = Path(words[1]) if len(words) > 1 else free(name)
        try:
            save_plot(plot, path)
            return f"The graph is in {path}.\n"
        except (OSError, ValueError) as error:
            return f"Could not write {path}: {error}. Name a .png, .svg, .pdf or .csv.\n"
    return (f":.{command} is a cursor-mode command harwood does not have: the mouse "
            "zooms and drags the graph, and :.snap file.png saves it. :.help lists "
            "what there is.\n")


def free(name: str) -> Path:
    """name, or name with a number in it, so that nothing is written over."""
    path = Path(name)
    number = 1
    while path.exists():
        path = Path(f"{Path(name).stem}-{number}{Path(name).suffix}")
        number += 1
    return path
