"""implot: reading numbers off an image by plotting cuts through it.

This is how the exercises decide things. You do not guess the overscan
section; you plot a line across the frame, expand the right-hand end, and read
where the illuminated pixels stop and the overscan begins. `implot` is the
instrument for that, and the keys matter because they are the ones the
tutorials tell you to press.

The keys and colon commands are IRAF's, answered by core.implot (the same
engine a cursor file drives); this panel draws what it says and hands it
the cursor. Positions are 1-based, as everywhere a user can see.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import implot
from . import keyguide, theme
from ._qt import on_main
from .theme import palette as _c

HELP = [
    ("l", "plot the line under the cursor"), ("c", "plot the column under the cursor"),
    ("a", "average: press at both ends of a range"), ("e", "expand: two corners"),
    ("j k", "step down, up"), ("m n", "previous, next image"),
    ("o", "overplot the next"), ("p", "profile: two points, region and background"),
    ("s", "statistics: two points"), ("space", "the pixel and its value"),
    ("r", "redraw"), ("q", "leave"),
]


class ImplotPanel(QtWidgets.QWidget):
    """One plot, and the keys that decide what it shows."""

    finished = QtCore.Signal()

    def __init__(self, data, name: str = "image", parent=None, images=None, line=None,
                 wcs: str = "logical", step: int = 0) -> None:
        super().__init__(parent)
        if images is None:
            frozen = np.asarray(data, dtype=float)
            self.engine = implot.Implot([name], line=line, wcs=wcs, step=step,
                                        reader=lambda _: (frozen, None))
        else:
            self.engine = implot.Implot(images, line=line, wcs=wcs, step=step)
        self.name = name
        self._cursor = (1.0, 1.0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(_c().ground)
        theme.follow(self, [self.plot])
        for axis in ("left", "bottom"):
            item = self.plot.getAxis(axis)
            item.setPen(pg.mkPen(_c().line))
            item.setTextPen(pg.mkPen(_c().muted))
            item.enableAutoSIPrefix(False)
        self.plot.setLabel("left", "counts", color=_c().muted)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plot.installEventFilter(self)
        layout.addWidget(self.plot, stretch=1)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        layout.addWidget(self.status)

        self.keys = keyguide.strip(HELP)
        layout.addWidget(self.keys)

        self.command = QtWidgets.QLineEdit()
        self.command.setObjectName("command")
        self.command.setPlaceholderText(":l 250 260   ·   :c 335 350   ·   :a 5   ·   "
                                        "? for the keys")
        self.command.returnPressed.connect(self._run_command)
        layout.addWidget(self.command)

        self.setFocusProxy(self.plot)
        self._render()

    # --------------------------------------------------------------- plots

    @property
    def last(self):
        x, y, label = self.engine.plot.curves[-1]
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float), label

    def plot_lines(self, first: int, last: int) -> None:
        """Average of image lines `first` to `last`, both 1-based and inclusive."""
        self.engine.colon(f":l {int(first)} {int(last)}")
        self._render()

    def plot_columns(self, first: int, last: int) -> None:
        self.engine.colon(f":c {int(first)} {int(last)}")
        self._render()

    def _render(self) -> None:
        state = self.engine.plot
        self.plot.clear()
        for k, (x, y, _label) in enumerate(state.curves):
            colour = _c().series[k % len(_c().series)]
            values = np.asarray(y, dtype=float)
            if state.logy:
                values = np.where(values > 0, values, np.nan)
            self.plot.plot(np.asarray(x, dtype=float), values,
                           pen=pg.mkPen(colour, width=1.2), connect="finite")
        self.plot.setLogMode(y=state.logy)
        self.plot.setLabel("bottom", state.xlabel, color=_c().muted)
        self.plot.setTitle(f"implot · {state.title}", color=_c().ink, size="10pt")
        x1, x2, y1, y2 = self.engine.window()
        self.plot.setRange(xRange=sorted((x1, x2)), yRange=sorted((y1, y2)), padding=0.02)
        y = self.engine.y
        lines = self.engine.status or [
            f"{state.title}   min {y.min():.1f}   max {y.max():.1f}   "
            f"mean {y.mean():.1f}   median {np.median(y):.1f}"]
        self.status.setText("\n".join(line for line in lines if line != "\a")
                            or "that key does nothing here; ? lists them")

    # -------------------------------------------------------------- input

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt's name)
        if event.type() == QtCore.QEvent.Type.KeyPress and self.key(event.text()):
            return True
        return super().eventFilter(obj, event)

    def _mouse_moved(self, position) -> None:
        box = self.plot.getPlotItem().vb
        if box.sceneBoundingRect().contains(position):
            point = box.mapSceneToView(position)
            self._cursor = (point.x(), point.y())

    def key(self, key: str) -> bool:
        if not key:
            return False
        if key == ":":
            self.command.setFocus()
            self.command.setText(":")
            return True
        if key == "?":
            self.keys.show()                      # every key is listed already
            return True
        self.engine.status = []
        self.engine.key(key, *self._cursor)
        if self.engine.done:
            self.finished.emit()
            return True
        self._render()
        return True

    def _run_command(self) -> None:
        text = self.command.text().strip()
        self.command.clear()
        self.plot.setFocus()
        if not text:
            return
        self.engine.status = []
        self.engine.colon(text)
        self._render()


@on_main
def run(data, name: str = "image", images=None, line=None, wcs="logical",
        step: int = 0) -> list[str]:
    """Open implot and wait for `q`; the lines it printed."""
    from .dock import show

    panel = show(ImplotPanel(data, name, images=images, line=line, wcs=wcs, step=step),
                 f"implot · {name}", size=(1080, 640))
    return panel.engine.messages
