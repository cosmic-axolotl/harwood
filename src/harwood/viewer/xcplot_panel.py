"""xcsao's graph -- RVSAO's xcplot -- and its keys, in the window.

Nothing is decided here: every key goes to `core.xcplot.Plot`, which is held to
RVSAO's by files of cursor reads. Above, the spectrum with the range correlated
marked; below, the correlation around its peak; beside them, RVSAO's summary. What
xcplot asks at the terminal (a template, a mode, the peak's fit, the velocity
limits) is asked in a dialog.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import xcplot
from . import gtplot, keyguide, theme
from ._qt import on_main
from .theme import palette as _c


def _keys() -> list[tuple[str, str]]:
    return [(line[1:2], line[4:]) for line in xcplot.KEYS.splitlines()[1:]]


class XcplotPanel(QtWidgets.QWidget):
    """The two graphs, the summary, what xcplot said, and the keys."""

    finished = QtCore.Signal()

    def __init__(self, plot: xcplot.Plot, parent=None) -> None:
        super().__init__(parent)
        self.plot = plot
        self.xc = plot.xc
        self._cursor = (0.0, 0.0)
        self._seen = self._asked = len(self.xc.out)
        plot.read_line = self._ask_line

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        row = QtWidgets.QHBoxLayout()
        graphs = QtWidgets.QVBoxLayout()
        self.graphs = [pg.PlotWidget(), pg.PlotWidget()]
        theme.follow(self, self.graphs, lambda: self.redraw())
        for graph in self.graphs:
            graph.setBackground(_c().ground)
            for name in ("left", "bottom"):
                axis = graph.getAxis(name)
                axis.setPen(pg.mkPen(_c().line))
                axis.setTextPen(pg.mkPen(_c().muted))
                axis.enableAutoSIPrefix(False)
            graph.getAxis("left").setWidth(70)
            graph.scene().sigMouseMoved.connect(
                lambda position, graph=graph: self._mouse_moved(graph, position))
            graph.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            graph.installEventFilter(self)
            graphs.addWidget(graph)
        row.addLayout(graphs, stretch=2)
        self.summary = QtWidgets.QLabel("")
        self.summary.setObjectName("summary")
        self.summary.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.summary.setStyleSheet("font-family: monospace;")
        row.addWidget(self.summary, stretch=1)
        layout.addLayout(row, stretch=1)
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addWidget(keyguide.strip(_keys()))
        self.setFocusProxy(self.graphs[0])
        self.redraw()
        self._say()

    def redraw(self) -> None:
        if not self.plot.frames:
            return
        frame = self.plot.frames[-1]
        for graph, panel in zip(self.graphs, (frame.spectrum, frame.correlation),
                                strict=True):
            graph.clear()
            graph.setVisible(panel is not None)
            if panel is None:
                continue
            graph.setTitle(panel.title, color=_c().ink, size="10pt")
            graph.plot(panel.x.astype(float), panel.y.astype(float),
                       pen=pg.mkPen(_c().sky, width=1.2))
            if panel.marks:
                xs = np.array([[a, c] for a, _, c, _ in panel.marks], float).ravel()
                ys = np.array([[b, d] for _, b, _, d in panel.marks], float).ravel()
                graph.plot(xs, ys, connect="pairs", pen=pg.mkPen(_c().accent))
            if panel.circles:
                xs, ys = zip(*panel.circles, strict=True)
                graph.plot(list(xs), list(ys), pen=None, symbol="o", symbolSize=4,
                           symbolBrush=pg.mkBrush(_c().accent))
            graph.setLabel("bottom", panel.xlabel, color=_c().muted)
            gtplot.set_window(graph, panel.window, (False, False))
        self.summary.setText("\n".join(t.text for t in frame.texts))

    def _say(self) -> None:
        text = self.xc.out[self._seen:]
        lines = [line for line in text.splitlines() if line.strip()]
        self.status.setText("\n".join(lines[-8:]))

    def _ask_line(self) -> str | None:
        from .ask import line

        text = self.xc.out[self._asked:]
        self._asked = len(self.xc.out)
        answer = line(text.splitlines()[-1] if text.splitlines() else text, title="xcsao")
        self._asked = len(self.xc.out)
        return answer

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.Type.KeyPress and self.key(event.text()):
            return True
        return super().eventFilter(obj, event)

    def _mouse_moved(self, graph, position) -> None:
        box = graph.getPlotItem().vb
        if box.sceneBoundingRect().contains(position):
            point = box.mapSceneToView(position)
            self._cursor = (point.x(), point.y())
            graph.setFocus()

    def key(self, key: str) -> bool:
        if not key or not key.isprintable():
            return False
        before = len(self.plot.frames)
        self._seen = self._asked = len(self.xc.out)
        if self.plot.press(*self._cursor, key):
            self.finished.emit()
            return True
        if len(self.plot.frames) != before:
            self.redraw()
        self._say()
        return True


@on_main
def run(plot: xcplot.Plot, title: str = "xcsao") -> None:
    """Open the panel on xcplot's graph and wait until a key leaves it."""
    from .dock import application, show

    application()
    if not plot.start():
        return
    show(XcplotPanel(plot), title, size=(1100, 760))
