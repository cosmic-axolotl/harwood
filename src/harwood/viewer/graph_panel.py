"""A core.graph.Graph on the screen: the same curves, axes and title as the
SVG a task writes without a screen, in a panel, until q."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import theme
from .theme import palette as _c


class GraphPanel(QtWidgets.QWidget):
    finished = QtCore.Signal()
    #: A graph is worth going back to; see the window's finish_panel.
    keep_after_quit = True

    def __init__(self, graph, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(_c().ground)
        theme.follow(self, [self.plot])
        for axis in ("left", "bottom"):
            item = self.plot.getAxis(axis)
            item.setPen(pg.mkPen(_c().line))
            item.setTextPen(pg.mkPen(_c().muted))
            item.enableAutoSIPrefix(False)
        self.plot.setTitle("<br>".join(graph.title), color=_c().muted, size="10pt")
        self.plot.setLabel("bottom", graph.xlabel, color=_c().muted)
        self.plot.setLabel("left", graph.ylabel, color=_c().muted)
        self.plot.setLogMode(y=graph.logy and not graph.elog)
        for x, y in graph.curves:
            if graph.logy and not graph.elog:
                y = np.where(y > 0, y, np.nan)
            self.plot.plot(x, graph.on_axis(y), pen=pg.mkPen(_c().accent, width=1.4),
                           connect="finite")
        if graph.marks is not None:
            self.plot.plot(graph.marks[0], graph.on_axis(graph.marks[1]), pen=None, symbol="+",
                           symbolSize=6, symbolPen=pg.mkPen(_c().accent))
        if graph.window is not None:
            self._set_window(graph)
        self.plot.installEventFilter(self)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setFocusProxy(self.plot)
        layout.addWidget(self.plot, stretch=1)
        hint = QtWidgets.QLabel(graph.hint or "q when you are done")
        hint.setObjectName("readout")
        layout.addWidget(hint)

    def _set_window(self, graph) -> None:
        """The window IRAF gives the plot, and on its elog scale, where the view is
        not in what the axis says, ticks that say what they are."""
        x1, x2, y1, y2 = graph.window
        if graph.elog:
            top = max(abs(y1), abs(y2))
            powers = [10.0 ** k for k in range(1, int(np.log10(top)) + 1)]
            values = sorted({0.0, *powers, *(-p for p in powers if y1 < 0)})
            self.plot.getAxis("left").setTicks(
                [[(float(graph.on_axis(v)), f"{v:g}") for v in values]])
            y1, y2 = float(graph.on_axis(y1)), float(graph.on_axis(y2))
        elif graph.logy:
            y1, y2 = float(np.log10(y1)), float(np.log10(y2))
        self.plot.setRange(xRange=(x1, x2), yRange=(y1, y2), padding=0)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt's name)
        if event.type() == QtCore.QEvent.Type.KeyPress and event.text() == "q":
            self.finished.emit()
            return True
        return super().eventFilter(obj, event)


def run(graph, title: str) -> None:
    from .dock import show

    show(GraphPanel(graph), title, size=(900, 560))


class MarkPanel(GraphPanel):
    """A graph to mark a value on: space (or a click) at the cursor marks,
    q ends -- histplot's and radplot's graphics cursor."""
    keep_after_quit = False     # a step in a task, not a result

    def __init__(self, graph, start: float, parent=None) -> None:
        super().__init__(graph, parent)
        self.events: list[tuple[float, float, str]] = []
        self.line = pg.InfiniteLine(angle=90 if "Histogram" in " ".join(graph.title) else 0,
                                    pos=start, pen=pg.mkPen(_c().warning, width=1.2))
        self.plot.addItem(self.line)
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self._cursor = (start, start)
        self.plot.scene().sigMouseMoved.connect(self._moved)

    def _moved(self, position) -> None:
        point = self.plot.getViewBox().mapSceneToView(position)
        self._cursor = (point.x(), point.y())

    def _mark(self, key: str) -> None:
        x, y = self._cursor
        self.events.append((x, y, key))
        self.line.setValue(x if self.line.angle == 90 else y)

    def _clicked(self, event) -> None:
        self._moved(event.scenePos())
        self._mark(" ")

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt's name)
        if event.type() == QtCore.QEvent.Type.KeyPress and event.text() and \
                event.text() != "q":
            self._mark(event.text())
            return True
        return super().eventFilter(obj, event)


def mark(graph, start: float):
    """Show the graph and give back the marks made on it, q last."""
    from .dock import show

    panel = show(MarkPanel(graph, start), graph.title[0], size=(900, 560))
    return [*panel.events, (float("nan"), float("nan"), "q")]
