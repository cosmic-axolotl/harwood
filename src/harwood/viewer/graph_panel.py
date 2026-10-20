"""A core.graph.Graph on the screen: the same curves, axes and title as the
SVG a task writes without a screen, in a panel that q closes."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import theme
from .theme import palette as _c


class GraphPanel(QtWidgets.QWidget):
    finished = QtCore.Signal()

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
        self.plot.setLogMode(y=graph.logy)
        for x, y in graph.curves:
            if graph.logy:
                y = np.where(y > 0, y, np.nan)
            self.plot.plot(x, y, pen=pg.mkPen(_c().accent, width=1.4), connect="finite")
        self.plot.installEventFilter(self)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self.plot, stretch=1)
        hint = QtWidgets.QLabel("q closes")
        hint.setObjectName("readout")
        layout.addWidget(hint)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt's name)
        if event.type() == QtCore.QEvent.Type.KeyPress and event.text() == "q":
            self.finished.emit()
            return True
        return super().eventFilter(obj, event)


def run(graph, title: str) -> None:
    from .dock import show

    show(GraphPanel(graph), title, size=(900, 560))
