"""RVSAO's plotspec in the window: a spectrum to edit by eye (xcsao's e) or a
correlation whose peak is chosen (xcsao's p). Every key goes to
`core.plotspec.Plotspec`, held to RVSAO's by files of cursor reads; what it asks is
asked in a dialog.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import plotspec
from . import gtplot, keyguide, theme
from ._qt import on_main
from .theme import palette as _c


def _keys() -> list[tuple[str, str]]:
    return [(line[:1], line[3:]) for line in plotspec.KEYS.splitlines()]


class PlotspecPanel(QtWidgets.QWidget):
    """The vector, what plotspec said, and the keys."""

    finished = QtCore.Signal()

    def __init__(self, ps: plotspec.Plotspec, out, parent=None) -> None:
        super().__init__(parent)
        self.ps = ps
        self._out = out                  # () -> what the task has said so far
        self._cursor = (0.0, 0.0)
        self._seen = self._asked = len(out())
        ps.read_line = self._ask_line
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.graph = pg.PlotWidget()
        theme.follow(self, [self.graph], lambda: self.redraw())
        self.graph.setBackground(_c().ground)
        for name in ("left", "bottom"):
            axis = self.graph.getAxis(name)
            axis.setPen(pg.mkPen(_c().line))
            axis.setTextPen(pg.mkPen(_c().muted))
            axis.enableAutoSIPrefix(False)
        self.graph.scene().sigMouseMoved.connect(self._mouse_moved)
        self.graph.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.graph.installEventFilter(self)
        layout.addWidget(self.graph, stretch=1)
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addWidget(keyguide.strip(_keys()))
        self.setFocusProxy(self.graph)
        self.redraw()

    def redraw(self) -> None:
        if not self.ps.frames:
            return
        frame = self.ps.frames[-1]
        self.graph.clear()
        self.graph.setTitle("<br>".join(frame.title), color=_c().ink, size="10pt")
        self.graph.plot(frame.x.astype(float), frame.y.astype(float),
                        pen=pg.mkPen(_c().sky, width=1.2))
        self.graph.setLabel("bottom", frame.xlabel, color=_c().muted)
        gtplot.set_window(self.graph, frame.window, (False, False))

    def _say(self) -> None:
        lines = [line for line in self._out()[self._seen:].splitlines() if line.strip()]
        self.status.setText("\n".join(lines[-8:]))

    def _ask_line(self) -> str | None:
        from .ask import line

        text = self._out()[self._asked:]
        self._asked = len(self._out())
        return line(text.splitlines()[-1] if text.splitlines() else text, title="plotspec")

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.Type.KeyPress and self.key(event.text()):
            return True
        return super().eventFilter(obj, event)

    def _mouse_moved(self, position) -> None:
        box = self.graph.getPlotItem().vb
        if box.sceneBoundingRect().contains(position):
            point = box.mapSceneToView(position)
            self._cursor = (point.x(), point.y())
            self.graph.setFocus()

    def key(self, key: str) -> bool:
        if not key or not key.isprintable():
            return False
        before = len(self.ps.frames)
        self._seen = self._asked = len(self._out())
        if self.ps.press(*self._cursor, key):
            self.finished.emit()
            return True
        if len(self.ps.frames) != before:
            self.redraw()
        self._say()
        return True


@on_main
def run(ps: plotspec.Plotspec, out, title: str = "plotspec") -> None:
    """Open the panel on the vector and wait until q."""
    from .dock import application, show

    application()
    ps.draw()
    if not ps.wait:
        return
    show(PlotspecPanel(ps, out), title, size=(1000, 640))
