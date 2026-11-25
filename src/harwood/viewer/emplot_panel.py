"""emsao's graph -- RVSAO's emplot -- and its keys, in the window.

Nothing is decided here: every key goes to `core.emplot.Plot`, which is held to
RVSAO's by files of cursor reads. The spectrum with each emission line marked where
the velocity puts it (solid where a line was found and fit, dashed where none was or
the cursor dropped it) and the absorption lines below; unzoomed, the summary page
beside it, as RVSAO prints it on the graph's right third. What emplot asks at the
terminal (a line's name, a wavelength limit, a velocity) is asked in a dialog; the
keys a question waits for (o, i, s, v) too, the first letter typed being the key.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import emplot
from . import gtplot, keyguide, theme
from ._qt import on_main
from .theme import palette as _c


def _keys() -> list[tuple[str, str]]:
    """emcursor's list, as the strip's pairs."""
    out = []
    for line in emplot.KEYS.splitlines()[1:]:
        key, what = line[:1], line[3:]
        out.append((key, what))
    return out


class EmplotPanel(QtWidgets.QWidget):
    """The graph, the summary, what emplot said, and the keys."""

    finished = QtCore.Signal()

    def __init__(self, plot: emplot.Plot, parent=None) -> None:
        super().__init__(parent)
        self.plot = plot
        self.em = plot.em
        self._cursor = (0.0, 0.0)
        self._seen = len(self.em.out)
        self._asked = len(self.em.out)
        plot.read_line = self._ask_line
        plot.read_key = self._ask_key

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        row = QtWidgets.QHBoxLayout()
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
        row.addWidget(self.graph, stretch=2)
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
        self.status.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        layout.addWidget(keyguide.strip(_keys()))
        self.setFocusProxy(self.graph)
        self.redraw()
        self._say()

    # ------------------------------------------------------------ drawing

    def redraw(self) -> None:
        if not self.plot.frames:
            return
        frame = self.plot.frames[-1]
        graph = self.graph
        graph.clear()
        graph.setTitle(frame.title, color=_c().ink, size="10pt")
        graph.plot(frame.x.astype(float), frame.y.astype(float),
                   pen=pg.mkPen(_c().sky, width=1.2))
        if frame.marks:
            xs = np.array([[a, c] for a, _, c, _ in frame.marks], float).ravel()
            ys = np.array([[b, d] for _, b, _, d in frame.marks], float).ravel()
            graph.plot(xs, ys, connect="pairs", pen=pg.mkPen(_c().accent, width=1.2))
        for label in frame.labels:
            item = pg.TextItem(label.text, color=_c().ink,
                               anchor=(0.0, 0.5) if label.up else (0.5, 1.0),
                               angle=90 if label.up else 0)
            item.setPos(label.x, label.y)
            graph.addItem(item)
        graph.setLabel("bottom", frame.xlabel, color=_c().muted)
        gtplot.set_window(graph, frame.window, (False, False))
        if frame.summary:
            lines = [t.text.rstrip("\n") for t in frame.summary]
            self.summary.setText("\n".join(lines))
            self.summary.setVisible(True)
        else:
            self.summary.setVisible(False)

    def _say(self) -> None:
        """What RVSAO would print: the last lines of it."""
        text = self.em.out[self._seen:]
        lines = [line for line in text.splitlines() if line.strip()]
        self.status.setText("\n".join(lines[-8:]))

    # ------------------------------------------------------------ questions

    def _prompt(self) -> str:
        """What emplot printed since the last answer: its question."""
        text = self.em.out[self._asked:]
        self._asked = len(self.em.out)
        return text.splitlines()[-1] if text.splitlines() else text

    def _ask_line(self) -> str | None:
        from .ask import line

        answer = line(self._prompt(), title="emsao")
        self._asked = len(self.em.out)
        return answer

    def _ask_key(self):
        from .ask import line

        answer = line(self._prompt(), title="emsao")
        self._asked = len(self.em.out)
        if not answer:
            return None
        return (*self._cursor, 1, answer.strip()[:1] or " ")

    # ------------------------------------------------------------ the keys

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
        """Handle one keystroke. True if it was ours."""
        if not key or not key.isprintable():
            return False
        before = len(self.plot.frames)
        self._seen = self._asked = len(self.em.out)
        if self.plot.press(*self._cursor, key):
            self.plot.finish()
            self.finished.emit()
            return True
        if len(self.plot.frames) != before:
            self.redraw()
        self._say()
        return True


@on_main
def run(plot: emplot.Plot, title: str = "emsao") -> None:
    """Open the panel on emplot's graph and wait until a key leaves it."""
    from .dock import application, show

    application()
    if not plot.start():
        plot.finish()
        return
    show(EmplotPanel(plot), title, size=(1100, 720))
    plot.finish()          # the window closed: the end of the cursor, as a file's

