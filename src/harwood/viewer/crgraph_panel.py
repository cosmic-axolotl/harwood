"""The graph of cosmicrays: flux against flux ratio, and IRAF's keys.

Nothing is computed here: every decision is `core.crfind.Session`'s, held to IRAF's by files of
cursor commands. The keys are IRAF's, all but w, for which the mouse is here; s, which draws
four surfaces of the pixels around a candidate there, draws them as an image here.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import crfind
from . import keyguide, theme
from ._qt import on_main
from .theme import palette as _c


class CrGraphPanel(QtWidgets.QWidget):
    """The candidates, and the keys that change what is replaced."""

    finished = QtCore.Signal()

    def __init__(self, session: crfind.Session, data=None, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.data = data
        self._cursor = (100.0, 1.0)
        self._section: pg.ImageItem | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(_c().ground)
        theme.follow(self, [self.plot], lambda: self.redraw())
        for name in ("left", "bottom"):
            axis = self.plot.getAxis(name)
            axis.setPen(pg.mkPen(_c().line))
            axis.setTextPen(pg.mkPen(_c().muted))
            axis.enableAutoSIPrefix(False)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        layout.addWidget(self.plot, stretch=1)
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addWidget(keyguide.strip(crfind.HELP))
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plot.installEventFilter(self)
        self.setFocusProxy(self.plot)
        self.redraw(rescale=True)

    def redraw(self, rescale: bool = False) -> None:
        session = self.session
        found = session.found
        self.plot.clear()
        self.plot.setLogMode(x=True, y=False)
        self.plot.setLabel("bottom", "Flux", color=_c().muted)
        self.plot.setLabel("left", "Flux ratio", color=_c().muted)
        self.plot.setTitle("Parameters of cosmic rays candidates", color=_c().ink, size="10pt")
        shown = session.shown()
        on_axis = [i for i in shown if found.flux[i] > 0]
        # Marks that differ in shape, so that colour is never the only channel.
        for flags, colour, symbol in (((crfind.NO, crfind.ALWAYSNO), _c().sky, "+"),
                                      ((crfind.YES, crfind.ALWAYSYES), _c().drop, "x")):
            chosen = [i for i in on_axis if found.flag[i] in flags]
            self._mark(chosen, colour, symbol)
        self._mark([i for i in on_axis if found.wt[i] != 0.0], _c().accent, "s")
        self.plot.addItem(pg.InfiniteLine(session.fluxratio, angle=0, pen=pg.mkPen(
            _c().accent, width=1.4, style=QtCore.Qt.PenStyle.DashLine)))
        if rescale:
            x0, x1, y0, y1 = session.window
            self.plot.setRange(xRange=(math.log10(x0), math.log10(x1)), yRange=(y0, y1),
                               padding=0)
        off = len(shown) - len(on_axis)
        self.status.setText(
            f"flux ratio {session.fluxratio:g}: x is replaced, + is kept, a square is trained."
            + (f" {off} with no flux above zero are off this axis." if off else ""))

    def _mark(self, chosen, colour, symbol) -> None:
        if chosen:
            found = self.session.found
            self.plot.plot([found.flux[i] for i in chosen], [found.ratio[i] for i in chosen],
                           pen=None, symbol=symbol, symbolSize=9 if symbol != "s" else 13,
                           symbolBrush=None, symbolPen=pg.mkPen(colour, width=1.4))

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.Type.KeyPress and self.key(event.text()):
            return True
        return super().eventFilter(obj, event)

    def _mouse_moved(self, position) -> None:
        box = self.plot.getPlotItem().vb
        if self._section is None and box.sceneBoundingRect().contains(position):
            point = box.mapSceneToView(position)
            self._cursor = (10.0 ** point.x(), point.y())

    def key(self, key: str) -> bool:
        """Handle one keystroke. Returns True if it was ours."""
        session = self.session
        if not key or key == "w":
            return False
        if self._section is not None:
            session.key(key, *self._cursor)
            self._section = None
            self.redraw(rescale=True)
            return True
        if key == "s":
            self._show_section()
        session.key(key, *self._cursor)
        if session.done:
            self.finished.emit()
        elif self._section is None:
            self.redraw(rescale=key in ("a", "r"))
            if key == " " and session.printed:
                self.status.setText(f"column and line {session.printed[-1]}")
        return True

    def _show_section(self) -> None:
        """cr_examine's s: the 11 by 11 pixels around the candidate, until the next key."""
        nearest = self.session.nearest(*self._cursor)
        if nearest is None or self.data is None:
            return
        found = self.session.found
        ny, nx = self.data.shape
        column, line = found.col[nearest], found.line[nearest]
        c1, c2 = max(1, column - 5), min(nx, column + 5)
        l1, l2 = max(1, line - 5), min(ny, line + 5)
        self.plot.clear()
        self.plot.setLogMode(x=False, y=False)
        self.plot.setTitle(f"Candidate at column {column}, line {line}: any key goes back",
                           size="10pt", color=_c().ink)
        self.plot.setLabel("bottom", "Column", color=_c().muted)
        self.plot.setLabel("left", "Line", color=_c().muted)
        self._section = pg.ImageItem(np.asarray(self.data[l1 - 1:l2, c1 - 1:c2], float).T)
        self._section.setRect(QtCore.QRectF(c1 - 0.5, l1 - 0.5, c2 - c1 + 1, l2 - l1 + 1))
        self.plot.addItem(self._section)
        self.plot.setRange(xRange=(c1 - 0.5, c2 + 0.5), yRange=(l1 - 0.5, l2 + 0.5), padding=0)


@on_main
def run(session: crfind.Session, data=None, title: str = "cosmicrays") -> None:
    """Open the panel and wait for q."""
    from .dock import show

    session.done = False
    show(CrGraphPanel(session, data), title, size=(980, 620))
