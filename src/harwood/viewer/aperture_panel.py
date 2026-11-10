"""apedit: setting the apertures by hand, on the profile they come from.

`apfind` puts an aperture where the profile peaks and sizes it where the profile falls
away, and most of the time that is right. The times it is not are the ones that matter:
two stars on the slit that the finder merged, a galaxy whose light reaches further than a
threshold says, a sky band that lands on a neighbour. None of those is a parameter you
can set in advance: you have to look at the profile.

So the profile is what this shows, with every aperture drawn on it, and the keys are
IRAF's, listed along the foot. b opens the background's icfit on the profile until its q.

Nothing is computed here. Every number and every line of text comes from
`core.apeditsession`, held to IRAF's by files of cursor commands and to what IRAF draws by
its graphics metacode. Positions across the slit are 1-based, as everywhere a user can see.
The window is the one gtools gives (`viewer.gtplot`): w and the /commands set it, and the
mouse zooms it besides; `:.snap file.png` saves the graph.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import apeditsession
from . import gtplot, keyguide, theme
from ._qt import on_main
from .icfit_panel import IcfitPanel
from .theme import palette as _c

HELP = apeditsession.HELP


class AperturePanel(QtWidgets.QWidget):
    """The spatial profile, the apertures on it, or while b lasts the background's fit."""

    finished = QtCore.Signal()

    def __init__(self, session: apeditsession.ApeditSession, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.said = ""
        self._cursor = (0.0, 0.0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, stretch=1)

        self.page = QtWidgets.QWidget()
        page = QtWidgets.QVBoxLayout(self.page)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(0)
        self.stack.addWidget(self.page)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(_c().ground)
        theme.follow(self, [self.plot], lambda: self.redraw())
        for name in ("left", "bottom"):
            self.plot.getAxis(name).setPen(pg.mkPen(_c().line))
            self.plot.getAxis(name).setTextPen(pg.mkPen(_c().muted))
            self.plot.getAxis(name).enableAutoSIPrefix(False)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plot.installEventFilter(self)
        page.addWidget(self.plot, stretch=1)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        page.addWidget(self.status)

        self.keys = keyguide.strip(HELP)
        page.addWidget(self.keys)

        self.command = QtWidgets.QLineEdit()
        self.command.setObjectName("command")
        self.command.returnPressed.connect(self._run_command)
        page.addWidget(self.command)

        self.setFocusProxy(self.plot)
        self.fit_panel: IcfitPanel | None = None
        self.redraw(rescale=True)
        self._after()

    # -------------------------------------------------------------- the plot

    def redraw(self, rescale: bool = False) -> None:
        session = self.session
        graph = session.graph()
        self.plot.clear()
        title = graph.title + (f"<br>{graph.params}" if graph.params else "")
        self.plot.setTitle(title, color=_c().ink, size="10pt")
        line = self._line
        line(np.arange(1.0, graph.y.size + 1), graph.y, pg.mkPen(_c().sky, width=1.2))
        # ap_gmark and ap_gmarkb, in the window's own units, as ggwind gives them.
        x0, x1, y0, y1 = graph.window
        dy = 0.025 * (y1 - y0)
        level = y1 - 4 * dy
        current = session.current
        for index, mark in enumerate(graph.marks, 1):
            width = 2.4 if index == current else 1.2
            pen = pg.mkPen(_c().accent, width=width)
            line([mark.centre] * 2, [level - 2 * dy, level + 2 * dy], pen)
            for edge in (mark.low, mark.high):
                line([edge] * 2, [level - dy, level + dy], pen)
            line([mark.low, mark.high], [level] * 2, pen)
            text = pg.TextItem(str(mark.number), color=_c().ink, anchor=(0.5, 1))
            text.setPos(*(v[0] for v in self._on_axes([mark.centre], [level + 2.5 * dy])))
            self.plot.addItem(text)
        low = y0 + 0.02 * (y1 - y0)
        tick = 0.005 * (y1 - y0)
        for xl, xu in graph.sample:
            pen = pg.mkPen(_c().muted, width=1.2)
            line([xl, xu], [low] * 2, pen)
            for edge in (xl, xu):
                line([edge] * 2, [low - tick, low + tick], pen)
        if rescale:
            gtplot.set_window(self.plot, graph.window, graph.logs)

    def _on_axes(self, x, y):
        return gtplot.on_axes(self.session.gt.logs(), np.asarray(x), np.asarray(y))

    def _line(self, x, y, pen) -> None:
        self.plot.plot(*self._on_axes(x, y), pen=pen)

    # ------------------------------------------------------------- the keys

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt's name)
        if event.type() == QtCore.QEvent.Type.KeyPress and self.key(event.text()):
            return True
        return super().eventFilter(obj, event)

    def _mouse_moved(self, position) -> None:
        box = self.plot.getPlotItem().vb
        if box.sceneBoundingRect().contains(position):
            point = box.mapSceneToView(position)
            self._cursor = gtplot.world(self.session.gt.logs(), point.x(), point.y())

    def key(self, key: str) -> bool:
        """Handle one keystroke. Returns True if it was ours."""
        if not key:
            return False
        if key == ":":
            self.command.setFocus()
            self.command.setText(":")
            return True
        if not (key in dict(HELP) or key in "et"
                or (self.session.pending and key.isprintable())):
            return False
        self.session.key(key, *self._cursor)
        self._after()
        return True

    def _run_command(self) -> None:
        text = self.command.text().strip()
        self.command.clear()
        rest = text.lstrip(":")
        if self.session.state != "keys":
            self.session.answer(text)
        elif rest.startswith("."):
            said = gtplot.cursor_mode(self.plot, rest[1:], "apedit.png")
            self.status.setText(said.strip())
            self.plot.setFocus()
            return
        elif rest:
            self.session.key(":", rest=rest)
        self.plot.setFocus()
        self._after()

    def _after(self) -> None:
        """What the session said; the graph, or icfit; or the end."""
        session = self.session
        if session.done:
            self.finished.emit()
            return
        if session.sub is not None:
            self._open_fit()
            return
        said = session.take()
        self.said += said
        self.redraw(rescale=session.newgraph)
        lines = [line for line in said.replace("\a", "\n").splitlines() if line.strip()]
        if lines:
            self.status.setText("\n".join(lines[-3:]))
        asking = session.state != "keys"
        self.command.setPlaceholderText(
            session.prompt if asking else
            ":show   ·   press a key over the plot, or type a colon command here")
        (self.command if asking else self.plot).setFocus()

    def _open_fit(self) -> None:
        self.fit_panel = IcfitPanel.on(self.session.sub)
        self.fit_panel.finished.connect(self._fit_finished)
        self.stack.addWidget(self.fit_panel)
        self.stack.setCurrentWidget(self.fit_panel)
        self.fit_panel.plot.setFocus()

    def _fit_finished(self) -> None:
        panel, self.fit_panel = self.fit_panel, None
        self.stack.setCurrentWidget(self.page)
        self.stack.removeWidget(panel)
        panel.deleteLater()
        self.session.fit_finished()
        self.plot.setFocus()
        self._after()


@on_main
def run(session: apeditsession.ApeditSession, title: str = "apedit") -> str:
    """Open the panel, wait for q, and give back all that was said."""
    from .dock import application, show

    application()   # before the panel: a widget made with no QApplication aborts Qt

    panel = show(AperturePanel(session), title, size=(1000, 560))
    return panel.said
