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
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import apeditsession
from . import keyguide, theme
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
        self.plot.plot(range(1, graph.y.size + 1), graph.y, pen=pg.mkPen(_c().sky, width=1.2))
        x0, x1, y0, y1 = graph.window
        dy = 0.025 * (y1 - y0)
        level = y1 - 4 * dy
        current = session.current
        for index, mark in enumerate(graph.marks, 1):
            width = 2.4 if index == current else 1.2
            pen = pg.mkPen(_c().accent, width=width)
            self.plot.plot([mark.centre] * 2, [level - 2 * dy, level + 2 * dy], pen=pen)
            for edge in (mark.low, mark.high):
                self.plot.plot([edge] * 2, [level - dy, level + dy], pen=pen)
            self.plot.plot([mark.low, mark.high], [level] * 2, pen=pen)
            text = pg.TextItem(str(mark.number), color=_c().ink, anchor=(0.5, 1))
            text.setPos(mark.centre, level + 2.5 * dy)
            self.plot.addItem(text)
        low = y0 + 0.02 * (y1 - y0)
        tick = 0.005 * (y1 - y0)
        for xl, xu in graph.sample:
            pen = pg.mkPen(_c().muted, width=1.2)
            self.plot.plot([xl, xu], [low] * 2, pen=pen)
            for edge in (xl, xu):
                self.plot.plot([edge] * 2, [low - tick, low + tick], pen=pen)
        if rescale:
            self.plot.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)

    # ------------------------------------------------------------- the keys

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
        """Handle one keystroke. Returns True if it was ours."""
        if not key:
            return False
        if key == ":":
            self.command.setFocus()
            self.command.setText(":")
            return True
        if key not in dict(HELP) and key not in "wet":
            return False
        self.session.key(key, *self._cursor)
        self._after()
        return True

    def _run_command(self) -> None:
        text = self.command.text().strip()
        self.command.clear()
        if self.session.state != "keys":
            self.session.answer(text)
        elif text.lstrip(":"):
            self.session.key(":", rest=text.lstrip(":"))
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
    from .dock import show

    panel = show(AperturePanel(session), title, size=(1000, 560))
    return panel.said
