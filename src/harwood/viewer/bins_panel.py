"""illumination's bins, set on the graph -- IRAF's il_gsetbins -- and its keys.

Nothing is decided here: every key goes to `core.longslit.BinsSession`, which is held
to IRAF's by files of cursor reads. The graph is the image averaged across the slit,
along the dispersion; each bin is marked as IRAF marks it, a bar between two ticks
near the bottom, and shaded besides, so that bins which overlap are seen to.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import longslit
from ..core.i18n import N_, tr
from . import gtplot, keyguide, theme
from ._qt import on_main
from .theme import palette as _c

PLACEHOLDER = N_(":bins 1:200 400:600   ·   :show   ·   s at each end of a bin, or type "
                 "a colon command here")


class BinsPanel(QtWidgets.QWidget):
    """The spectrum along the dispersion, the bins, and the keys that set them."""

    finished = QtCore.Signal()

    def __init__(self, session: longslit.BinsSession, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self._cursor = (0.0, 0.0)
        self._seen = len(session.out)

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
        self.status.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        layout.addWidget(keyguide.strip(longslit.BINS_KEYS))
        self.command = QtWidgets.QLineEdit()
        self.command.setObjectName("command")
        self.command.setPlaceholderText(tr(PLACEHOLDER))
        self.command.returnPressed.connect(self._run_command)
        layout.addWidget(self.command)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plot.installEventFilter(self)
        self.setFocusProxy(self.plot)
        self.redraw(rescale=True)
        self._say()

    # ------------------------------------------------------------ drawing

    def redraw(self, rescale: bool = False) -> None:
        frame = self.session.frames[-1]
        logs = self.session.gt.logs()
        self.plot.clear()
        self.plot.setLabel("bottom", frame.xlabel, color=_c().muted)
        self.plot.setTitle("<br>".join(frame.title), color=_c().ink, size="10pt")
        self.plot.plot(*gtplot.on_axes(logs, frame.x, frame.y),
                       pen=pg.mkPen(_c().sky, width=1.2))
        y1, y2, y3 = frame.heights
        pen = pg.mkPen(_c().accent, width=1.5)
        for xa, xb, tick_a, tick_b in frame.regions:
            self.plot.plot(*gtplot.on_axes(logs, [xa, xb], [y3, y3]), pen=pen)
            for at, tick in ((xa, tick_a), (xb, tick_b)):
                if tick:
                    self.plot.plot(*gtplot.on_axes(logs, [at, at], [y1, y2]), pen=pen)
            if not logs[0]:
                region = pg.LinearRegionItem(sorted((xa - 0.5, xb + 0.5)), movable=False,
                                             brush=pg.mkBrush(_c().accent + "22"),
                                             pen=pg.mkPen(None))
                region.setZValue(-10)
                self.plot.addItem(region)
        if rescale:
            gtplot.set_window(self.plot, frame.window, logs)

    def _say(self) -> None:
        """What IRAF would print: the last lines of it, and the prompt waiting."""
        session = self.session
        text = session.out[self._seen:].replace("\a", "") + session.err
        session.err = ""
        lines = [line for line in text.splitlines() if line.strip()]
        if session.state == "s":
            lines.append(tr(N_("now s (or any key) at the bin's other end")))
        self.status.setText("\n".join([*lines[-8:], f"bins = {session.bins}"]))

    # ------------------------------------------------------------ the keys

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.Type.KeyPress and self.key(event.text()):
            return True
        return super().eventFilter(obj, event)

    def _mouse_moved(self, position) -> None:
        box = self.plot.getPlotItem().vb
        if box.sceneBoundingRect().contains(position):
            point = box.mapSceneToView(position)
            self._cursor = gtplot.world(self.session.gt.logs(), point.x(), point.y())

    def key(self, key: str, rest: str = "") -> bool:
        """Handle one keystroke (or, with ':', a colon command). True if it was ours."""
        session = self.session
        if not key:
            return False
        if key == ":" and not rest and not session.state:
            self.command.setFocus()
            self.command.setText(":")
            return True
        before = len(session.frames)
        if not session.state:
            self._seen = len(session.out)
        try:
            session.key(key, *self._cursor, rest=rest)
        except longslit.LongslitError:      # I: as q here; the task goes on
            session.done = True
        if session.done:
            self.finished.emit()
            return True
        self.redraw(rescale=len(session.frames) != before)
        self._say()
        return True

    def _run_command(self) -> None:
        text = self.command.text().strip().lstrip(":")
        self.command.clear()
        self.plot.setFocus()
        if text.startswith("."):
            self.session.out += gtplot.cursor_mode(self.plot, text[1:], "bins.png")
            self._say()
        elif text:
            self.key(":", rest=text)


@on_main
def run(session: longslit.BinsSession, title: str = "illumination") -> None:
    """Open the panel and wait for q."""
    from .dock import application, show

    application()   # before the panel: a widget made with no QApplication aborts Qt

    session.done = False
    show(BinsPanel(session), title, size=(1000, 560))
