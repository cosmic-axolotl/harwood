"""Marking arc lines by hand: IRAF's identify, with its keys.

The arc, with a tick over each line identified. Mark a few lines with m, fit with f,
which hands the keys to icfit until its q, then l lets the line list place the rest:
the sequence the IRAF tutorial walks through, and the only reliable way through a
thorium lamp whose dispersion you do not already know.

Nothing is computed here. Every number and every line of text comes from
`core.identifysession`, held to IRAF's by files of cursor commands and to what IRAF
draws by its graphics metacode. The graph is drawn as gtools leaves it
(`viewer.gtplot`): w and the /commands set its window, and the mouse zooms it
besides; `:.snap file.png` saves it.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import identifysession
from . import gtplot, keyguide, theme
from ._qt import on_main
from .icfit_panel import IcfitPanel
from .theme import palette as _c

HELP = identifysession.HELP

#: What a tick is, as a share of the plot's height: IRAF has it of the screen's.
_GAP, _TICK = identifysession.GAP / 0.74, identifysession.TICK / 0.74


class IdentifyPanel(QtWidgets.QWidget):
    """The arc and its features, or while f lasts the fit."""

    finished = QtCore.Signal()

    def __init__(self, session: identifysession.IdentifySession, parent=None) -> None:
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

        self.keys = keyguide.strip([("space" if k == " " else k, what) for k, what in HELP])
        page.addWidget(self.keys)

        self.command = QtWidgets.QLineEdit()
        self.command.setObjectName("command")
        self.command.returnPressed.connect(self._run_command)
        page.addWidget(self.command)

        self.setFocusProxy(self.plot)
        self.fit_panel: IcfitPanel | None = None
        self.session.start()
        self.redraw(rescale=True)
        self._after()

    # -------------------------------------------------------------- the plot

    def redraw(self, rescale: bool = False) -> None:
        session = self.session
        graph = session.graph()
        self.plot.clear()
        self.plot.setTitle(graph.title, color=_c().ink, size="10pt")
        self.plot.setLabel("bottom", graph.xlabel or "pixel", color=_c().muted)
        self.plot.setLabel("left", graph.ylabel, color=_c().muted)
        on_axes = self._on_axes
        if graph.kind == "mark":
            self.plot.plot(*on_axes(graph.x, graph.y), pen=None,
                           symbol=gtplot.SYMBOLS.get(graph.mark, "+"),
                           symbolSize=11 * max(graph.size) / 2, symbolBrush=None,
                           symbolPen=pg.mkPen(_c().sky, width=1.2))
        else:
            xs, ys = (gtplot.steps(graph.x, graph.y) if graph.kind == "histogram"
                      else (graph.x, graph.y))
            self.plot.plot(*on_axes(xs, ys), pen=pg.mkPen(_c().sky, width=1.2))
        # id_mark: the tick's gap and length in NDC, so through elog where it is.
        _, (y0, y1) = on_axes([], graph.window[2:])
        current = (session.features[session.current - 1].fit
                   if 0 < session.current <= len(session.features) else None)
        for tick in graph.ticks:
            sign = 1 if tick.upward else -1
            (x,), (base,) = on_axes([tick.x], [tick.y])
            start = base + sign * _GAP * (y1 - y0)
            end = start + sign * _TICK * (y1 - y0)
            width = 2.4 if tick.x == current else 1.2
            self.plot.plot([x, x], [start, end], pen=pg.mkPen(_c().accent, width=width))
            if tick.label:
                text = pg.TextItem(tick.label, color=_c().ink, angle=90,
                                   anchor=(0, 0.5) if tick.upward else (1, 0.5))
                text.setPos(x, end)
                self.plot.addItem(text)
        if rescale:
            gtplot.set_window(self.plot, graph.window, graph.logs)

    def _on_axes(self, x, y):
        return gtplot.on_axes(self.session.gt.logs(), np.asarray(x), np.asarray(y))

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
        if not (key in dict(HELP) or (self.session.pending and key.isprintable())):
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
            said = gtplot.cursor_mode(self.plot, rest[1:], "identify.png")
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
        rescale = session.newgraph
        self.redraw(rescale=rescale)
        lines = [line for line in said.replace("\a", "").splitlines() if line.strip()]
        if lines:
            self.status.setText("\n".join(lines[-3:]))
        asking = session.state != "keys"
        self.command.setPlaceholderText(
            session.prompt if asking else
            ":features   ·   press a key over the plot, or type a colon command here")
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
def run(session: identifysession.IdentifySession, title: str = "identify") -> str:
    """Open the panel, wait for q and its answer, and give back all that was said."""
    from .dock import show

    panel = show(IdentifyPanel(session), title, size=(1150, 780))
    return panel.said


@on_main
def ask_answer(current: str) -> str:
    """reidentify's 'Fit dispersion function interactively?', with IRAF's four answers
    as buttons: no, yes, NO (and stop asking), YES (and stop asking)."""
    box = QtWidgets.QMessageBox()
    box.setWindowTitle("reidentify")
    box.setText("Fit dispersion function interactively?")
    box.setInformativeText("No to all and Yes to all stop asking, as IRAF's NO and YES do.")
    buttons = {}
    for word, label, role in (("yes", "Yes", QtWidgets.QMessageBox.ButtonRole.YesRole),
                              ("no", "No", QtWidgets.QMessageBox.ButtonRole.NoRole),
                              ("YES", "Yes to all", QtWidgets.QMessageBox.ButtonRole.YesRole),
                              ("NO", "No to all", QtWidgets.QMessageBox.ButtonRole.NoRole)):
        buttons[box.addButton(label, role)] = word
    default = next(b for b, w in buttons.items() if w == current) if current in \
        buttons.values() else next(iter(buttons))
    box.setDefaultButton(default)
    box.exec()
    answer = buttons.get(box.clickedButton(), current)
    box.deleteLater()
    return answer
