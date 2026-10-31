"""sensfunc's graphs: the stars' sensitivities, their residuals, and the
keys that work on them.

The graphs are IRAF's (s sensitivity, r residuals, a residuals against
airmass) and core.sensfunc.Session answers the keys -- the engine a cursor
file drives -- so what is deleted, moved or shifted here is what the same
keys in a file would do. A key acts on the graph under the mouse.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import sensfunc as sf
from . import keyguide, theme
from ._qt import on_main
from .theme import palette as _c

HELP = [("d u", "delete, undelete: then p, s or w"), ("m", "move to the cursor"),
        ("w", "reweight"), ("a", "add a point"), ("s", "shift"), ("c", "composite"),
        ("e", "residual extinction"), ("f g", "fit"), ("o", "original"), ("q", "done")]


class SensfuncPanel(QtWidgets.QWidget):
    finished = QtCore.Signal()

    def __init__(self, session: sf.Session, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self._cursor = (0.0, 0.0, 1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.setBackground(_c().ground)
        theme.follow(self, [self.canvas])
        self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.canvas.installEventFilter(self)
        self.canvas.scene().sigMouseMoved.connect(self._moved)
        layout.addWidget(self.canvas, stretch=1)
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        layout.addWidget(self.status)
        keys = keyguide.strip(HELP)
        keys.setObjectName("keys")
        keys.setWordWrap(True)
        layout.addWidget(keys)
        self.command = QtWidgets.QLineEdit()
        self.command.setObjectName("command")
        self.command.setPlaceholderText(":order 4   ·   :function legendre   ·   "
                                        ":graphs sra   ·   :vstats")
        self.command.returnPressed.connect(self._run_command)
        layout.addWidget(self.command)
        self.plots: list = []
        self.render()

    def render(self) -> None:
        self.canvas.clear()
        self.plots = []
        session = self.session
        for index, graph in enumerate(session.drawn()):
            plot = self.canvas.addPlot(row=index, col=0, title=graph.title)
            for axis in ("left", "bottom"):
                item = plot.getAxis(axis)
                item.setPen(pg.mkPen(_c().line))
                item.setTextPen(pg.mkPen(_c().muted))
                item.enableAutoSIPrefix(False)
            self.plots.append(plot)
            # Marks that differ in shape, so that colour is never the only channel.
            for kind, symbol, colour in (("plus", "+", _c().accent), ("box", "s", _c().accent),
                                         ("cross", "x", _c().warning)):
                points = [(x, y) for mark, x, y in graph.marks if mark == kind]
                if points:
                    x, y = zip(*points, strict=True)
                    plot.plot(x, y, pen=None, symbol=symbol, symbolPen=pg.mkPen(colour),
                              symbolBrush=None, symbolSize=8)
            for x, y, dashed in graph.lines:
                style = QtCore.Qt.PenStyle.DashLine if dashed else QtCore.Qt.PenStyle.SolidLine
                plot.plot(x, y, pen=pg.mkPen(_c().line, style=style))
            if graph.bars is not None:
                x, y, height = graph.bars
                beam = (x.max() - x.min()) / 80
                plot.addItem(pg.ErrorBarItem(x=x, y=y, height=height, beam=beam,
                                             pen=pg.mkPen(_c().accent)))
            if graph.widths is not None:
                x, y, width = graph.widths
                plot.addItem(pg.ErrorBarItem(x=x, y=y, width=width, beam=0,
                                             pen=pg.mkPen(_c().warning)))
            x1, x2, y1, y2 = (float(v) for v in graph.window)
            if graph.kind in "ra":
                plot.addLine(y=0, pen=pg.mkPen(_c().line))
            plot.setXRange(x1, x2, padding=0)
            plot.setYRange(y1, y2, padding=0)
        self.status.setText(session.title())

    def _moved(self, position) -> None:
        for index, plot in enumerate(self.plots):
            if plot.sceneBoundingRect().contains(position):
                point = plot.vb.mapSceneToView(position)
                self._cursor = (point.x(), point.y(), index + 1)
                return

    def key(self, key: str, command: str = "") -> None:
        x, y, wcs = self._cursor
        session = self.session
        session.key(x, y, wcs, key, command)
        pending = session.pending
        if pending is not None and pending[0] == "w" and isinstance(pending[2], tuple):
            star, point = pending[1]
            weight, ok = QtWidgets.QInputDialog.getDouble(
                self, "sensfunc", "New weight",
                float(session.stars[star].iwts[point]), 0.0, 1e6, 3)
            session.key(x, y, wcs, ":", str(weight) if ok else "")
            session.pending = None
        elif pending is not None and pending[0] == "e":
            answer = QtWidgets.QMessageBox.question(
                self, "sensfunc", session.messages[-1].strip())
            session.key(x, y, wcs,
                        "y" if answer == QtWidgets.QMessageBox.StandardButton.Yes else "n")
        for message in session.messages[-2:]:
            if message.strip("\a"):
                self.status.setText(message.strip("\a").splitlines()[0])
        if command.startswith(("stats", "vstats")) and session.messages and \
                not command.split()[1:]:
            QtWidgets.QMessageBox.information(self, "sensfunc", session.messages[-1])
        if session.done:
            self.finished.emit()
            return
        if pending is None or session.pending is None:
            self.render()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt's name)
        if event.type() == QtCore.QEvent.Type.KeyPress and event.text():
            text = event.text()
            if text == ":":
                self.command.setFocus()
                self.command.setText(":")
            elif text != "I":
                self.key(text)
            return True
        return super().eventFilter(obj, event)

    def _run_command(self) -> None:
        text = self.command.text().strip().lstrip(":").strip()
        self.command.clear()
        self.canvas.setFocus()
        if text:
            self.key(":", text)


@on_main
def edit(session: sf.Session) -> None:
    """Show the graphs until q."""
    from .dock import show

    show(SensfuncPanel(session), f"sensfunc · aperture {session.aperture}",
         size=(1000, 720))


class WindowCursor:
    """The window in place of a cursor file."""

    def edit(self, session: sf.Session) -> None:
        edit(session)
