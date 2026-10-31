"""The interactive half of mkapfile: the curve of growth, the keys, the colon
commands, in a tab beside the image.

Nothing is computed here. Every number and every line of text comes from
`core.photcal.apsession`, which is held to IRAF's by a file of cursor commands,
so a session at the keyboard and one from `--gcommands` are the same session.

The tab stays after q, as a graph does: the results are written by then, the
graphs can still be gone through, and the × closes it.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.photcal import apsession
from . import keyguide, theme
from ._qt import on_main
from .theme import palette as _c

HELP = apsession.HELP

#: The keys after which what is on the screen is out of date.
_REDRAW = set("mrbxyagnpfdu")


class CogPanel(QtWidgets.QWidget):
    finished = QtCore.Signal()
    keep_after_quit = True

    def __init__(self, session: apsession.ApSession, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.said = list(session.printed)
        self._quit = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(_c().ground)
        theme.follow(self, [self.plot], lambda: self.redraw(keep_view=True))
        for name in ("left", "bottom"):
            axis = self.plot.getAxis(name)
            axis.setPen(pg.mkPen(_c().line))
            axis.setTextPen(pg.mkPen(_c().muted))
            axis.enableAutoSIPrefix(False)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        layout.addWidget(self.plot, stretch=1)

        # All that was said, not the last line: `:show model` is a dozen lines.
        self.text = QtWidgets.QPlainTextEdit("\n".join(self.said))
        self.text.setReadOnly(True)
        self.text.setMaximumHeight(110)
        self.text.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        layout.addWidget(self.text)

        self.keys = keyguide.strip(HELP)
        layout.addWidget(self.keys)

        self.command = QtWidgets.QLineEdit()
        self.command.setObjectName("command")
        self.command.setPlaceholderText(":show model   ·   press a key over the plot, "
                                        "or type a colon command here")
        self.command.returnPressed.connect(self._run_command)
        layout.addWidget(self.command)

        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plot.installEventFilter(self)
        self.setFocusProxy(self.plot)

        self.redraw()
        x0, x1, y0, y1 = session.graph().window
        self._cursor = ((x0 + x1) / 2, (y0 + y1) / 2)

    def redraw(self, keep_view: bool = False) -> None:
        graph = self.session.graph()
        self.plot.clear()
        self.plot.setTitle("<br>".join(graph.title), color=_c().muted, size="10pt")
        self.plot.setLabel("bottom", graph.xlabel, color=_c().muted)
        self.plot.setLabel("left", graph.ylabel, color=_c().muted)
        # Marks that differ in shape, so that colour is never the only channel.
        for (x, y), colour, symbol in ((graph.kept, _c().sky, "+"),
                                       (graph.dropped, _c().drop, "x")):
            if len(x):
                self.plot.plot(x, y, pen=None, symbol=symbol, symbolSize=11,
                               symbolBrush=pg.mkBrush(colour), symbolPen=None)
        for x, y, dashed in graph.lines:
            pen = (pg.mkPen(_c().muted, width=1.4, style=QtCore.Qt.PenStyle.DashLine)
                   if dashed else pg.mkPen(_c().accent, width=1.4))
            self.plot.plot(x, y, pen=pen, connect="finite")
        if not keep_view:
            x0, x1, y0, y1 = graph.window
            box = self.plot.getViewBox()
            box.invertY(y0 > y1)
            box.setRange(xRange=(x0, x1), yRange=(min(y0, y1), max(y0, y1)), padding=0)

    def _say(self, lines: list[str]) -> None:
        self.said += lines
        for line in lines:
            self.text.appendPlainText(line)

    # ------------------------------------------------------------- input

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt's name)
        if event.type() == QtCore.QEvent.Type.KeyPress and self.key(event.text()):
            return True
        return super().eventFilter(obj, event)

    def _mouse_moved(self, position) -> None:
        box = self.plot.getViewBox()
        if box.sceneBoundingRect().contains(position):
            point = box.mapSceneToView(position)
            self._cursor = (point.x(), point.y())

    def _scale(self) -> tuple[float, float]:
        """Pixels per unit on each axis, so that nearest means nearest on the
        screen, however the plot has been zoomed."""
        box = self.plot.getViewBox()
        (x0, x1), (y0, y1) = box.viewRange()
        return (box.width() or 1.0) / max(x1 - x0, 1e-12), \
            (box.height() or 1.0) / max(y1 - y0, 1e-12)

    def key(self, key: str) -> bool:
        """Handle one keystroke. Returns True if it was ours."""
        if key == "q":
            if not self._quit:
                self._quit = True
                self.text.appendPlainText("The results are written. This tab stays "
                                          "to look through; its × closes it.")
            self.finished.emit()
        elif key == ":":
            self.command.setFocus()
            self.command.setText(":")
        elif key and key in dict(HELP):
            self._say(self.session.key(key, *self._cursor, scale=self._scale()))
            if key in _REDRAW:
                self.redraw(keep_view=key in "du")
        else:
            return False
        return True

    def _run_command(self) -> None:
        text = self.command.text().strip().lstrip(":")
        self.command.clear()
        self.plot.setFocus()
        if text:
            self._say(self.session.key(":", rest=text))
            self.redraw()


@on_main
def run(session: apsession.ApSession, title: str = "mkapfile") -> list[str]:
    """Open the panel, wait for q, and give back all that was said."""
    from .dock import show

    panel = show(CogPanel(session), title, size=(980, 700))
    return panel.said
