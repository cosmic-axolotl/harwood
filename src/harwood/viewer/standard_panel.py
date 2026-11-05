"""standard's bands, on the spectrum: add one by marking its ends, delete one.

The spectrum is drawn with each band as a box at its mean counts per pixel,
as IRAF's std_graph draws it. The keys are IRAF's and core.stdstar.Editor
answers them -- the same engine a cursor file drives -- so the bands chosen
here are the ones a cursor file with the same keys would choose. The graph is
drawn as gtools leaves it (`viewer.gtplot`): w and the /commands set its window,
and the mouse zooms it besides; `:.snap file.png` saves it.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import gtplot, keyguide, theme
from ._qt import on_main
from .theme import palette as _c

HELP = [("a", "add a band: press at both ends"), ("d", "delete the nearest band"),
        (":show", "list the bands"), ("r", "redraw"),
        ("w", "window, then a key: a b c d e f g j k l m n p r t u x y z ?"),
        ("q", "done with this spectrum")]


class StandardPanel(QtWidgets.QWidget):
    finished = QtCore.Signal()

    def __init__(self, editor, parent=None) -> None:
        super().__init__(parent)
        self.editor = editor
        self._x = float(editor.spec.wave[0])
        self._y = 0.0
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(_c().ground)
        theme.follow(self, [self.plot])
        for axis in ("left", "bottom"):
            item = self.plot.getAxis(axis)
            item.setPen(pg.mkPen(_c().line))
            item.setTextPen(pg.mkPen(_c().muted))
            item.enableAutoSIPrefix(False)
        self.plot.scene().sigMouseMoved.connect(self._moved)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plot.installEventFilter(self)
        layout.addWidget(self.plot, stretch=1)
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        layout.addWidget(self.status)
        keys = keyguide.strip(HELP)
        keys.setObjectName("keys")
        layout.addWidget(keys)
        self.command = QtWidgets.QLineEdit()
        self.command.setObjectName("command")
        self.command.setPlaceholderText(":show")
        self.command.returnPressed.connect(self._run_command)
        layout.addWidget(self.command)
        self.render(rescale=True)

    def render(self, rescale: bool = False) -> None:
        editor = self.editor
        graph = editor.graph()
        spec = editor.spec
        title = (f"{spec.image}{spec.section}({spec.ap}): {graph.title}" if graph.image
                 else graph.title)
        self.plot.setTitle(title, color=_c().muted, size="10pt")
        self.plot.setLabel("bottom", graph.xlabel, color=_c().muted)
        self.plot.setLabel("left", graph.ylabel, color=_c().muted)
        self.plot.clear()
        on_axes = self._on_axes
        if graph.kind == "mark":
            self.plot.plot(*on_axes(graph.x, graph.y), pen=None,
                           symbol=gtplot.SYMBOLS.get(graph.mark, "+"), symbolSize=8,
                           symbolBrush=None, symbolPen=pg.mkPen(_c().accent, width=1.2))
        else:
            xs, ys = (gtplot.steps(graph.x, graph.y) if graph.kind == "histogram"
                      else (graph.x, graph.y))
            self.plot.plot(*on_axes(xs, ys), pen=pg.mkPen(_c().accent, width=1.2))
        # gmark's box: as wide as the band, and of a height in NDC, so through elog.
        _, (y0, y1) = on_axes([], graph.window[2:])
        half = graph.box_height / (graph.window[3] - graph.window[2] or 1.0) * (y1 - y0) / 2
        for wave, dwave, level in graph.boxes:
            (left, right), (middle,) = on_axes([wave - dwave / 2, wave + dwave / 2], [level])
            x = [left, right, right, left, left]
            y = [middle - half, middle - half, middle + half, middle + half, middle - half]
            self.plot.plot(x, y, pen=pg.mkPen(_c().warning, width=1.4))
        if rescale:
            gtplot.set_window(self.plot, graph.window, graph.logs)
        pending = editor.pending
        if pending is not None:
            (at,), _ = on_axes([float(pending)], [])
            self.plot.addItem(pg.InfiniteLine(angle=90, pos=at,
                                              pen=pg.mkPen(_c().warning, width=1)))
        self.status.setText(f"{len(editor.bands)} bands" + (
            " · a again at the other end" if pending is not None else ""))

    def _on_axes(self, x, y):
        return gtplot.on_axes(self.editor.gt.logs(), np.asarray(x, dtype=float),
                              np.asarray(y, dtype=float))

    def _moved(self, position) -> None:
        point = self.plot.getViewBox().mapSceneToView(position)
        self._x, self._y = gtplot.world(self.editor.gt.logs(), point.x(), point.y())

    def key(self, key: str, command: str = "") -> None:
        window = self.editor.window
        self.editor.key(self._x, key, command, self._y)
        for message in self.editor.messages:
            if message.strip("\a"):
                self.status.setText(message.strip("\a").splitlines()[0])
        if command == "show" and self.editor.messages:
            QtWidgets.QMessageBox.information(self, "standard star data",
                                              self.editor.messages[-1])
        self.editor.messages.clear()
        if self.editor.done:
            self.finished.emit()
            return
        self.render(rescale=self.editor.newgraph or self.editor.window != window)

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
        self.plot.setFocus()
        if text.startswith("."):
            self.status.setText(gtplot.cursor_mode(self.plot, text[1:], "standard.png")
                                .strip())
        elif text:
            self.key(":", text)


@on_main
def edit(editor) -> None:
    """Show the spectrum and its bands until q."""
    from .dock import show

    spec = editor.spec
    show(StandardPanel(editor), f"standard · {spec.image}({spec.ap})", size=(1000, 600))


class WindowCursor:
    """The window in place of a cursor file, for stdstar.run."""

    def edit(self, editor) -> None:
        edit(editor)
