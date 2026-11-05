"""The interactive half of fitparams: the fit of each equation, the keys, the
colon commands and the prompts, in a tab beside the image.

Nothing is computed here. Every number and every line of text comes from
`core.photcal.fitsession`, which is held to IRAF's by files of cursor commands
and answers, so a session at the keyboard and one from `--gcommands` are the same
session. Where IRAF asks at the terminal (save this fit? where next?) the panel
offers the answers as buttons; a line typed at the foot does the same, and an
empty one takes the default the prompt names.

One tab holds every equation, and stays after the last one to look through.

The graph is drawn as gtools leaves it -- the window `w` and `:/xwindow` set,
flipped axes, `:/transpose`, a logarithmic axis on IRAF's elog scale with its
decades marked, the marks `:/mark` names -- and the mouse zooms it besides;
nearest is measured in the view on the screen. `:.` commands are cursor mode's,
which in IRAF never reach the task: here the panel takes them (`:.snap file.png`
saves the graph, `:.help` says what there is).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.gtools import elog, unelog
from ..core.photcal import fitsession
from . import keyguide, theme
from ._qt import on_main
from .theme import palette as _c

HELP = fitsession.HELP

#: Cursor mode, as far as the panel has it.
CURSOR_MODE = [(":.snap [file]", "the graph to a file: .png, .svg, .pdf or .csv"),
               (":.write [file]", "the same"),
               (":.help", "these")]


def _bar(horizontal: bool, ends: bool) -> QtGui.QPainterPath:
    """gtools' hline, vline, hebar and vebar as pyqtgraph symbols."""
    path = QtGui.QPainterPath()
    a, b = (QtCore.QPointF(-0.5, 0), QtCore.QPointF(0.5, 0)) if horizontal else (
        QtCore.QPointF(0, -0.5), QtCore.QPointF(0, 0.5))
    path.moveTo(a)
    path.lineTo(b)
    if ends:
        for end in (a, b):
            tick = QtCore.QPointF(0, 0.2) if horizontal else QtCore.QPointF(0.2, 0)
            path.moveTo(end - tick)
            path.lineTo(end + tick)
    return path


#: A gtools mark as pyqtgraph draws it; lines and histograms of one point each
#: are nothing in IRAF, and dots here.
SYMBOLS = {"plus": "+", "cross": "x", "box": "s", "diamond": "d", "circle": "o",
           "point": "o", "line": "o", "histogram": "o",
           "hline": _bar(True, False), "vline": _bar(False, False),
           "hebar": _bar(True, True), "vebar": _bar(False, True)}

#: The answers worth a button, by what is being asked.
_CHOICES = {"save": ("yes", "no"), "quit": ("yes", "no"), "end": ("yes", "no"),
            "command": ("next", "again", "prev", "first", "last", "go", "quit"),
            "graphkey": ("h", "i", "j", "k", "l")}


class FitPanel(QtWidgets.QWidget):
    finished = QtCore.Signal()
    keep_after_quit = True

    def __init__(self, session: fitsession.FitSession, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.said = ""
        self._done = False

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

        self.params = QtWidgets.QLabel("")
        self.params.setObjectName("readout")
        self.params.setWordWrap(True)
        layout.addWidget(self.params)

        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumHeight(110)
        self.text.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        layout.addWidget(self.text)

        self.keys = keyguide.strip(HELP)
        layout.addWidget(self.keys)

        self.buttons = QtWidgets.QWidget()
        self._row = QtWidgets.QHBoxLayout(self.buttons)
        self._row.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.buttons)

        self.command = QtWidgets.QLineEdit()
        self.command.setObjectName("command")
        self.command.returnPressed.connect(self._run_command)
        layout.addWidget(self.command)

        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plot.installEventFilter(self)
        self.setFocusProxy(self.plot)

        self._cursor = (0.0, 0.0)
        self._sync(force=True)
        x0, x1, y0, y1 = session.graph().window
        self._cursor = ((x0 + x1) / 2, (y0 + y1) / 2)

    def redraw(self, keep_view: bool = False) -> None:
        graph = self.session.graph()
        if not graph.overplot:
            self.plot.clear()
        self.plot.setTitle("<br>".join(graph.title), color=_c().muted, size="10pt")
        self.plot.setLabel("bottom", graph.xlabel, color=_c().muted)
        self.plot.setLabel("left", graph.ylabel, color=_c().muted)
        self.params.setText("   ·   ".join(graph.params))
        # Marks that differ in shape, so that colour is never the only channel.
        scale = max(graph.size) / 2
        kept = 4 if graph.mark in ("point", "line", "histogram") else 11 * scale
        for (x, y), colour, symbol, size in (
                (graph.kept, _c().sky, SYMBOLS.get(graph.mark, "+"), kept),
                (graph.deleted, _c().muted, "x", 11 * scale),
                (graph.fit, _c().accent, "s", 7)):
            if len(x):
                self.plot.plot(*self._on_axes(x, y), pen=None, symbol=symbol,
                               symbolSize=size, symbolBrush=pg.mkBrush(colour),
                               symbolPen=pg.mkPen(colour, width=1.2))
        x, y = graph.rejected
        if len(x):
            self.plot.plot(*self._on_axes(x, y), pen=None, symbol="d", symbolSize=13,
                           symbolBrush=None, symbolPen=pg.mkPen(_c().drop, width=1.6))
        if not keep_view and not graph.overplot:
            self._set_window(graph)

    def _on_axes(self, x, y):
        """Where the values go on the plot: through elog on a logarithmic axis."""
        xlog, ylog = self.session.gt.logs()
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return (np.array([elog(v) for v in x]) if xlog else x,
                np.array([elog(v) for v in y]) if ylog else y)

    def _set_window(self, graph) -> None:
        """The window gtools gives, the way its axes run; a logarithmic axis
        drawn on elog, with ticks at its decades that say what they are."""
        x1, x2, y1, y2 = graph.window
        box = self.plot.getViewBox()
        box.invertX(x1 > x2)
        box.invertY(y1 > y2)
        ranges = []
        for name, (a, b), log in (("bottom", (x1, x2), graph.logs[0]),
                                  ("left", (y1, y2), graph.logs[1])):
            axis = self.plot.getAxis(name)
            if log:
                low, high = sorted((elog(a), elog(b)))
                steps = range(int(np.floor(low)), int(np.ceil(high)) + 1)
                axis.setTicks([[(float(k), f"{unelog(k):g}") for k in steps]])
                ranges.append((low, high))
            else:
                axis.setTicks(None)
                ranges.append(tuple(sorted((a, b))))
        box.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)

    # ---------------------------------------------------------- what is shown

    def _sync(self, force: bool = False) -> None:
        """Bring the screen up to what the session has done."""
        session = self.session
        new = session.take().replace("\a", "")
        if new:
            self.said += new
            self.text.moveCursor(self.text.textCursor().MoveOperation.End)
            self.text.insertPlainText(new)
            self.text.ensureCursorVisible()
        if session.newgraph or force:
            self.redraw()
        self._offer()
        if session.state == "done" and not self._done:
            self._done = True
            self.text.appendPlainText("The results are written. This tab stays to look "
                                      "through; its × closes it.")
            self.finished.emit()

    def _offer(self) -> None:
        """The buttons for what is being asked, and the line to type it in."""
        session = self.session
        while self._row.count():
            item = self._row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        asking = session.state not in ("fit", "done")
        self.buttons.setVisible(asking and session.state in _CHOICES)
        for word in _CHOICES.get(session.state, ()):
            button = QtWidgets.QPushButton(word)
            button.clicked.connect(lambda _=False, w=word: self._answer(w))
            self._row.addWidget(button)
        self._row.addStretch(1)
        self.command.setPlaceholderText(
            session.prompt if asking else
            ":errors   ·   press a key over the plot, or type a colon command here")
        (self.command if asking else self.plot).setFocus()

    # ------------------------------------------------------------- input

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt's name)
        if event.type() == QtCore.QEvent.Type.KeyPress and self.key(event.text()):
            return True
        return super().eventFilter(obj, event)

    def _world(self, x: float, y: float) -> tuple[float, float]:
        """A point of the plot as the graph's coordinates: back off elog."""
        xlog, ylog = self.session.gt.logs()
        return (unelog(x) if xlog else x), (unelog(y) if ylog else y)

    def _mouse_moved(self, position) -> None:
        box = self.plot.getViewBox()
        if box.sceneBoundingRect().contains(position):
            point = box.mapSceneToView(position)
            self._cursor = self._world(point.x(), point.y())

    def _scale(self) -> tuple[tuple, tuple[float, float]]:
        """The view as zoomed, in the graph's coordinates, and its size on the
        screen, so that nearest means nearest there."""
        box = self.plot.getViewBox()
        (x0, x1), (y0, y1) = box.viewRange()
        (x0, y0), (x1, y1) = self._world(x0, y0), self._world(x1, y1)
        return (x0, x1, y0, y1), (box.width() or 1.0, box.height() or 1.0)

    def key(self, key: str) -> bool:
        """Handle one keystroke. Returns True if it was ours. After w, the next
        printable key is the window's."""
        if self.session.pending and key and key.isprintable():
            self.session.key(key, *self._cursor, scale=self._scale())
            self._sync()
        elif key == ":":
            self.command.setFocus()
            self.command.setText(":")
        elif key and key in dict(HELP):
            self.session.key(key, *self._cursor, scale=self._scale())
            if key in "du":
                self.redraw(keep_view=True)
            self._sync()
        else:
            return False
        return True

    def _answer(self, text: str) -> None:
        self.text.insertPlainText(text + "\n")
        self.session.answer(text)
        self._sync()

    def _run_command(self) -> None:
        text = self.command.text().strip()
        self.command.clear()
        if self.session.state not in ("fit", "done"):
            self._answer(text)
            return
        self.plot.setFocus()
        rest = text.lstrip(":")
        if rest.startswith("."):
            self._cursor_mode(rest[1:])
        elif rest:
            self.session.key(":", rest=rest)
            self._sync()

    def _cursor_mode(self, text: str) -> None:
        """IRAF's cursor mode, the part of it worth having beside a mouse."""
        words = text.split()
        command = words[0] if words else ""
        if command and "help".startswith(command):
            said = "".join(f"{what:<16}{why}\n" for what, why in CURSOR_MODE)
        elif command and ("snap".startswith(command) or "write".startswith(command)):
            from .window import save_plot

            path = Path(words[1]) if len(words) > 1 else _free("fitparams.png")
            try:
                save_plot(self.plot, path)
                said = f"The graph is in {path}.\n"
            except (OSError, ValueError) as error:
                said = f"Could not write {path}: {error}. Name a .png, .svg, .pdf or .csv.\n"
        else:
            said = (f":.{command} is a cursor-mode command harwood does not have: the mouse "
                    "zooms and drags the graph, and :.snap file.png saves it. :.help lists "
                    "what there is.\n")
        self.text.moveCursor(self.text.textCursor().MoveOperation.End)
        self.text.insertPlainText(said)
        self.text.ensureCursorVisible()


def _free(name: str) -> Path:
    """name, or name with a number in it, so that nothing is written over."""
    path = Path(name)
    number = 1
    while path.exists():
        path = Path(f"{Path(name).stem}-{number}{Path(name).suffix}")
        number += 1
    return path


@on_main
def run(session: fitsession.FitSession, title: str = "fitparams") -> str:
    """Open the panel, wait for the last equation, and give back all that was said."""
    from .dock import show

    panel = show(FitPanel(session), title, size=(1000, 760))
    return panel.said
