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

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.photcal import fitsession
from . import gtplot, keyguide, theme
from ._qt import on_main
from .theme import palette as _c

HELP = fitsession.HELP

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
                (graph.kept, _c().sky, gtplot.SYMBOLS.get(graph.mark, "+"), kept),
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
        return gtplot.on_axes(self.session.gt.logs(), x, y)

    def _set_window(self, graph) -> None:
        gtplot.set_window(self.plot, graph.window, graph.logs)

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

    def _mouse_moved(self, position) -> None:
        box = self.plot.getViewBox()
        if box.sceneBoundingRect().contains(position):
            point = box.mapSceneToView(position)
            self._cursor = gtplot.world(self.session.gt.logs(), point.x(), point.y())

    def _scale(self) -> tuple[tuple, tuple[float, float]]:
        return gtplot.scale(self.plot, self.session.gt.logs())

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
        said = gtplot.cursor_mode(self.plot, text, "fitparams.png")
        self.text.moveCursor(self.text.textCursor().MoveOperation.End)
        self.text.insertPlainText(said)
        self.text.ensureCursorVisible()


@on_main
def run(session: fitsession.FitSession, title: str = "fitparams") -> str:
    """Open the panel, wait for the last equation, and give back all that was said."""
    from .dock import show

    panel = show(FitPanel(session), title, size=(1000, 760))
    return panel.said
