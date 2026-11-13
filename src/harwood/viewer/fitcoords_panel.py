"""The graph of fitcoords by hand -- IRAF's igsfit -- and its keys.

Nothing is decided here: every key goes to `core.igsfit.Session`, which is held to
IRAF's by files of cursor reads. The graph is drawn as gtools leaves it
(`viewer.gtplot`): the points kept as +, the deleted as x; the keys that ask a second
key (x y z d u w) say so under the graph, as IRAF's prompts do. The mouse zooms
besides, and `:.snap file.png` saves the graph.

One difference, on purpose: a fit that cannot be made (no degrees of freedom left
after deleting) ends IRAF's task; here it is said, and the fit before it stays.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import fitcoords as fc
from ..core import igsfit
from . import gtplot, keyguide, theme
from ._qt import on_main
from .theme import palette as _c


class FitcoordsPanel(QtWidgets.QWidget):
    """The feature coordinates, their residuals, and the keys that change the fit."""

    finished = QtCore.Signal()

    def __init__(self, session: igsfit.Session, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self._cursor = (0.0, 0.0)
        self._seen = len(session.out)       # what was printed before the last key

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
        layout.addWidget(keyguide.strip(igsfit.KEYS))
        self.command = QtWidgets.QLineEdit()
        self.command.setObjectName("command")
        self.command.setPlaceholderText(
            ":xorder 4   ·   :show   ·   :/xwindow   ·   press a key over the plot, or "
            "type a colon command here")
        self.command.returnPressed.connect(self._run_command)
        layout.addWidget(self.command)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plot.installEventFilter(self)
        self.setFocusProxy(self.plot)
        self.redraw(rescale=True)
        self._say()

    # ------------------------------------------------------------ drawing

    def redraw(self, rescale: bool = False) -> None:
        session = self.session
        frame = session.frames[-1] if session.frames else session.graph()
        self.plot.clear()
        self.plot.setLabel("bottom", frame.xlabel, color=_c().muted)
        self.plot.setLabel("left", frame.ylabel, color=_c().muted)
        self.plot.setTitle("<br>".join(frame.title), color=_c().ink, size="10pt")
        # what d and u changed since the graph was drawn is shown at once
        now = {(x, y): kind for kind, x, y, erased in frame.over if not erased}
        kept = [(x, y) for kind, x, y in frame.marks if now.get((x, y), kind) == "plus"]
        gone = [(x, y) for kind, x, y in frame.marks if now.get((x, y), kind) == "cross"]
        # + and x: the shape says it, not the colour alone
        for points, colour, symbol in ((kept, _c().sky, "+"), (gone, _c().drop, "x")):
            if points:
                xs, ys = zip(*points, strict=True)
                self.plot.plot(*gtplot.on_axes(frame.logs, xs, ys), pen=None,
                               symbol=symbol, symbolSize=10, symbolBrush=None,
                               symbolPen=pg.mkPen(colour, width=1.5))
        if rescale:
            gtplot.set_window(self.plot, frame.window, frame.logs)

    def _say(self, extra: str = "") -> None:
        """What IRAF would print: the last lines of it, and the prompt waiting."""
        session = self.session
        text = session.out[self._seen:].replace("\a", "")
        lines = [line for line in text.splitlines() if line.strip()]
        shown = "\n".join(lines[-8:])
        if extra:
            shown = (shown + "\n" + extra).strip()
        c = session.common
        rms = "INDEF" if session.rms >= igsfit.INDEFR else f"{session.rms:.4g}"
        summary = (f"{c.function}, xorder {c.xorder}, yorder {c.yorder}: rms {rms}, "
                   f"{int((session.pts[fc.W] == 0).sum())} of {session.coords.n} "
                   "points deleted.")
        if session.state in ("x", "y"):
            summary += "  The new axis is drawn at r."
        self.status.setText(shown + ("\n" if shown else "") + summary)

    # ------------------------------------------------------------ the keys

    def _scale(self):
        frame = self.session.frames[-1] if self.session.frames else None
        return gtplot.scale(self.plot, frame.logs if frame else (False, False))

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.Type.KeyPress and self.key(event.text()):
            return True
        return super().eventFilter(obj, event)

    def _mouse_moved(self, position) -> None:
        box = self.plot.getPlotItem().vb
        if box.sceneBoundingRect().contains(position):
            point = box.mapSceneToView(position)
            frame = self.session.frames[-1] if self.session.frames else None
            self._cursor = gtplot.world(frame.logs if frame else (False, False),
                                        point.x(), point.y())

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
        problem = ""
        try:
            session.key(key, *self._cursor, rest=rest, scale=self._scale())
        except fc.FitcoordsError as error:
            if str(error) == "Interrupt":
                session.interrupted = True
                session.done = True
            else:
                problem = f"{error}: the fit before it is kept. Undelete points or lower " \
                          "the orders, and press f again."
        if session.done:
            self.finished.emit()
            return True
        self.redraw(rescale=len(session.frames) != before)
        self._say(problem)
        return True

    def _run_command(self) -> None:
        text = self.command.text().strip().lstrip(":")
        self.command.clear()
        self.plot.setFocus()
        if text.startswith("."):
            self._say(gtplot.cursor_mode(self.plot, text[1:], "fitcoords.png").strip())
        elif text:
            self.key(":", rest=text)


@on_main
def run(session: igsfit.Session, title: str = "fitcoords") -> None:
    """Open the panel and wait for q."""
    from .dock import application, show

    application()   # before the panel: a widget made with no QApplication aborts Qt

    session.done = False
    show(FitcoordsPanel(session), title, size=(1000, 660))


def _ask(title: str, text: str, choices, current: str, more: str = "") -> str:
    from .dock import application

    application()
    box = QtWidgets.QMessageBox()
    box.setWindowTitle(title)
    box.setText(text)
    if more:
        box.setInformativeText(more)
    buttons = {}
    for word, label, role in choices:
        buttons[box.addButton(label, role)] = word
    default = next((b for b, w in buttons.items() if w == current), next(iter(buttons)))
    box.setDefaultButton(default)
    box.exec()
    answer = buttons.get(box.clickedButton(), current)
    box.deleteLater()
    return answer


@on_main
def ask_answer(question: str, current: str) -> str:
    """'Fit <image> interactively?', with IRAF's four answers as buttons."""
    yes, no = QtWidgets.QMessageBox.ButtonRole.YesRole, QtWidgets.QMessageBox.ButtonRole.NoRole
    return _ask("fitcoords", question,
                (("yes", "Yes", yes), ("no", "No", no), ("YES", "Yes to all", yes),
                 ("NO", "No to all", no)), current,
                "No to all and Yes to all stop asking, as IRAF's NO and YES do.")


@on_main
def ask_write() -> bool:
    """'Write coordinate map to the database?'"""
    yes, no = QtWidgets.QMessageBox.ButtonRole.YesRole, QtWidgets.QMessageBox.ButtonRole.NoRole
    return _ask("fitcoords", "Write coordinate map to the database?",
                (("yes", "Yes", yes), ("no", "No", no)), "yes",
                "No leaves database/fc<fitname> as it was; the deleted points are "
                "kept in the deletion list either way.") == "yes"
