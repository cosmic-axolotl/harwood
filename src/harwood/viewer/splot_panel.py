"""splot's window: the spectrum, what the keys draw over it, and IRAF's keys.

Nothing is measured here: every key goes to `core.splot.Splot`, held to IRAF's by
files of cursor commands, which runs in a thread of its own and reads the keys
from this window as IRAF's splot reads the cursor -- e, k and d take their second
point, deblend its line marks and its choices, from the keys that follow. When
splot waits for a key the window draws what it has: the graph as gtools leaves it
(`viewer.gtplot`), the overplots, the fits and lines the keys drew, and on the
status line what splot printed. A question (the box of s, the constant of f's m,
the next image of g) is asked on the command line below the graph.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import splot
from . import gtplot, keyguide, theme
from ._qt import on_main
from .theme import palette as _c

END = object()


class _Reads:
    """The cursor as splot reads it: blocks until the window sends a key."""

    def __init__(self, panel: SplotPanel):
        self.panel = panel

    def __iter__(self):
        return self

    def __next__(self):
        self.panel._idle.set()
        item = self.panel._keys.get()
        self.panel._idle.clear()
        if item is END:
            raise StopIteration
        return item


class _Answers:
    """splot's questions: asked on the command line, answered with Enter."""

    def __init__(self, panel: SplotPanel):
        self.panel = panel

    def __iter__(self):
        return self

    def __next__(self):
        self.panel._asking.set()
        self.panel._idle.set()
        item = self.panel._answers.get()
        self.panel._asking.clear()
        self.panel._idle.clear()
        if item is END:
            raise StopIteration
        return item


class SplotPanel(QtWidgets.QWidget):
    """The spectrum and splot's keys."""

    finished = QtCore.Signal()

    def __init__(self, params: dict, images: list[str], parent=None) -> None:
        super().__init__(parent)
        self._keys: queue.Queue = queue.Queue()
        self._answers: queue.Queue = queue.Queue()
        self._idle = threading.Event()
        self._asking = threading.Event()
        self._cursor = (0.0, 0.0)
        self._shown_out = 0
        self._frames = -1
        self.error = None
        self.session = splot.Splot(params, _Answers(self))
        from . import icfit_panel

        # t's continuum fit opens icfit's own window, as IRAF's icg_fit takes over
        # the graphics
        self.session.icfit_window = icfit_panel.make

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(_c().ground)
        theme.follow(self, [self.plot], lambda: self.redraw(rescale=False))
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
        layout.addWidget(keyguide.strip(splot.HELP))
        self.command = QtWidgets.QLineEdit()
        self.command.setObjectName("command")
        self.command.setPlaceholderText(
            ":units nm   ·   press a key over the plot, or type a colon command here")
        self.command.returnPressed.connect(self._run_command)
        layout.addWidget(self.command)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plot.installEventFilter(self)
        self.setFocusProxy(self.plot)

        self._thread = threading.Thread(target=self._run, args=(images,), daemon=True)
        self._thread.start()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(40)

    # ---------------------------------------------------------- the session

    def _run(self, images) -> None:
        try:
            self.session.run(images, _Reads(self))
        except Exception as error:  # noqa: BLE001 -- shown, and the window closes
            self.error = error
        finally:
            self._idle.set()

    def _poll(self) -> None:
        if not self._idle.is_set():
            return
        if not self._thread.is_alive():
            self._timer.stop()
            self.redraw(rescale=False)
            if self.error is not None:
                self.status.setText(f"splot stopped: {self.error}")
            self.finished.emit()
            return
        if self._asking.is_set():
            name = self.session.asking
            prompt = splot.PROMPTS.get(name, name)
            self.command.setPlaceholderText(f"{prompt} -- type it and press Enter")
            if not self.command.hasFocus():
                self.command.setFocus()
        frames = len(self.session.graphs)
        self.redraw(rescale=frames != self._frames)
        self._frames = frames

    def send(self, key: str, command: str = "") -> None:
        """A cursor read at the mouse's position."""
        self._idle.clear()
        self._keys.put((float(self._cursor[0]), float(self._cursor[1]), 1, key, command))

    # ------------------------------------------------------------- drawing

    def redraw(self, rescale: bool) -> None:
        session = self.session
        if not session.graphs:
            return
        frame = session.graphs[-1]
        self.plot.clear()
        self.plot.setTitle(frame["title"], color=_c().ink, size="10pt")
        self.plot.setLabel("bottom", frame["xlabel"], color=_c().muted)
        sh = session.sh
        if sh is not None and sh.flabel:
            self.plot.setLabel("left", sh.flabel + (f" ({sh.funits})" if sh.funits else ""),
                               color=_c().muted)
        for k, (x, y) in enumerate(frame["curves"]):
            if frame.get("histogram"):
                x, y = gtplot.steps(x, y)
            style = QtCore.Qt.PenStyle.SolidLine if k == 0 else QtCore.Qt.PenStyle.DashLine
            self.plot.plot(np.asarray(x, float), np.asarray(y, float),
                           pen=pg.mkPen(_c().ink if k == 0 else _c().sky, width=1.2,
                                        style=style))
        for item in session.drawn:
            self._overlay(item)
        if rescale:
            gtplot.set_window(self.plot, session.window, (False, False))
        new = session.out[self._shown_out:]
        self._shown_out = len(session.out)
        lines = [line for line in new.replace("\a", "").splitlines() if line.strip()]
        if lines:
            self.status.setText("\n".join(lines[-3:]))

    @staticmethod
    def _pen(kind: int):
        """IRAF's line types as splot uses them -- 1 what the keys mark, 2 a total fit,
        3 a component, 4 the continuum -- each its own style, not only its colour."""
        style = {1: QtCore.Qt.PenStyle.SolidLine, 2: QtCore.Qt.PenStyle.DashLine,
                 3: QtCore.Qt.PenStyle.DotLine, 4: QtCore.Qt.PenStyle.DashDotLine}
        colour = {1: _c().accent, 2: _c().accent, 3: _c().sky, 4: _c().muted}
        return pg.mkPen(colour.get(kind, _c().accent), width=1.4,
                        style=style.get(kind, QtCore.Qt.PenStyle.SolidLine))

    def _overlay(self, item) -> None:
        kind = item[0]
        solid = self._pen(1)
        if kind == "line":
            x1, y1, x2, y2 = (float(v) for v in item[1])
            self.plot.plot([x1, x2], [y1, y2], pen=self._pen(item[2] if len(item) > 2 else 1))
        elif kind == "curve":
            x, y = item[1]
            self.plot.plot(np.asarray(x, float), np.asarray(y, float),
                           pen=self._pen(item[2] if len(item) > 2 else 2))
        elif kind == "polyline":
            pts = item[1]
            if pts:
                xs, ys = zip(*pts, strict=True)
                self.plot.plot(np.asarray(xs, float), np.asarray(ys, float),
                               pen=self._pen(item[2]))
        elif kind == "vline":
            x, y = (float(v) for v in item[1])
            self.plot.plot([x], [y], pen=None, symbol="t1", symbolSize=12,
                           symbolBrush=None, symbolPen=solid)
        elif kind == "box":
            x1, y1, x2, y2 = (float(v) for v in item[1])
            self.plot.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], pen=solid)
        elif kind in ("label", "mabove", "mbelow"):
            x, y, text = item[1][:3]
            anchor = {"label": (0.5, 0.5), "mabove": (0.5, 1.0), "mbelow": (0.5, 0.0)}
            label = pg.TextItem(str(text), color=_c().ink, anchor=anchor[kind])
            label.setPos(float(x), float(y))
            self.plot.addItem(label)
            if kind != "label":
                box = self.plot.getViewBox()
                (_, _), (y0, y1) = box.viewRange()
                span = (y1 - y0) or 1.0
                sign = 1 if kind == "mabove" else -1
                self.plot.plot([float(x), float(x)],
                               [float(y) + sign * 0.02 * span,
                                float(y) + sign * 0.05 * span], pen=solid)

    # --------------------------------------------------------------- input

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.Type.KeyPress and self.key(event.text()):
            return True
        return super().eventFilter(obj, event)

    def _mouse_moved(self, position) -> None:
        box = self.plot.getPlotItem().vb
        if box.sceneBoundingRect().contains(position):
            point = box.mapSceneToView(position)
            self._cursor = (point.x(), point.y())

    def key(self, key: str) -> bool:
        """One keystroke over the graph: to splot, unless it is a question's turn."""
        if not key or not self._thread.is_alive():
            return False
        if self._asking.is_set():
            self.command.setFocus()
            return True
        if key == ":":
            self.command.setFocus()
            self.command.setText(":")
            return True
        self.send(key)
        return True

    def _run_command(self) -> None:
        text = self.command.text().strip()
        self.command.clear()
        self.command.setPlaceholderText(
            ":units nm   ·   press a key over the plot, or type a colon command here")
        self.plot.setFocus()
        if self._asking.is_set():
            self._idle.clear()
            self._answers.put(text)
            return
        text = text.lstrip(":")
        if text.startswith("."):
            self.status.setText(gtplot.cursor_mode(self.plot, text[1:], "splot.png").strip())
        elif text:
            self.send(":", text)

    def closeEvent(self, event) -> None:
        self._keys.put(END)
        self._answers.put(END)
        super().closeEvent(event)


@on_main
def run(params: dict, images: list[str], title: str = "splot") -> SplotPanel:
    """Open the window; splot runs until q on the last image, or the window closes."""
    from .dock import application, show

    application()   # before the panel: a widget made with no QApplication aborts Qt

    panel = SplotPanel(params, images)
    show(panel, title, size=(1000, 640))
    # closed from its frame, the panel's own closeEvent does not run: splot is told
    # here that the cursor has ended, and goes on to its end
    panel._keys.put(END)
    panel._answers.put(END)
    panel._thread.join(timeout=5)
    return panel
