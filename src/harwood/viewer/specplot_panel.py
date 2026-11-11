"""specplot's window: the spectra stacked, their labels, and IRAF's keys.

Nothing is laid out here: every key goes to `core.specplot.Specplot`, held to
IRAF's by files of cursor commands, which runs in a thread of its own and reads
the keys from this window as IRAF's specplot reads the cursor -- s and t take
their next points from the keys that follow. When specplot waits for a key the
window draws its last graph: each spectrum as its ptype says (lines, histogram or
marks), its label at its place, and on the status line what specplot printed. A
question (the spectrum a and i add, the coordinate of u and t) is asked on the
command line below the graph.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import specplot
from . import gtplot, keyguide, theme
from ._qt import on_main
from .splot_panel import END, _Answers, _Reads
from .theme import palette as _c

HINT = ":show   ·   press a key over the plot, or type a colon command here"


class SpecplotPanel(QtWidgets.QWidget):
    """The spectra and specplot's keys."""

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
        self.yscale = bool(params.get("yscale", False))
        self.session = specplot.Specplot(params, _Answers(self))

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
        # IRAF draws no scale on y unless yscale: the spectra are offset, its
        # numbers would mean nothing
        self.plot.getAxis("left").setStyle(showValues=self.yscale)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        # the labels sit where IRAF writes them, right of the graph at xlpos 1.02:
        # in the scene, outside the plot's clipping, moved when the plot is
        self._labels: list = []
        box = self.plot.getViewBox()
        box.sigRangeChanged.connect(self._place_labels)
        box.sigResized.connect(self._place_labels)
        layout.addWidget(self.plot, stretch=1)
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        layout.addWidget(keyguide.strip(specplot.HELP))
        self.command = QtWidgets.QLineEdit()
        self.command.setObjectName("command")
        self.command.setPlaceholderText(HINT)
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
            if self.error is not None or self.session.fatal:
                why = self.error or self.session.fatal.replace("\a", "")
                self.status.setText(f"specplot stopped: {why}")
            self.finished.emit()
            return
        if self._asking.is_set():
            prompt = self.session.out.rsplit("\n", 1)[-1].replace("\a", "").strip()
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
        new = session.out[self._shown_out:]
        self._shown_out = len(session.out)
        lines = [line for line in new.replace("\a", "").splitlines() if line.strip()]
        if lines:
            self.status.setText("\n".join(lines[-3:]))
        if not session.graphs:
            return
        frame = session.graphs[-1]
        erased = {item[1] for item in session.drawn if item[0] == "erase"}
        self.plot.clear()
        for item, _, _ in self._labels:
            self.plot.scene().removeItem(item)
        self._labels = []
        self.plot.setTitle("<br>".join(t for t in (frame["title"], frame["params"]) if t),
                           color=_c().ink, size="10pt")
        self.plot.setLabel("bottom", frame["xlabel"], color=_c().muted)
        ylabel = str(session.params.get("ylabel", ""))
        self.plot.setLabel("left", ylabel, color=_c().muted)
        colours = [_c().ink, _c().accent, _c().sky, _c().muted]
        for sp in frame["spectra"]:
            if sp["index"] in erased:
                continue
            colour = colours[(int(sp["color"]) - 1) % len(colours)]
            x, y = np.asarray(sp["x"], float), np.asarray(sp["y"], float)
            ptype = int(sp["ptype"]) if str(sp["ptype"]).lstrip("-").isdigit() else 1
            if ptype == 0:
                x, y = gtplot.steps(x, y)
            if ptype < 0:
                self.plot.plot(x, y, pen=None, symbol="+", symbolSize=5,
                               symbolPen=pg.mkPen(colour))
            else:
                style = {1: QtCore.Qt.PenStyle.SolidLine, 2: QtCore.Qt.PenStyle.DashLine,
                         3: QtCore.Qt.PenStyle.DotLine,
                         4: QtCore.Qt.PenStyle.DashDotLine}.get(
                             ptype, QtCore.Qt.PenStyle.SolidLine)
                self.plot.plot(x, y, pen=pg.mkPen(colour, width=1.2, style=style))
            if sp["label"].strip():
                self._label(sp["label"], *sp["at"])
        for item in session.drawn:
            if item[0] == "text":
                self._label(*item[1][2:], *item[1][:2])
        width = max((item.boundingRect().width() for item, _, _ in self._labels),
                    default=0.0)
        self.plot.getPlotItem().layout.setContentsMargins(0, 0, width + 30, 0)
        if rescale:
            gtplot.set_window(self.plot, session.window, (False, False))
        self._place_labels()

    def _label(self, label: str, x, y) -> None:
        text = pg.TextItem(str(label), color=_c().ink, anchor=(0, 0.5))
        self.plot.scene().addItem(text)
        self._labels.append((text, float(x), float(y)))

    def _place_labels(self, *_) -> None:
        box = self.plot.getViewBox()
        for item, x, y in self._labels:
            item.setPos(box.mapViewToScene(QtCore.QPointF(x, y)))

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
        """One keystroke over the graph: to specplot, unless it is a question's turn."""
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
        self.command.setPlaceholderText(HINT)
        self.plot.setFocus()
        if self._asking.is_set():
            self._idle.clear()
            self._answers.put(text)
            return
        text = text.lstrip(":")
        if text.startswith("."):
            self.status.setText(gtplot.cursor_mode(self.plot, text[1:],
                                                   "specplot.png").strip())
        elif text:
            self.send(":", text)

    def closeEvent(self, event) -> None:
        self._keys.put(END)
        self._answers.put(END)
        super().closeEvent(event)


@on_main
def run(params: dict, images: list[str], title: str = "specplot") -> SpecplotPanel:
    """Open the window; specplot runs until q, or the window closes."""
    from .dock import application, show

    application()   # before the panel: a widget made with no QApplication aborts Qt

    panel = SpecplotPanel(params, images)
    show(panel, title, size=(1000, 700))
    panel._keys.put(END)
    panel._answers.put(END)
    panel._thread.join(timeout=5)
    return panel
