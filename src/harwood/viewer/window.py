"""The harwood viewer: one window, everything in it.

Built rather than borrowed. A generic FITS viewer would need an adapter layer
for every one of the things that make this one worth having: keys that call
harwood tasks, overlays that come *from* those tasks (apertures, tvmark points,
masks), coordinates that are 1-based because the rest of harwood is, and
interactions that end up in the provenance log.

Layout, from the design sketch in section 8.3:

    +----------------------------------+---------------------------+
    | frame  scale  stretch  colours   |  Examine                  |
    |                                  |    radial profile + fit   |
    |            the image             |    centre, FWHM, peak     |
    |         apertures, marks         +---------------------------+
    |                                  |  Parameters . rimexam     |
    |                                  |    each one explained     |
    +----------------------------------+---------------------------+
    | Session   hw> display 'obj.fits[SCI,1]' --frame 1 --zscale    |
    +--------------------------------------------------------------+
    | x = 612.03  y = 539.91  (1-based)  value = 5561.3  a  d       |
    +--------------------------------------------------------------+

Section 8.4's keyboard rule: **lower case belongs to the task, upper case and
`:` commands belong to the window** -- the same split as IRAF's cursor mode, so
the habit transfers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import examine, logfile, pixels
from ..core.display import Marker

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

#: Dark, low-saturation, and quiet: the image is the only thing that should
#: pull the eye. Sized to sit beside a terminal at night without glare.
INK = "#d7dee5"
MUTED = "#8b9aa8"
PANEL = "#1b222b"
GROUND = "#141a21"
LINE = "#2a3541"
ACCENT = "#e0a33e"
SKY = "#4fb59b"

#: Text sizes in points at scale 1.0, and the range Ctrl+/Ctrl- moves through.
#: Points rather than pixels: a pixel size ignores the display's own scaling,
#: which is how text ends up unreadable on a high-density screen and enormous
#: on a projector.
BASE = {"body": 9.0, "mono": 8.5, "small": 8.0}
SCALES = (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.3)


def stylesheet(scale: float = 1.0) -> str:
    """The window's appearance at a given text scale.

    Everything that sets a size does it through `BASE`, so one number moves
    the whole interface together. Anything that hard-codes a width in pixels
    -- a label column, a fixed field -- has to be scaled alongside it or the
    text outgrows its box, which is the usual way a zoom feature half works.
    """
    body = BASE["body"] * scale
    mono = BASE["mono"] * scale
    small = BASE["small"] * scale
    return f"""
QMainWindow, QWidget {{ background: {GROUND}; color: {INK};
    font-family: "IBM Plex Sans", "DejaVu Sans", sans-serif;
    font-size: {body:.1f}pt; }}
QDockWidget {{ color: {MUTED}; titlebar-close-icon: none;
    font-size: {body:.1f}pt; }}
QDockWidget::title {{ background: {PANEL}; padding: {5 * scale:.0f}px 9px;
    border-bottom: 1px solid {LINE}; font-weight: 600; }}
QToolBar {{ background: {PANEL}; border-bottom: 1px solid {LINE}; spacing: 7px;
    padding: 4px 8px; }}
QComboBox, QLineEdit {{ background: {GROUND}; color: {INK};
    border: 1px solid {LINE}; border-radius: 3px; padding: 3px 7px;
    font-size: {body:.1f}pt;
    selection-background-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: {16 * scale:.0f}px; }}
QLineEdit:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QToolButton {{ color: {INK}; padding: 3px 9px; border-radius: 3px;
    font-size: {body:.1f}pt; }}
QToolButton:hover {{ background: {LINE}; }}
QPlainTextEdit {{ background: {GROUND}; color: {INK}; border: none;
    font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace;
    font-size: {mono:.1f}pt; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {LINE}; color: {INK};
    font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace;
    font-size: {mono:.1f}pt; }}
QLabel#hint {{ color: {MUTED}; font-size: {small:.1f}pt; }}
QLabel#readout {{ font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace;
    font-size: {mono:.1f}pt; color: {INK}; }}
QLabel#param {{ color: {MUTED}; font-weight: 600; font-size: {body:.1f}pt; }}
QToolTip {{ background: {PANEL}; color: {INK}; border: 1px solid {LINE};
    padding: 4px; font-size: {small:.1f}pt; }}
"""


#: Kept for anything that wants the default appearance without asking.
STYLE = stylesheet()

COLORMAPS = {
    "grey": [(0, 0, 0), (255, 255, 255)],
    "heat": [(0, 0, 0), (178, 34, 34), (255, 170, 0), (255, 255, 224)],
    "cool": [(6, 12, 33), (32, 84, 140), (86, 176, 186), (238, 248, 230)],
    "sepia": [(12, 9, 6), (120, 78, 46), (214, 170, 116), (255, 246, 232)],
}

#: What each key does, shown in the session log on `?`. The measuring keys
#: come from `core.examine`, so the window cannot list a key it cannot answer.
KEYS = {
    **examine.KEYS,
    "n": "next frame",
    "p": "previous frame",
    "?": "list these keys",
    "q": "end imexamine, when it is running as a task",
    "Z": "reset the view to the whole image",
}


class Frame:
    """One loaded image and how it is being shown."""

    def __init__(self, data: np.ndarray, header: Any = None, name: str = "") -> None:
        self.data = np.asarray(data, dtype=float)
        self.header = header
        self.name = name or "image"
        self.markers: list[Marker] = []
        self.wcs = self._read_wcs()

    def _read_wcs(self):
        if self.header is None:
            return None
        try:
            from astropy.wcs import WCS

            wcs = WCS(self.header)
            return wcs if wcs.has_celestial else None
        except Exception:
            return None

    def limits(self, scale: str) -> tuple[float, float]:
        finite = self.data[np.isfinite(self.data)]
        if finite.size == 0:
            return 0.0, 1.0
        if scale == "minmax":
            return float(finite.min()), float(finite.max())
        if scale == "99.5%":
            return tuple(float(v) for v in np.percentile(finite, [0.25, 99.75]))
        from astropy.visualization import ZScaleInterval

        low, high = ZScaleInterval().get_limits(finite)
        return float(low), float(high)


class HarwoodWindow(QtWidgets.QMainWindow):
    """The one window."""

    #: q pressed, or the window closed: an imexamine waiting on it can return.
    examine_done = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("harwood")
        self.setWindowTitle("harwood")
        # A starting size that fits a laptop, and a floor low enough that the
        # window can actually be made small. A minimum larger than the screen
        # is how a window becomes impossible to arrange, which is most of what
        # is annoying about IRAF's.
        self.resize(1500, 900)
        self.setMinimumSize(640, 420)
        self.text_scale = 1.0
        self._param_labels: list[QtWidgets.QLabel] = []
        self._fixed_widths: list[tuple[QtWidgets.QWidget, int]] = []
        self._hints: list[tuple[QtWidgets.QLabel, str]] = []
        self.setStyleSheet(stylesheet(self.text_scale))

        self.frames: dict[int, Frame] = {}
        self.current = 1
        # IRAF's rimexam defaults; the panel shows these five.
        self.examine_params = {
            "radius": 5.0, "buffer": 5.0, "width": 5.0, "rplot": 8.0,
            "fittype": "moffat",
        }
        #: The rest of rimexam, set by the imexamine task; not in the panel.
        self.examine_settings = examine.Settings()
        #: Every measurement made in this window, as (image, Result).
        self.measurements: list[tuple[str, examine.Result]] = []
        self.examine_logfile = ""
        self._blink_timer = QtCore.QTimer(self)
        self._blink_timer.timeout.connect(self._blink_step)

        self._build_toolbar()
        self._build_image()
        self._build_examine()
        self._build_params()
        self._build_work()
        self._build_session()
        self._build_status()

        self._build_shortcuts()
        self._default_state = self.saveState()

        from ._qt import set_window

        set_window(self)

        self.log("harwood viewer. Hover for coordinates; press ? over the image "
                 "for the examine keys.")
        self.log("Ctrl+ and Ctrl- resize the text, Ctrl+0 resets it. "
                 "F11 full screen, Ctrl+M maximise, Ctrl+Shift+R puts the "
                 "panels back where they started.")

    # ------------------------------------------- size, both kinds

    def _build_shortcuts(self) -> None:
        """Text size, window size, and a way back from a mangled layout.

        Several bindings for one action on purpose: on many keyboards `+` is
        a shifted key, so Ctrl+= and Ctrl+Shift+= have to mean the same thing
        as Ctrl++ or the shortcut works for some people and not others.
        """
        def bind(sequences, slot):
            for sequence in sequences:
                shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
                shortcut.setContext(QtCore.Qt.ApplicationShortcut)
                shortcut.activated.connect(slot)

        bind(("Ctrl++", "Ctrl+=", "Ctrl+Shift+=", "Ctrl+Shift++"),
             lambda: self.zoom_text(+1))
        bind(("Ctrl+-", "Ctrl+_"), lambda: self.zoom_text(-1))
        bind(("Ctrl+0",), lambda: self.set_text_scale(1.0))
        bind(("F11",), self.toggle_full_screen)
        bind(("Ctrl+M",), self.toggle_maximised)
        bind(("Ctrl+Shift+R",), self.reset_layout)

    def zoom_text(self, direction: int) -> None:
        """One step up or down the scale, stopping at the ends."""
        current = min(range(len(SCALES)),
                      key=lambda index: abs(SCALES[index] - self.text_scale))
        index = max(0, min(len(SCALES) - 1, current + direction))
        if SCALES[index] != self.text_scale:
            self.set_text_scale(SCALES[index])
        else:
            self.log(f"text is already at its {'largest' if direction > 0 else 'smallest'}"
                     f" ({self.text_scale:.2f}x).")

    def set_text_scale(self, scale: float) -> None:
        """Resize every piece of text in the window together.

        The stylesheet carries the sizes, so most of this is one assignment.
        The exception is anything that fixed a width in pixels to line a
        column up: those have to move with the text or it outgrows its box --
        which is the usual way a zoom feature half works.
        """
        self.text_scale = float(scale)
        self.setStyleSheet(stylesheet(self.text_scale))

        metrics = QtGui.QFontMetrics(self.font())
        width = max(62, int(metrics.horizontalAdvance("fittype") * 1.7))
        for label in self._param_labels:
            label.setFixedWidth(width)
            label.setContentsMargins(0, 0, 0, 0)
        for widget, original in self._fixed_widths:
            widget.setFixedWidth(int(original * self.text_scale))
        for label, text in self._hints:
            _size_hint(label, text, width + 6)
        if hasattr(self, "params_dock"):
            self.params_dock.setMinimumWidth(int(220 * self.text_scale))

        self.log(f"text scale {self.text_scale:.2f}x"
                 + ("  (Ctrl+0 to reset)" if self.text_scale != 1.0 else ""))

    def toggle_full_screen(self) -> None:
        """F11. Leaves the window where it was, rather than maximised."""
        if self.isFullScreen():
            self.showNormal()
            self.log("left full screen.")
        else:
            self.showFullScreen()
            self.log("full screen -- F11 again to come back.")

    def toggle_maximised(self) -> None:
        """Ctrl+M, for when the window manager gives no button to click."""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def reset_layout(self) -> None:
        """Put the docks back where they started.

        Dragging a dock somewhere unhelpful is easy and undoing it by hand is
        not. Qt remembers the arrangement; this remembers the one it began
        with, so there is always a way back.
        """
        self.restoreState(self._default_state)
        self.log("panels back where they started.")

    # ------------------------------------------------------------ building

    def _build_toolbar(self) -> None:
        bar = QtWidgets.QToolBar()
        bar.setObjectName("tools")
        bar.setMovable(False)
        self.addToolBar(bar)

        self.frame_box = self._combo(bar, "frame", [], self._frame_changed, width=230)
        self.scale_box = self._combo(bar, "scale", ["zscale", "minmax", "99.5%"],
                                     self._redraw)
        self.stretch_box = self._combo(bar, "stretch", ["linear", "sqrt", "log", "asinh"],
                                       self._redraw)
        self.cmap_box = self._combo(bar, "colours", list(COLORMAPS), self._redraw)

        bar.addSeparator()
        for label, slot, tip in (
            ("blink", self._toggle_blink, "Alternate between loaded frames"),
            ("tile", self._tile, "Show every frame side by side"),
            ("header", self._show_header, "Show this frame's FITS header"),
            ("regions", self._clear_marks, "Clear the overlays on this frame"),
            ("A-", lambda: self.zoom_text(-1), "Smaller text  (Ctrl-)"),
            ("A+", lambda: self.zoom_text(+1), "Larger text  (Ctrl+)"),
            ("reset", self.reset_layout,
             "Put the panels back where they started  (Ctrl+Shift+R)"),
        ):
            action = QtGui.QAction(label, self)
            action.setToolTip(tip)
            action.triggered.connect(slot)
            bar.addAction(action)

    def _combo(self, bar, label, items, slot, width=110):
        bar.addWidget(QtWidgets.QLabel(f" {label} "))
        box = QtWidgets.QComboBox()
        box.addItems(items)
        box.setFixedWidth(width)
        # Remembered with the width it was given, so a text scale multiplies
        # the original rather than compounding on the last scale.
        self._fixed_widths.append((box, width))
        box.currentTextChanged.connect(lambda *_: slot())
        bar.addWidget(box)
        return box

    def _build_image(self) -> None:
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground(GROUND)
        self.view = self.graphics.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setMenuEnabled(False)
        self.image_item = pg.ImageItem()
        self.view.addItem(self.image_item)
        self.overlay = pg.ItemGroup()
        self.view.addItem(self.overlay)

        self.graphics.scene().sigMouseMoved.connect(self._mouse_moved)
        self.graphics.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.graphics.installEventFilter(self)
        self.setCentralWidget(self.graphics)

    def _build_examine(self) -> None:
        dock = QtWidgets.QDockWidget("Examine")
        dock.setObjectName("examine")
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(GROUND)
        self.plot.showGrid(x=False, y=False)
        for axis in ("left", "bottom"):
            self.plot.getAxis(axis).setPen(pg.mkPen(LINE))
            self.plot.getAxis(axis).setTextPen(pg.mkPen(MUTED))
        self.plot.setTitle("nothing examined yet", color=MUTED, size="10pt")
        layout.addWidget(self.plot, stretch=1)

        self.readout = QtWidgets.QLabel("press r over a star")
        self.readout.setObjectName("readout")
        self.readout.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.readout)

        dock.setWidget(body)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self.examine_dock = dock

    def _build_params(self) -> None:
        """The parameter editor, section 5: it explains rather than just accepts."""
        dock = QtWidgets.QDockWidget("Parameters · rimexam")
        dock.setObjectName("parameters")
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(body)
        form.setContentsMargins(8, 6, 8, 8)
        form.setSpacing(2)

        self.param_widgets: dict[str, QtWidgets.QWidget] = {}
        fields = [
            ("radius", "5", "Aperture radius: summed for the flux, used for "
                            "centring, and fitted for the FWHM. r and a widen "
                            "it to three times the direct FWHM, as IRAF does."),
            ("buffer", "5", "Gap between the aperture and the sky annulus, so "
                            "the star's own wings stay out of its sky."),
            ("width", "5", "Width of the annulus used to estimate the sky."),
            ("rplot", "8", "How far out the radial profile is drawn. Drawing "
                           "only; the measurement uses the radius."),
        ]
        for name, default, explanation in fields:
            row = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(name)
            label.setObjectName("param")
            label.setFixedWidth(62)
            self._param_labels.append(label)
            edit = QtWidgets.QLineEdit(default)
            edit.setValidator(QtGui.QDoubleValidator(0.1, 500.0, 2))
            edit.editingFinished.connect(self._params_changed)
            row.addWidget(label)
            row.addWidget(edit)
            form.addLayout(row)
            hint = _hint(explanation)
            self._hints.append((hint, explanation))
            form.addWidget(hint)
            self.param_widgets[name] = edit

        row = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("fittype")
        label.setObjectName("param")
        label.setFixedWidth(62)
        self._param_labels.append(label)
        box = QtWidgets.QComboBox()
        box.addItems(["gaussian", "moffat"])
        box.setCurrentText("moffat")
        box.currentTextChanged.connect(lambda *_: self._params_changed())
        row.addWidget(label)
        row.addWidget(box)
        form.addLayout(row)
        explanation = ("Profile fitted to measure the FWHM. Moffat has wings; "
                       "it fits real seeing better than a gaussian.")
        hint = _hint(explanation)
        self._hints.append((hint, explanation))
        form.addWidget(hint)
        self.param_widgets["fittype"] = box
        form.addStretch(1)

        scroll.setWidget(body)
        dock.setWidget(scroll)
        # A minimum, not a fixed width: 300 was wide enough that the image
        # could not be given the room on a small screen. The scroll area
        # handles the rest.
        dock.setMinimumWidth(220)
        self.params_dock = dock
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self.resizeDocks([self.examine_dock, dock], [420, 300], QtCore.Qt.Vertical)

    def _build_work(self) -> None:
        """Where an interactive panel goes when a task opens one.

        A tab rather than a floating dialog: the whole reason for one window is
        that a radial profile and the star it came from should be the same
        glance apart. Hidden until something needs it, so an empty strip does
        not sit there taking room.
        """
        self.work = QtWidgets.QTabWidget()
        self.work.setDocumentMode(True)
        self.work.setTabsClosable(False)
        self.work_dock = QtWidgets.QDockWidget("Work")
        self.work_dock.setObjectName("work")
        self.work_dock.setWidget(self.work)
        self.work_dock.hide()
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.work_dock)

    def open_panel(self, panel, title: str) -> None:
        """Dock a panel into this window and bring it forward."""
        index = self.work.addTab(panel, title)
        self.work.setCurrentIndex(index)
        self.work_dock.setWindowTitle(title)
        self.work_dock.show()
        self.resizeDocks([self.work_dock], [460], QtCore.Qt.Vertical)
        panel.setFocus()
        self.log(f"  {title} opened; press q there when you are done")

    def close_panel(self, panel) -> None:
        index = self.work.indexOf(panel)
        if index >= 0:
            self.work.removeTab(index)
        panel.setParent(None)
        if self.work.count() == 0:
            self.work_dock.hide()

    def _build_session(self) -> None:
        dock = QtWidgets.QDockWidget("Session")
        dock.setObjectName("session")
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.session = QtWidgets.QPlainTextEdit()
        self.session.setReadOnly(True)
        self.session.setMaximumBlockCount(4000)
        layout.addWidget(self.session, stretch=1)

        self.prompt = QtWidgets.QLineEdit()
        self.prompt.setPlaceholderText("hw>  display obj.fits · imexamine · "
                                       "ccdproc --plan · help phot")
        self.prompt.returnPressed.connect(self._run_command)
        layout.addWidget(self.prompt)
        self._history: list[str] = []
        self._history_at = 0
        self.prompt.installEventFilter(self)

        dock.setWidget(body)
        dock.setMinimumHeight(170)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

    def _build_status(self) -> None:
        self.status = QtWidgets.QLabel("")
        self.statusBar().addWidget(self.status)
        self.statusBar().setSizeGripEnabled(False)

    # ------------------------------------------------------------- frames

    def load(self, data, header=None, frame: int = 1, name: str = "",
             scale: str = "zscale") -> None:
        self.frames[frame] = Frame(data, header, name)
        self.current = frame
        labels = [f"{n}  {f.name}" for n, f in sorted(self.frames.items())]
        self.frame_box.blockSignals(True)
        self.frame_box.clear()
        self.frame_box.addItems(labels)
        self.frame_box.setCurrentText(f"{frame}  {self.frames[frame].name}")
        self.frame_box.blockSignals(False)
        if scale != self.scale_box.currentText():
            self.scale_box.blockSignals(True)
            self.scale_box.setCurrentText(scale)
            self.scale_box.blockSignals(False)

        low, high = self.frames[frame].limits(scale)
        self.log(f"display '{name}' --frame {frame} --{scale}", prompt=True)
        self.log(f"  frame {frame} <- {name}   z1={low:.1f} z2={high:.1f}")
        self._redraw(reset_view=True)

    def add_markers(self, markers: list[Marker], frame: int | None = None) -> None:
        target = self.frames.get(frame or self.current)
        if target is None:
            return
        target.markers.extend(markers)
        self._redraw()

    def select_frame(self, frame: int) -> None:
        """Show a loaded frame, as if it had been picked in the toolbar."""
        if frame not in self.frames:
            return
        self.current = frame
        self.frame_box.blockSignals(True)
        self.frame_box.setCurrentText(f"{frame}  {self.frames[frame].name}")
        self.frame_box.blockSignals(False)
        self._redraw(reset_view=True)

    def _step_frame(self, direction: int) -> None:
        numbers = sorted(self.frames)
        if len(numbers) < 2:
            self.log("  only one frame is loaded")
            return
        position = numbers.index(self.current) if self.current in numbers else 0
        self.select_frame(numbers[(position + direction) % len(numbers)])
        self.log(f"  frame {self.current}  {self.frames[self.current].name}")

    def _frame_changed(self) -> None:
        text = self.frame_box.currentText()
        if text:
            self.current = int(text.split()[0])
            self._redraw(reset_view=True)

    # ------------------------------------------------------------ drawing

    def _redraw(self, reset_view: bool = False) -> None:
        frame = self.frames.get(self.current)
        if frame is None:
            return
        low, high = frame.limits(self.scale_box.currentText())
        shown = _stretch(frame.data, low, high, self.stretch_box.currentText())

        self.image_item.setImage(shown, levels=(0.0, 1.0), autoLevels=False)
        stops = COLORMAPS[self.cmap_box.currentText()]
        colours = pg.ColorMap(np.linspace(0, 1, len(stops)), stops)
        self.image_item.setLookupTable(colours.getLookupTable(0.0, 1.0, 256))

        self._draw_overlay(frame)
        if reset_view:
            self.view.autoRange(padding=0.02)

    def _draw_overlay(self, frame: Frame) -> None:
        for item in list(self.overlay.childItems()):
            self.view.scene().removeItem(item)
        for marker in frame.markers:
            sx, sy = pixels.to_scene(marker.x, marker.y)
            pen = pg.mkPen(marker.colour, width=1.4)
            if marker.shape == "circle":
                item = QtWidgets.QGraphicsEllipseItem(
                    sx - marker.radius, sy - marker.radius,
                    2 * marker.radius, 2 * marker.radius)
            else:
                item = QtWidgets.QGraphicsRectItem(
                    sx - marker.radius, sy - marker.radius,
                    2 * marker.radius, 2 * marker.radius)
            item.setPen(pen)
            item.setParentItem(self.overlay)
            if marker.label:
                text = pg.TextItem(marker.label, color=marker.colour, anchor=(0, 1))
                text.setPos(sx + marker.radius, sy + marker.radius)
                text.setParentItem(self.overlay)

    # ------------------------------------------------------------- cursor

    def _mouse_moved(self, position) -> None:
        frame = self.frames.get(self.current)
        if frame is None or not self.view.sceneBoundingRect().contains(position):
            return
        point = self.view.mapSceneToView(position)
        x, y = pixels.from_scene(point.x(), point.y())
        row, col = int(round(y)) - 1, int(round(x)) - 1
        inside = 0 <= row < frame.data.shape[0] and 0 <= col < frame.data.shape[1]
        value = f"{frame.data[row, col]:.1f}" if inside else "-"

        sky = ""
        if frame.wcs is not None and inside:
            try:
                coord = frame.wcs.pixel_to_world(col, row)
                sky = (f"   a = {coord.ra.to_string(unit='hour', sep=':', precision=2)}"
                       f"   d = {coord.dec.to_string(sep=':', precision=1)}  [ICRS]")
            except Exception:
                sky = ""
        self.status.setText(
            f"  x = {x:8.2f}   y = {y:8.2f}   (1-based)    value = {value:>10}{sky}"
        )
        self._cursor = (x, y)

    def _key(self, key: str) -> bool:
        """Section 8.4: lower case is the task's, upper case is the window's."""
        frame = self.frames.get(self.current)
        position = getattr(self, "_cursor", None)
        if not key:
            return False

        if key.isupper() or key == ":":
            if key == "Z":
                self.view.autoRange(padding=0.02)
                self.log("  [Z] view reset")
                return True
            return False

        if key == "?":
            for name, what in KEYS.items():
                self.log(f"  [{name}] {what}")
            return True
        if key == "q":
            self.examine_done.emit()
            return True
        if key in ("n", "p"):
            self._step_frame(+1 if key == "n" else -1)
            return True
        if frame is None or position is None:
            return False
        if key in examine.KEYS:
            self.examine_at(key, *position)
            return True
        return False

    # ------------------------------------------------------ examine keys

    def configure_examine(self, settings: examine.Settings, logfile_name: str = "") -> None:
        """Take the imexamine task's parameters: the panel shows the four it
        has room for, and the rest are kept here."""
        self.examine_settings = settings
        self.examine_logfile = logfile_name
        for name in ("radius", "buffer", "width", "rplot"):
            self.param_widgets[name].setText(f"{getattr(settings, name):g}")
        self.param_widgets["fittype"].setCurrentText(settings.fittype)
        self._read_params()

    def wait_for_examine(self) -> None:
        """Keep the keys answering until q, or until the window is closed.

        A local event loop rather than the application's: inside `harwood gui`
        the application loop is already running, and the prompt has to come
        back when the task ends rather than when the window does.
        """
        loop = QtCore.QEventLoop()
        self.examine_done.connect(loop.quit)
        try:
            if self.isVisible():
                loop.exec()
        finally:
            self.examine_done.disconnect(loop.quit)

    def closeEvent(self, event) -> None:
        self.examine_done.emit()
        super().closeEvent(event)

    def examine_at(self, key: str, x: float, y: float) -> examine.Result | None:
        """Answer one key at one 1-based position on the current frame."""
        frame = self.frames.get(self.current)
        if frame is None:
            return None
        params = self._read_params()
        settings = self.examine_settings
        for name in ("radius", "buffer", "width", "rplot", "fittype"):
            setattr(settings, name, params[name])

        result = examine.examine(frame.data, key, x, y, settings)
        self.measurements.append((frame.name, result))
        for number, text in enumerate(result.line.splitlines()):
            self.log(("  " if number == 0 else "    ") + text)
        logfile.write(self.examine_logfile, "imexamine", frame.name,
                      result.line.splitlines())

        if result.plot is not None:
            self._plot(result)
        self._readout(frame, result)
        if result.ok and key in ("r", "a"):
            self._mark_star(frame, result.x, result.y, result.values["radius"], params)
        return result

    def _plot(self, result: examine.Result) -> None:
        plot = result.plot
        self.plot.clear()
        self.plot.setTitle(f"imexamine · {result.key} ({plot['title']})", color=INK,
                           size="10pt")
        self.plot.setLabel("bottom", plot["xlabel"], color=MUTED)
        self.plot.setLabel("left", plot["ylabel"], color=MUTED)
        if plot["kind"] == "profile":
            self.plot.plot(plot["x"], plot["y"], pen=None, symbol="o", symbolSize=4.5,
                           symbolBrush=pg.mkBrush(130, 165, 200, 190),
                           symbolPen=pg.mkPen(None))
        elif plot["kind"] == "histogram":
            self.plot.plot(plot["x"], plot["y"], stepMode="center", fillLevel=0,
                           brush=pg.mkBrush(79, 181, 155, 90), pen=pg.mkPen(SKY))
        else:
            self.plot.plot(plot["x"], plot["y"], pen=pg.mkPen(SKY, width=1))
        if "fit_x" in plot:
            self.plot.plot(plot["fit_x"], plot["fit_y"], pen=pg.mkPen(ACCENT, width=2))

    def _readout(self, frame: Frame, result: examine.Result) -> None:
        v = result.values
        if not result.ok:
            self.readout.setText(result.line.split("  ", 1)[-1])
        elif result.key in ("r", "a"):
            scale = _arcsec_per_pixel(frame)
            seeing = f"  ({v['fwhm'] * scale:.2f}\")" if scale else ""
            fit = self.examine_settings.fittype
            beta = f"   beta {v['beta']:.2f}" if fit == "moffat" else ""
            self.readout.setText(
                f"centre   x= {v['x']:.2f}   y= {v['y']:.2f}   r= {v['radius']:.2f}\n"
                f"FWHM     {v['fwhm']:.2f} px{seeing}   {fit}{beta}\n"
                f"enclosed {v['enclosed']:.2f}   direct {v['direct']:.2f} px\n"
                f"e        {v['e']:.2f}   pa "
                + ("INDEF" if v['pa'] != v['pa'] else f"{v['pa']:.0f}") + "\n"
                f"mag      {v['mag']:.3f}   flux {v['flux']:.1f}\n"
                f"peak     {v['peak']:.1f}   sky {v['sky']:.2f}")
        else:
            self.readout.setText(result.line.split("  ", 1)[-1].replace("  ", "\n", 2))

    def _mark_star(self, frame: Frame, x: float, y: float, radius: float,
                   params: dict) -> None:
        """The aperture actually used -- after the a key widened it -- and the
        sky annulus beyond it."""
        frame.markers = [m for m in frame.markers if m.tag != "examine"]
        frame.markers += [
            Marker(x, y, radius=radius, colour=ACCENT, tag="examine"),
            Marker(x, y, radius=radius + params["buffer"], colour=SKY, tag="examine"),
            Marker(x, y, radius=radius + params["buffer"] + params["width"],
                   colour=SKY, tag="examine"),
        ]
        self._draw_overlay(frame)

    # ------------------------------------------------------------ toolbar

    def _toggle_blink(self) -> None:
        if self._blink_timer.isActive():
            self._blink_timer.stop()
            self.log("  blink off")
        elif len(self.frames) > 1:
            self._blink_timer.start(700)
            self.log(f"  blink on, {len(self.frames)} frames")
        else:
            self.log("  blink needs more than one frame")

    def _blink_step(self) -> None:
        numbers = sorted(self.frames)
        self.current = numbers[(numbers.index(self.current) + 1) % len(numbers)]
        self.frame_box.blockSignals(True)
        self.frame_box.setCurrentText(f"{self.current}  {self.frames[self.current].name}")
        self.frame_box.blockSignals(False)
        self._redraw()

    def _tile(self) -> None:
        self.log("  tile is not built yet")

    def _show_header(self) -> None:
        frame = self.frames.get(self.current)
        if frame is None or frame.header is None:
            self.log("  no header on this frame")
            return
        self.log(f"  header of {frame.name}:")
        for card in repr(frame.header).splitlines():
            if card.strip():
                self.log(f"    {card}")

    def _clear_marks(self) -> None:
        frame = self.frames.get(self.current)
        if frame is not None:
            frame.markers.clear()
            self._draw_overlay(frame)
            self.log("  overlays cleared")

    # ------------------------------------------------------------ helpers

    def _read_params(self) -> dict:
        values = {}
        for name, widget in self.param_widgets.items():
            if isinstance(widget, QtWidgets.QComboBox):
                values[name] = widget.currentText()
            else:
                try:
                    values[name] = float(widget.text())
                except ValueError:
                    values[name] = self.examine_params[name]
        self.examine_params.update(values)
        return values

    def _params_changed(self) -> None:
        self._read_params()

    def _run_command(self) -> None:
        """Run a harwood command typed at the prompt, in this process.

        The same commands as the shell, because they are the same commands --
        the CLI is invoked directly rather than through a subprocess, so a task
        that opens a panel opens it *here*, and `display` loads into the frame
        you are looking at.
        """
        import shlex

        line = self.prompt.text().strip()
        self.prompt.clear()
        if not line:
            return
        self._history.append(line)
        self._history_at = len(self._history)
        self.log(line, prompt=True)

        if line in ("exit", "quit", "bye"):
            self.close()
            return

        try:
            arguments = shlex.split(line)
        except ValueError as error:
            self.log(f"  {error} -- unbalanced quotes?")
            return

        import contextlib
        import io
        import warnings

        import click

        from ..cli.main import app

        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured), \
                    contextlib.redirect_stderr(captured), \
                    warnings.catch_warnings():
                # A 1987 header writes its date as 01/09/87 and astropy says so
                # every time it is read. It is true and it is not actionable, and
                # a session log full of it is a session log nobody reads.
                warnings.filterwarnings("ignore", message=".*datfix.*")
                warnings.filterwarnings("ignore", message=".*FITSFixedWarning.*")
                app.main(args=arguments, prog_name="harwood", standalone_mode=False)
        except click.ClickException as error:
            captured.write(f"{error.format_message()}\n")
        except (click.exceptions.Exit, SystemExit):
            pass
        except Exception as error:               # a failure must not end the session
            captured.write(f"{type(error).__name__}: {error}\n")

        for line in captured.getvalue().splitlines():
            if line.strip():
                self.log(f"  {line}")

    def eventFilter(self, obj, event) -> bool:
        """Up and down walk the history at the prompt; keys over the image are
        the examine keys."""
        if obj is getattr(self, "prompt", None) and \
                event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Up and self._history:
                self._history_at = max(0, self._history_at - 1)
                self.prompt.setText(self._history[self._history_at])
                return True
            if event.key() == QtCore.Qt.Key_Down and self._history:
                self._history_at = min(len(self._history), self._history_at + 1)
                self.prompt.setText(self._history[self._history_at]
                                    if self._history_at < len(self._history) else "")
                return True
            return False
        if event.type() == QtCore.QEvent.KeyPress:
            return self._key(event.text())
        return super().eventFilter(obj, event)

    def log(self, text: str, prompt: bool = False) -> None:
        self.session.appendPlainText(("hw> " if prompt else "") + text)
        self.session.verticalScrollBar().setValue(
            self.session.verticalScrollBar().maximum())


def _hint(text: str) -> QtWidgets.QLabel:
    """An explanation under a field -- section 5: the editor explains, it does
    not merely accept. Sized so it can never be clipped, because a truncated
    explanation is worse than none: it looks like the whole answer."""
    label = QtWidgets.QLabel(text)
    label.setObjectName("hint")
    label.setWordWrap(True)
    label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
    _size_hint(label, text, 68)
    return label


def _size_hint(label, text: str, indent: int) -> None:
    """Give a wrapped explanation room for the height it will actually take.

    Recomputed whenever the text scale changes: a height worked out at one
    font size clips the same words at a larger one, and a truncated
    explanation is worse than none because it looks like the whole answer.
    """
    from PySide6 import QtWidgets as _widgets

    label.setContentsMargins(indent, 0, 0, 8)
    available = max(120, label.width() - indent) if label.width() else 240
    metrics = label.fontMetrics()
    rectangle = metrics.boundingRect(
        0, 0, available, 10000,
        int(_widgets.QLabel().alignment()) | 0x1000,      # TextWordWrap
        text)
    label.setMinimumHeight(rectangle.height() + 10)


def _stretch(data: np.ndarray, low: float, high: float, kind: str) -> np.ndarray:
    """Normalise to 0..1 through the chosen transfer function."""
    span = max(high - low, 1e-12)
    norm = np.clip((data - low) / span, 0.0, 1.0)
    if kind == "sqrt":
        return np.sqrt(norm)
    if kind == "log":
        return np.log10(1 + 1000 * norm) / np.log10(1001)
    if kind == "asinh":
        return np.arcsinh(10 * norm) / np.arcsinh(10)
    return norm


def _arcsec_per_pixel(frame: Frame) -> float | None:
    if frame.wcs is None:
        return None
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        return float(np.mean(proj_plane_pixel_scales(frame.wcs.celestial)) * 3600)
    except Exception:
        return None
