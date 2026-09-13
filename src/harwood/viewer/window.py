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

from ..core import pixels
from ..core.display import Marker
from . import analysis

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

STYLE = f"""
QMainWindow, QWidget {{ background: {GROUND}; color: {INK};
    font-family: "IBM Plex Sans", "DejaVu Sans", sans-serif; font-size: 12px; }}
QDockWidget {{ color: {MUTED}; titlebar-close-icon: none; }}
QDockWidget::title {{ background: {PANEL}; padding: 5px 9px;
    border-bottom: 1px solid {LINE}; font-weight: 600; }}
QToolBar {{ background: {PANEL}; border-bottom: 1px solid {LINE}; spacing: 7px;
    padding: 4px 8px; }}
QComboBox, QLineEdit {{ background: {GROUND}; color: {INK};
    border: 1px solid {LINE}; border-radius: 3px; padding: 3px 7px;
    selection-background-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 16px; }}
QLineEdit:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QToolButton {{ color: {INK}; padding: 3px 9px; border-radius: 3px; }}
QToolButton:hover {{ background: {LINE}; }}
QPlainTextEdit {{ background: {GROUND}; color: {INK}; border: none;
    font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace; font-size: 11px; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {LINE}; color: {INK};
    font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace; }}
QLabel#hint {{ color: {MUTED}; font-size: 11px; }}
QLabel#readout {{ font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace;
    font-size: 11px; color: {INK}; }}
QLabel#param {{ color: {MUTED}; font-weight: 600; }}
"""

COLORMAPS = {
    "grey": [(0, 0, 0), (255, 255, 255)],
    "heat": [(0, 0, 0), (178, 34, 34), (255, 170, 0), (255, 255, 224)],
    "cool": [(6, 12, 33), (32, 84, 140), (86, 176, 186), (238, 248, 230)],
    "sepia": [(12, 9, 6), (120, 78, 46), (214, 170, 116), (255, 246, 232)],
}

#: What each imexamine key does, shown in the session log on demand.
KEYS = {
    "r": "radial profile and fit at the cursor",
    "a": "aperture photometry at the cursor",
    "m": "statistics of a box around the cursor",
    "l": "plot the line through the cursor",
    "c": "plot the column through the cursor",
    "?": "list these keys",
    "q": "leave examine mode",
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

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("harwood")
        self.resize(1500, 900)
        self.setStyleSheet(STYLE)

        self.frames: dict[int, Frame] = {}
        self.current = 1
        self.examine_params = {
            "radius": 12.0, "buffer": 5.0, "width": 5.0, "fittype": "gaussian",
        }
        self._blink_timer = QtCore.QTimer(self)
        self._blink_timer.timeout.connect(self._blink_step)

        self._build_toolbar()
        self._build_image()
        self._build_examine()
        self._build_params()
        self._build_work()
        self._build_session()
        self._build_status()

        from ._qt import set_window

        set_window(self)

        self.log("harwood viewer. Hover for coordinates; press ? over the image "
                 "for the examine keys.")

    # ------------------------------------------------------------ building

    def _build_toolbar(self) -> None:
        bar = QtWidgets.QToolBar()
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
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(body)
        form.setContentsMargins(8, 6, 8, 8)
        form.setSpacing(2)

        self.param_widgets: dict[str, QtWidgets.QWidget] = {}
        fields = [
            ("radius", "12", "Radius in pixels used for the profile and the fit."),
            ("buffer", "5", "Gap between the profile radius and the sky annulus, so "
                            "the star's own wings stay out of its sky."),
            ("width", "5", "Width of the annulus used to estimate the sky."),
        ]
        for name, default, explanation in fields:
            row = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(name)
            label.setObjectName("param")
            label.setFixedWidth(62)
            edit = QtWidgets.QLineEdit(default)
            edit.setValidator(QtGui.QDoubleValidator(0.1, 500.0, 2))
            edit.editingFinished.connect(self._params_changed)
            row.addWidget(label)
            row.addWidget(edit)
            form.addLayout(row)
            form.addWidget(_hint(explanation))
            self.param_widgets[name] = edit

        row = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("fittype")
        label.setObjectName("param")
        label.setFixedWidth(62)
        box = QtWidgets.QComboBox()
        box.addItems(["gaussian", "moffat"])
        box.currentTextChanged.connect(lambda *_: self._params_changed())
        row.addWidget(label)
        row.addWidget(box)
        form.addLayout(row)
        form.addWidget(_hint("Profile fitted to measure the FWHM. Moffat has wings; "
                             "it fits real seeing better than a gaussian."))
        self.param_widgets["fittype"] = box
        form.addStretch(1)

        scroll.setWidget(body)
        dock.setWidget(scroll)
        dock.setMinimumWidth(300)
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
        if frame is None or position is None:
            return False

        x, y = position
        if key == "r":
            self._examine_radial(frame, x, y)
            return True
        if key == "a":
            self._examine_phot(frame, x, y)
            return True
        if key == "m":
            stats = analysis.statistics(frame.data, x, y)
            self.log(f"  [m] x={x:.2f} y={y:.2f}  " + "  ".join(
                f"{k}={v:.4g}" for k, v in stats.items()))
            return True
        if key in ("l", "c"):
            self._examine_cut(frame, x, y, key)
            return True
        return False

    # ------------------------------------------------------ examine keys

    def _examine_radial(self, frame: Frame, x: float, y: float) -> None:
        params = self._read_params()
        profile = analysis.radial_profile(
            frame.data, x, y, radius=params["radius"], buffer=params["buffer"],
            width=params["width"], fittype=params["fittype"],
        )
        self.plot.clear()
        self.plot.setTitle("imexamine · r (radial profile)", color=INK, size="10pt")
        self.plot.setLabel("bottom", "radius (pixels)", color=MUTED)
        self.plot.setLabel("left", "counts above sky", color=MUTED)
        self.plot.plot(profile.radii, profile.values - profile.sky, pen=None,
                       symbol="o", symbolSize=4.5,
                       symbolBrush=pg.mkBrush(130, 165, 200, 190),
                       symbolPen=pg.mkPen(None))
        if profile.converged:
            self.plot.plot(profile.fit_radii, profile.fit - profile.sky,
                           pen=pg.mkPen(ACCENT, width=2))

        if profile.converged:
            scale = _arcsec_per_pixel(frame)
            seeing = f"  ({profile.fwhm * scale:.2f}\")" if scale else ""
            self.readout.setText(
                f"centre   x= {profile.x:.2f}   y= {profile.y:.2f}\n"
                f"FWHM     {profile.fwhm:.2f} px{seeing}\n"
                f"peak     {profile.peak:.1f}     sky {profile.sky:.1f}"
                f"     fit: {profile.fittype}"
            )
            self.log(f"  [r] x={profile.x:.2f} y={profile.y:.2f}  "
                     f"FWHM={profile.fwhm:.2f} px  peak={profile.peak:.0f}  "
                     f"sky={profile.sky:.1f}")
            self._mark_star(frame, profile.x, profile.y, params)
        else:
            self.readout.setText(profile.message or "no fit")
            self.log(f"  [r] x={x:.2f} y={y:.2f}  {profile.message}")

    def _mark_star(self, frame: Frame, x: float, y: float, params: dict) -> None:
        frame.markers = [m for m in frame.markers if m.tag != "examine"]
        frame.markers += [
            Marker(x, y, radius=params["radius"] / 2, colour=ACCENT, tag="examine"),
            Marker(x, y, radius=params["radius"] + params["buffer"], colour=SKY,
                   tag="examine"),
            Marker(x, y, radius=params["radius"] + params["buffer"] + params["width"],
                   colour=SKY, tag="examine"),
        ]
        self._draw_overlay(frame)

    def _examine_phot(self, frame: Frame, x: float, y: float) -> None:
        params = self._read_params()
        result = analysis.quick_phot(
            frame.data, x, y, aperture=params["radius"] / 2,
            annulus=params["radius"] + params["buffer"],
            dannulus=params["width"],
        )
        self.readout.setText(
            f"centre   x= {result['x']:.2f}   y= {result['y']:.2f}\n"
            f"flux     {result['flux']:.1f}   over {result['area']:.0f} px\n"
            f"sky      {result['sky']:.2f}      mag {result['mag']:.3f}"
        )
        self.log(f"  [a] x={result['x']:.2f} y={result['y']:.2f}  "
                 f"flux={result['flux']:.1f}  sky={result['sky']:.2f}  "
                 f"mag={result['mag']:.3f}")

    def _examine_cut(self, frame: Frame, x: float, y: float, key: str) -> None:
        row, col = int(round(y)) - 1, int(round(x)) - 1
        if key == "l":
            values = frame.data[np.clip(row, 0, frame.data.shape[0] - 1), :]
            axis = np.arange(1, values.size + 1)
            title, label = f"imexamine · l (line {row + 1})", "x (pixels)"
        else:
            values = frame.data[:, np.clip(col, 0, frame.data.shape[1] - 1)]
            axis = np.arange(1, values.size + 1)
            title, label = f"imexamine · c (column {col + 1})", "y (pixels)"
        self.plot.clear()
        self.plot.setTitle(title, color=INK, size="10pt")
        self.plot.setLabel("bottom", label, color=MUTED)
        self.plot.setLabel("left", "counts", color=MUTED)
        self.plot.plot(axis, values, pen=pg.mkPen(SKY, width=1))
        self.readout.setText(f"{title}\nmin {values.min():.1f}   "
                             f"max {values.max():.1f}   median {np.median(values):.1f}")
        self.log(f"  [{key}] {title.split('·')[1].strip()}")

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
    label.setContentsMargins(68, 0, 0, 8)
    label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
    metrics = label.fontMetrics()
    lines = max(1, int(len(text) / 38) + 1)
    label.setMinimumHeight(lines * metrics.height() + 10)
    return label


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
