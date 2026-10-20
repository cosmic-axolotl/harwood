"""The harwood viewer: one window, everything in it.

Built rather than borrowed. A generic FITS viewer would need an adapter layer
for every one of the things that make this one worth having: keys that call
harwood tasks, overlays that come *from* those tasks (apertures, tvmark points,
masks), coordinates that are 1-based because the rest of harwood is, and
interactions that end up in the provenance log.

Layout, from the design sketch in section 8.3:

    +-----------+----------------------------------+---------------------------+
    | Tasks     | frame  scale  stretch  colours   |  Examine                  |
    |  all >    |                                  |    radial profile + fit   |
    |  noao >   |            the image             |    centre, FWHM, peak     |
    |  cards,   |         apertures, marks         +---------------------------+
    |  or a     |                                  |  Parameters . rimexam     |
    |  form     |                                  |    each one explained     |
    +-----------+----------------------------------+---------------------------+
    | Command line   what ran, and what it said                                |
    | hw> imcombine @bias.lis Zero.fits && imstatistics Zero.fits  [History]   |
    +--------------------------------------------------------------------------+
    | x = 612.03  y = 539.91  (1-based)  value = 5561.3  a  d                   |
    +--------------------------------------------------------------------------+

Two ways in, by clicks (Tasks) and by typing (Command line), and one way
through: both send a command line to run_line, and both show up below.

Section 8.4's keyboard rule: **lower case belongs to the task, upper case and
`:` commands belong to the window** -- the same split as IRAF's cursor mode, so
the habit transfers.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import examine, logfile, pixels, render
from ..core.display import Marker
from . import theme
from .theme import palette as _c

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

#: The look -- colours, fonts, text size -- lives in theme.py: dark or light,
#: colour-blind safe or not, dyslexia friendly or not, all from one switch.
SCALES = theme.SCALES
COLORMAPS = render.COLORMAPS


def stylesheet(scale: float = 1.0) -> str:
    """The window's appearance at a given text scale (in the theme in force)."""
    current = theme.current()
    prefs = replace(current.preferences, scale=scale, autoscale=False)
    return replace(current, preferences=prefs).stylesheet()


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


def _wireframe(plot: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """The s key's surface seen from simexam's angles: every line and every
    column of the box, projected -- angh turns the box about its vertical
    axis, angv tilts the view down onto it."""
    z = np.asarray(plot["z"], dtype=float)
    ny, nx = z.shape
    span = float(np.ptp(z)) or 1.0
    height = (z - z.min()) / span * max(nx, ny) * 0.6
    h, v = np.radians(plot["angh"]), np.radians(plot["angv"])
    jj, ii = np.mgrid[0:ny, 0:nx].astype(float)
    ii -= (nx - 1) / 2
    jj -= (ny - 1) / 2
    across = ii * np.cos(h) - jj * np.sin(h)
    depth = ii * np.sin(h) + jj * np.cos(h)
    up = height * np.cos(v) + depth * np.sin(v)
    lines = [(across[j], up[j]) for j in range(ny)]
    lines += [(across[:, i], up[:, i]) for i in range(nx)]
    return lines


#: The scale-box entry for limits given by the display task (z1 and z2).
FIXED = "z1 z2"


class Frame:
    """One loaded image and how it is being shown."""

    def __init__(self, data: np.ndarray, header: Any = None, name: str = "",
                 window: tuple[float, float] | None = None, table: Any = None) -> None:
        self.data = np.asarray(data, dtype=float)
        self.header = header
        self.name = name or "image"
        #: z1 and z2 as the display task worked them out, shown under "z1 z2"
        self.window = window
        #: the lookup table of ztrans=user, for the "user" stretch
        self.table = table
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
        if scale == FIXED:
            return self.window or render.limits(self.data, "zscale")
        return render.limits(self.data, scale)

    def stretch(self, kind: str) -> tuple[str, Any]:
        """The transfer to draw with: "user" needs this frame's table."""
        if kind == "user" and self.table is None:
            return "linear", None
        return kind, self.table if kind == "user" else None


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
        self._param_labels: list[QtWidgets.QLabel] = []
        self._fixed_widths: list[tuple[QtWidgets.QWidget, int]] = []
        self._hints: list[tuple[QtWidgets.QLabel, str]] = []
        self._follow_system_scheme()
        theme.apply()
        theme.on_change(self._restyle)
        # Text that follows the window's size: recomputed a moment after the
        # last resize, so dragging a corner does not restyle on every pixel.
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(120)
        self._resize_timer.timeout.connect(self._follow_window_size)

        #: The command running beside the window, or "" -- one at a time.
        self._busy = ""
        self._beside_running = False
        from ._qt import accept_calls

        accept_calls()

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

        self._build_menus()
        self._build_toolbar()
        self._build_image()
        self._build_tasks()
        self._build_examine()
        self._build_params()
        self._build_work()
        self._build_files()
        self._build_docs()
        self._build_session()
        self._build_status()
        self._build_panels_menu()

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
                shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
                shortcut.activated.connect(slot)

        bind(("Ctrl++", "Ctrl+=", "Ctrl+Shift+=", "Ctrl+Shift++"),
             lambda: self.zoom_text(+1))
        bind(("Ctrl+-", "Ctrl+_"), lambda: self.zoom_text(-1))
        bind(("Ctrl+0",), lambda: self.set_text_scale(1.0))
        bind(("F11",), self.toggle_full_screen)
        bind(("Ctrl+M",), self.toggle_maximised)
        bind(("Ctrl+Shift+R",), self.reset_layout)

    @property
    def text_scale(self) -> float:
        """The person's own text size (Ctrl+/Ctrl-); the window's size, when
        autoscale is on, multiplies it."""
        return theme.current().preferences.scale

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
        """Resize every piece of text in the window together."""
        theme.update(scale=float(scale))
        self.log(f"text scale {self.text_scale:.2f}x"
                 + ("  (Ctrl+0 to reset)" if self.text_scale != 1.0 else ""))

    def _restyle(self) -> None:
        """The theme changed: colours, fonts or size. Everything that is not
        carried by the stylesheet is redone here -- plot colours, and any
        width fixed in pixels to line a column up, which has to move with
        the text or the text outgrows its box (the usual way a zoom feature
        half works)."""
        current = theme.current()
        self.setStyleSheet("")                    # the application's applies
        scale = current.scale
        metrics = QtGui.QFontMetrics(QtWidgets.QApplication.font())
        width = max(62, int(metrics.horizontalAdvance("fittype") * 1.7))
        for label in self._param_labels:
            label.setFixedWidth(width)
            label.setContentsMargins(0, 0, 0, 0)
        for widget, original in self._fixed_widths:
            widget.setFixedWidth(int(original * scale))
        for label, text in self._hints:
            _size_hint(label, text, width + 6)
        if hasattr(self, "params_dock"):
            self.params_dock.setMinimumWidth(int(220 * scale))
        if hasattr(self, "graphics"):
            self.graphics.setBackground(_c().ground)
        if hasattr(self, "plot"):
            theme.style_plot(self.plot)
            if self.measurements:
                last = self.measurements[-1][1]
                if last.plot is not None:
                    self._plot(last)
        if hasattr(self, "session"):
            _line_height(self.session, current.line_height)
        if self.frames:
            self._draw_overlay(self.frames[self.current])
        self._sync_menus()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_resize_timer"):
            self._resize_timer.start()

    def _follow_window_size(self) -> None:
        """autoscale: text grows and shrinks with the window, from 1.0 at the
        size it opens with, within 0.75x and 1.75x of the person's own."""
        if not theme.current().preferences.autoscale:
            return
        factor = min(self.width() / 1500, self.height() / 900)
        factor = round(max(0.75, min(1.75, factor)), 2)
        if abs(factor - theme.current().window_factor) >= 0.03:
            theme.update(window_factor=factor, save=False)

    def _follow_system_scheme(self) -> None:
        """theme=system: what the desktop says, now and when it changes."""
        hints = QtGui.QGuiApplication.styleHints()
        scheme = getattr(hints, "colorScheme", None)
        if scheme is None:
            return

        def read(*_):
            dark = hints.colorScheme() != QtCore.Qt.ColorScheme.Light
            if dark != theme.current().system_dark:
                theme.update(system_dark=dark, save=False)

        read()
        hints.colorSchemeChanged.connect(read)

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

    def _build_menus(self) -> None:
        """File (export) and View (the look). Every choice also has words and
        a check mark, never colour alone."""
        bar = self.menuBar()
        files = bar.addMenu("&File")
        for label, slot, shortcut in (
            ("Export image...", self.export_image, "Ctrl+E"),
            ("Export view (with marks)...", self.export_view, "Ctrl+Shift+E"),
            ("Export plot...", self.export_plot, "Ctrl+Alt+E"),
        ):
            action = files.addAction(label)
            action.setShortcut(QtGui.QKeySequence(shortcut))
            action.triggered.connect(slot)
        files.addSeparator()
        action = files.addAction("Open text files...")
        action.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        action.triggered.connect(self.choose_files)
        files.addSeparator()
        files.addAction("Close window", self.close)


        view = bar.addMenu("&View")
        self._theme_actions = {}
        group = QtGui.QActionGroup(self)
        for name, label in (("dark", "Dark"), ("light", "Light"),
                            ("system", "Follow the system")):
            action = view.addAction(label)
            action.setCheckable(True)
            action.triggered.connect(lambda _=False, n=name: self.set_look(theme=n))
            group.addAction(action)
            self._theme_actions[name] = action
        view.addSeparator()
        self._look_actions = {}
        for key, label, tip in (
            ("colorblind", "Colour-blind safe colours",
             "Okabe-Ito colours for data and warnings; colour is never the only sign"),
            ("dyslexia", "Dyslexia-friendly text",
             "Atkinson Hyperlegible, wider spacing, cream ground in the light theme"),
            ("autoscale", "Text follows the window size",
             "Text grows and shrinks with the window, on top of Ctrl+ and Ctrl-"),
            ("reduce_motion", "Reduce motion", "No animations; blink steps slower"),
        ):
            action = view.addAction(label)
            action.setCheckable(True)
            action.setToolTip(tip)
            action.triggered.connect(lambda checked, k=key: self.set_look(**{k: checked}))
            self._look_actions[key] = action
        fonts = view.addMenu("Dyslexia font")
        self._font_actions = {}
        group = QtGui.QActionGroup(self)
        for name, label in (("atkinson", "Atkinson Hyperlegible"),
                            ("opendyslexic", "OpenDyslexic")):
            action = fonts.addAction(label)
            action.setCheckable(True)
            action.triggered.connect(lambda _=False, n=name: self.set_look(dyslexia_font=n))
            group.addAction(action)
            self._font_actions[name] = action
        view.addSeparator()
        view.addAction("Larger text  (Ctrl+)", lambda: self.zoom_text(+1))
        view.addAction("Smaller text  (Ctrl-)", lambda: self.zoom_text(-1))
        view.addAction("Normal text  (Ctrl+0)", lambda: self.set_text_scale(1.0))
        view.addSeparator()
        view.addAction("Where these choices are kept", self._show_preferences_file)
        self._sync_menus()

    def _build_panels_menu(self) -> None:
        """Every panel by name, so a closed one can always be brought back,
        and the way to put them all where they started."""
        panels = self.menuBar().addMenu("&Panels")
        for dock in self.findChildren(QtWidgets.QDockWidget):
            action = dock.toggleViewAction()
            if dock is getattr(self, "tasks_dock", None):
                action.setText("Tasks (the package tree)")
                action.setShortcut(QtGui.QKeySequence("Ctrl+T"))
            if dock is getattr(self, "session_dock", None):
                action.setText("Command line (the hw> prompt)")
            if dock is getattr(self, "files_dock", None):
                action.setText("Files (the text editor)")
            if dock is getattr(self, "docs_dock", None):
                action.setText("Documentation (F1)")
            panels.addAction(action)
        panels.addAction("Go to the prompt", self.go_to_prompt, QtGui.QKeySequence("Ctrl+L"))
        panels.addSeparator()
        panels.addAction("Put the panels back  (Ctrl+Shift+R)", self.reset_layout)

        # Help last, where every application keeps it.
        helping = self.menuBar().addMenu("&Help")
        action = helping.addAction("Documentation")
        action.setShortcut(QtGui.QKeySequence("F1"))
        action.triggered.connect(lambda: self.show_docs("index.html"))
        helping.addAction("Documentation in the browser",
                          lambda: self.docs.open_in_browser())
        helping.addAction("The sample night (try the examples)",
                          lambda: self.show_docs("start/sample.html"))

    def _sync_menus(self) -> None:
        if not hasattr(self, "_theme_actions"):
            return
        prefs = theme.current().preferences
        for name, action in self._theme_actions.items():
            action.setChecked(prefs.theme == name)
        for key, action in self._look_actions.items():
            action.setChecked(bool(getattr(prefs, key)))
        for name, action in self._font_actions.items():
            action.setChecked(prefs.dyslexia_font == name)

    def set_look(self, **changes) -> None:
        """Change the look (theme, colorblind, dyslexia, ...) and say so."""
        theme.update(**changes)
        words = ", ".join(f"{k} {'on' if v is True else 'off' if v is False else v}"
                          for k, v in changes.items())
        self.log(f"view: {words}  (saved in {theme.Preferences.path()})")

    def _show_preferences_file(self) -> None:
        path = theme.Preferences.path()
        self.log(f"view choices are kept in {path}"
                 + ("" if path.exists() else " (written the first time one changes)"))

    # ------------------------------------------------------------ export

    def rendered(self, frame: Frame | None = None) -> np.ndarray:
        """The current frame as shown -- scale, stretch, colours -- rows top
        down: what the window draws and what `harwood impicture` writes."""
        frame = frame or self.frames[self.current]
        kind, table = frame.stretch(self.stretch_box.currentText())
        return render.rgb(frame.data, kind=kind, colormap=self.cmap_box.currentText(),
                          window=frame.limits(self.scale_box.currentText()), table=table)

    def _save_dialog(self, title: str, stem: str, formats: list[tuple[str, str]]):
        filters = ";;".join(f"{label} (*.{ext})" for label, ext in formats)
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, title, f"{stem}.{formats[0][1]}", filters)
        if not path:
            return None
        from pathlib import Path

        target = Path(path)
        if not target.suffix:
            ext = next((e for label, e in formats if chosen.startswith(label)),
                       formats[0][1])
            target = target.with_suffix(f".{ext}")
        return target

    IMAGE_FORMATS = [("PNG", "png"), ("JPEG", "jpg"), ("TIFF", "tif"), ("WebP", "webp")]

    def export_image(self, path=None) -> object:
        """The whole frame at its own resolution, as stretched -- no marks."""
        frame = self.frames.get(self.current)
        if frame is None:
            self.log("  nothing to export: load an image first (display obj.fits)")
            return None
        from pathlib import Path

        stem = Path(frame.name).stem.split("[")[0] or "image"
        target = Path(path) if path else self._save_dialog("Export image", stem,
                                                          self.IMAGE_FORMATS)
        if target is None:
            return None
        from .png import save_rgb

        saved = save_rgb(self.rendered(frame), target)
        scale = self.scale_box.currentText()
        kind, _ = frame.stretch(self.stretch_box.currentText())
        low, high = frame.limits(scale)
        limits = f"--z1 {low:g} --z2 {high:g}" if scale == FIXED else f"--scale {scale}"
        lutfile = " --lutfile <the display's lutfile>" if kind == "user" else ""
        self.log(f"  exported {frame.name} -> {saved}  ({scale}, "
                 f"{kind}, {self.cmap_box.currentText()})"
                 f"  -- the same as: harwood impicture '{frame.name}' {saved.name} "
                 f"{limits} --stretch {kind}{lutfile} "
                 f"--colormap {self.cmap_box.currentText()}")
        return saved

    def export_view(self, path=None) -> object:
        """What is on screen: the visible region, with apertures and marks."""
        from pathlib import Path

        target = Path(path) if path else self._save_dialog("Export view", "view",
                                                          self.IMAGE_FORMATS)
        if target is None:
            return None
        picture = self.graphics.grab()
        if not picture.save(str(target)):
            self.log(f"  could not write {target}: is the folder writable?")
            return None
        self.log(f"  exported the view -> {target}")
        return target

    PLOT_FORMATS = [("PNG", "png"), ("SVG", "svg"), ("PDF", "pdf"), ("CSV data", "csv")]

    def export_plot(self, path=None, widget=None) -> object:
        """The examine plot (or another), as a picture or as its numbers."""
        from pathlib import Path

        plot = widget or self.plot
        target = Path(path) if path else self._save_dialog("Export plot", "plot",
                                                          self.PLOT_FORMATS)
        if target is None:
            return None
        save_plot(plot, target)
        self.log(f"  exported the plot -> {target}")
        return target

    def _build_toolbar(self) -> None:
        bar = QtWidgets.QToolBar()
        bar.setObjectName("tools")
        bar.setMovable(False)
        self.addToolBar(bar)

        self.frame_box = self._combo(bar, "frame", [], self._frame_changed, width=230)
        self.scale_box = self._combo(bar, "scale", ["zscale", "minmax", "99.5%", FIXED],
                                     self._redraw)
        self.stretch_box = self._combo(bar, "stretch", list(render.STRETCHES),
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
        self.graphics.setBackground(_c().ground)
        self.view = self.graphics.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setMenuEnabled(False)
        self.image_item = pg.ImageItem()
        self.view.addItem(self.image_item)
        self.overlay = pg.ItemGroup()
        self.view.addItem(self.overlay)

        self.graphics.scene().sigMouseMoved.connect(self._mouse_moved)
        self.graphics.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.graphics.installEventFilter(self)
        self.setCentralWidget(self.graphics)

    def _build_tasks(self) -> None:
        """The click-only way in: IRAF's package tree and a form per task,
        on the left. Everything it runs goes through run_line, like the
        prompt, so the session shows it as if it had been typed."""
        from .browser import TaskBrowser

        self.browser = TaskBrowser(self.run_line, current_image=self._shown_file)
        self.browser.to_prompt.connect(self._to_prompt)
        self.browser.manual.connect(self._docs_for_task)
        self.tasks_dock = QtWidgets.QDockWidget("Tasks")
        self.tasks_dock.setObjectName("tasks")
        self.tasks_dock.setWidget(self.browser)
        self.tasks_dock.setMinimumWidth(280)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.tasks_dock)
        self.resizeDocks([self.tasks_dock], [360], QtCore.Qt.Orientation.Horizontal)

    def _shown_file(self) -> str:
        """The frame on display, when it came from a file that is still there."""
        from pathlib import Path

        frame = self.frames.get(self.current)
        if frame is None or not frame.name:
            return ""
        return frame.name if Path(frame.name.split("[", 1)[0]).is_file() else ""

    def _to_prompt(self, line: str) -> None:
        """A form's command, put at the prompt to be changed by hand."""
        if line.startswith("harwood "):
            line = line[len("harwood "):]
        self.prompt.setText(line)
        self.prompt.setFocus()

    def _build_examine(self) -> None:
        dock = QtWidgets.QDockWidget("Examine")
        dock.setObjectName("examine")
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(_c().ground)
        self.plot.showGrid(x=False, y=False)
        for axis in ("left", "bottom"):
            self.plot.getAxis(axis).setPen(pg.mkPen(_c().line))
            self.plot.getAxis(axis).setTextPen(pg.mkPen(_c().muted))
        self.plot.setTitle("nothing examined yet", color=_c().muted, size="10pt")
        layout.addWidget(self.plot, stretch=1)

        self.readout = QtWidgets.QLabel("press r over a star")
        self.readout.setObjectName("readout")
        self.readout.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.readout)

        dock.setWidget(body)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.examine_dock = dock

    def _build_params(self) -> None:
        """The parameter editor, section 5: it explains rather than just accepts."""
        dock = QtWidgets.QDockWidget("Parameters · rimexam")
        dock.setObjectName("parameters")
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
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
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.resizeDocks([self.examine_dock, dock], [420, 300], QtCore.Qt.Orientation.Vertical)

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
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.work_dock)

    def open_panel(self, panel, title: str) -> None:
        """Dock a panel into this window and bring it forward."""
        index = self.work.addTab(panel, title)
        self.work.setCurrentIndex(index)
        self.work_dock.setWindowTitle(title)
        self.work_dock.show()
        self.resizeDocks([self.work_dock], [460], QtCore.Qt.Orientation.Vertical)
        panel.setFocus()
        self.log(f"  {title} opened; press q there when you are done")

    def _build_files(self) -> None:
        """The text editor: lists, coordinates, tables, several in tabs. Beside
        the panels at the bottom, out of sight until a file is opened."""
        from .files import FileEditor

        self.files = FileEditor(open_shortcut=False)    # Ctrl+O is the File menu's
        self.files.said.connect(self.log)
        self.files_dock = QtWidgets.QDockWidget("Files")
        self.files_dock.setObjectName("files")
        self.files_dock.setWidget(self.files)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.files_dock)
        self.tabifyDockWidget(self.work_dock, self.files_dock)
        self.files_dock.hide()

    def _build_docs(self) -> None:
        """The documentation: the built site, beside the panels at the bottom,
        out of sight until asked for (F1, Help, `docs` at the prompt)."""
        from .docs import DocsPanel

        self.docs = DocsPanel()
        self.docs_dock = QtWidgets.QDockWidget("Documentation")
        self.docs_dock.setObjectName("docs")
        self.docs_dock.setWidget(self.docs)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.docs_dock)
        self.tabifyDockWidget(self.files_dock, self.docs_dock)
        self.docs_dock.hide()
        self._docs_placed = False

    def show_docs(self, url: str = "index.html") -> None:
        """A page of the documentation, in the window (or the browser, if this
        installation cannot show web pages)."""
        from .docs import available

        if available():
            if not self.docs_dock.isVisible() and not self._docs_placed:
                # The first time, a window of its own beside this one, big enough
                # to read: squeezed into a corner of the panels it is no use. It
                # is still a panel -- drag it back in, or onto another to tab it.
                self._docs_placed = True
                self.docs_dock.setFloating(True)
                frame = self.frameGeometry()
                self.docs_dock.resize(int(frame.width() * 0.62), int(frame.height() * 0.85))
                self.docs_dock.move(frame.x() + int(frame.width() * 0.34),
                                    frame.y() + int(frame.height() * 0.08))
            self.docs_dock.show()
            self.docs_dock.raise_()
        self.docs.show_page(url)

    def _docs_for_task(self, name: str) -> None:
        from ..docsite import url_for

        self.show_docs(url_for(name))

    def edit_files(self, paths: list[str]) -> None:
        """`edit a.coo b.coo` at the prompt, or File > Open text files."""
        self.files_dock.show()
        self.files_dock.raise_()
        self.resizeDocks([self.files_dock], [420], QtCore.Qt.Orientation.Vertical)
        if paths:
            self.files.open(paths)
        self.files.setFocus()

    def choose_files(self) -> None:
        self.edit_files([])
        self.files.choose()

    def close_panel(self, panel) -> None:
        index = self.work.indexOf(panel)
        if index >= 0:
            self.work.removeTab(index)
        panel.setParent(None)
        if self.work.count() == 0:
            self.work_dock.hide()

    def _build_session(self) -> None:
        dock = QtWidgets.QDockWidget("Command line")
        dock.setObjectName("session")
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.session = QtWidgets.QPlainTextEdit()
        self.session.setReadOnly(True)
        self.session.setMaximumBlockCount(4000)
        layout.addWidget(self.session, stretch=1)

        self.clock = QtWidgets.QWidget()
        clock_row = QtWidgets.QHBoxLayout(self.clock)
        clock_row.setContentsMargins(6, 3, 6, 3)
        self.clock_label = QtWidgets.QLabel("")
        self.clock_label.setObjectName("clock")
        self.clock_bar = QtWidgets.QProgressBar()
        self.clock_bar.setTextVisible(False)
        self.clock_bar.setMaximumWidth(220)
        self.clock_bar.setMaximumHeight(10)
        clock_row.addWidget(self.clock_label, stretch=1)
        clock_row.addWidget(self.clock_bar)
        self.clock.hide()
        layout.addWidget(self.clock)

        from .prompt import CommandLine

        self.prompt = CommandLine()
        self.prompt.setPlaceholderText("hw>  display obj.fits · imexamine · "
                                       "ccdproc --plan · help phot")
        self.prompt.returnPressed.connect(self._run_command)
        self.prompt.said.connect(lambda text: self.statusBar().showMessage(text, 4000))
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.prompt, stretch=1)
        recall = QtWidgets.QPushButton("🕘 History")
        recall.setObjectName("history")
        recall.setToolTip("Search what was typed before, here and in the terminal "
                          "(Ctrl+R at the prompt)")
        recall.clicked.connect(self._search_history)
        row.addWidget(recall)
        layout.addLayout(row)
        keys = QtWidgets.QLabel("Tab completes · ↑ ↓ walk the history, Ctrl+R searches "
                                "it · ; runs the next command anyway, && only if this "
                                "one worked")
        keys.setObjectName("hint")
        keys.setWordWrap(True)
        keys.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(keys)

        self.session_dock = dock
        dock.setWidget(body)
        dock.setMinimumHeight(170)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _build_status(self) -> None:
        self.status = QtWidgets.QLabel("")
        self.statusBar().addWidget(self.status)
        self.statusBar().setSizeGripEnabled(False)

    # ------------------------------------------------------------- frames

    def load(self, data, header=None, frame: int = 1, name: str = "",
             scale: str = "zscale", window: tuple[float, float] | None = None,
             stretch: str | None = None, table: Any = None) -> None:
        """Show an image in a frame. With `window`, the limits are those z1
        and z2 (the display task's), listed as "z1 z2" in the scale box."""
        self.frames[frame] = Frame(data, header, name, window=window, table=table)
        self.current = frame
        labels = [f"{n}  {f.name}" for n, f in sorted(self.frames.items())]
        self.frame_box.blockSignals(True)
        self.frame_box.clear()
        self.frame_box.addItems(labels)
        self.frame_box.setCurrentText(f"{frame}  {self.frames[frame].name}")
        self.frame_box.blockSignals(False)
        if window is not None:
            scale = FIXED
        for box, value in ((self.scale_box, scale), (self.stretch_box, stretch)):
            if value and value != box.currentText():
                box.blockSignals(True)
                box.setCurrentText(value)
                box.blockSignals(False)

        low, high = self.frames[frame].limits(scale)
        how = "" if window is not None else f" --{scale}"
        self.log(f"display '{name}' --frame {frame}{how}", prompt=True)
        self.log(f"  frame {frame} <- {name}   z1={low:g} z2={high:g}")
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
        # The pixels the export writes: render.rgb, turned back bottom up.
        shown = self.rendered(frame)[::-1]
        self.image_item.setImage(shown, autoLevels=False, levels=(0, 255))

        self._draw_overlay(frame)
        if reset_view:
            self.view.autoRange(padding=0.02)

    def _draw_overlay(self, frame: Frame) -> None:
        for item in list(self.overlay.childItems()):
            self.view.scene().removeItem(item)
        for marker in frame.markers:
            sx, sy = pixels.to_scene(marker.x, marker.y)
            # A theme token ("accent", "sky") follows the theme; anything
            # else is a colour a task chose and is drawn as given.
            colour = getattr(_c(), marker.colour, marker.colour)
            pen = pg.mkPen(colour, width=1.4)
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
                text = pg.TextItem(marker.label, color=colour, anchor=(0, 1))
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

        if key == ":":
            # imexamine's colon commands: typed at the prompt, answered there.
            self.prompt.setFocus()
            self.prompt.setText(":")
            return True
        if key.isupper():
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
        if self._beside_running:
            # The task's thread would be cut off halfway through writing.
            event.ignore()
            self.statusBar().showMessage(f"{self._busy} is still running; close the "
                                         "window when it has finished", 6000)
            return
        if not self.files.close_all():          # it asked, and was told to wait
            event.ignore()
            return
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

        result = examine.examine(examine.native(frame.data, frame.header), key, x, y,
                                 settings)
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
        self.plot.setAspectLocked(False)
        self.plot.setTitle(f"imexamine · {result.key} ({plot['title']})", color=_c().ink,
                           size="10pt")
        self.plot.setLabel("bottom", plot["xlabel"], color=_c().muted)
        self.plot.setLabel("left", plot["ylabel"], color=_c().muted)
        if plot["kind"] == "profile":
            self.plot.plot(plot["x"], plot["y"], pen=None, symbol="o", symbolSize=4.5,
                           symbolBrush=pg.mkBrush(_translucent(_c().sky, 200)),
                           symbolPen=pg.mkPen(None))
        elif plot["kind"] == "histogram":
            self.plot.plot(plot["x"], plot["y"], stepMode="center", fillLevel=0,
                           brush=pg.mkBrush(_translucent(_c().sky, 90)),
                           pen=pg.mkPen(_c().sky))
        elif plot["kind"] == "contour":
            z = plot["z"]
            for number, level in enumerate(plot["levels"]):
                curve = pg.IsocurveItem(data=z, level=level,
                                        pen=pg.mkPen(_c().sky if number % 2 else _c().accent))
                curve.setTransform(QtGui.QTransform.fromTranslate(
                    float(plot["x"][0]), float(plot["y"][0])))
                self.plot.addItem(curve)
            self.plot.setAspectLocked(True)
        elif plot["kind"] == "surface":
            for xs, ys in _wireframe(plot):
                self.plot.plot(xs, ys, pen=pg.mkPen(_c().sky, width=1))
        else:
            self.plot.plot(plot["x"], plot["y"], pen=pg.mkPen(_c().sky, width=1))
        if "fit_x" in plot:
            self.plot.plot(plot["fit_x"], plot["fit_y"], pen=pg.mkPen(_c().accent, width=2))

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
            Marker(x, y, radius=radius, colour="accent", tag="examine"),
            Marker(x, y, radius=radius + params["buffer"], colour="sky", tag="examine"),
            Marker(x, y, radius=radius + params["buffer"] + params["width"],
                   colour="sky", tag="examine"),
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
        """Enter at the prompt."""
        if self._busy:
            self.statusBar().showMessage(f"still running {self._busy}; the line "
                                         "stays here until it finishes", 5000)
            return
        line = self.prompt.take()
        if line:
            self.run_line(line)

    def go_to_prompt(self) -> None:
        """Ctrl+L: the command line, open and with the cursor in it."""
        self.session_dock.show()
        self.session_dock.raise_()
        self.prompt.setFocus()

    def _search_history(self) -> None:
        self.prompt.setFocus()
        self.prompt.search_history()

    def run_line(self, line: str) -> tuple[bool, str]:
        """Run a harwood command line in this process; (ok, what it printed).

        The same commands as the shell, because they are the same commands --
        the CLI is invoked directly rather than through a subprocess, so a task
        that opens a panel opens it *here*, and `display` loads into the frame
        you are looking at. The prompt and the task forms both come through
        here, and both show up in the session the same way.

        Several commands on one line (`;`, `&&`, `||`; cli/line.py) run one
        after the other, each shown as it starts, and a command skipped says
        why. ok is the last one's that ran, as in the shell.
        """
        line = line.strip()
        if self._busy:
            # Only reachable from the window while a command runs beside it: a
            # second Run, or Enter at the prompt. One at a time, said plainly.
            self.statusBar().showMessage(f"still running {self._busy}; this waits "
                                         "for it to finish", 5000)
            return False, f"still running {self._busy}\n"
        self._busy = line.split()[0] if line else "a command"
        try:
            return self._run_line(line)
        finally:
            self._busy = ""

    def _run_line(self, line: str) -> tuple[bool, str]:
        from ..cli import line as command_line

        try:
            steps = command_line.split(line)
        except ValueError as error:
            self.log(line, prompt=True)
            answer = f"{error} -- {error.fix}" if getattr(error, "fix", "") else str(error)
            self.log("  " + answer)
            return False, answer
        if len(steps) <= 1:
            return self._run_one(steps[0].command) if steps else (True, "")

        self.log(line, prompt=True)
        printed: list[str] = []

        def one(command: str) -> bool:
            if command in ("exit", "quit", "bye"):
                raise _Leave
            ok, text = self._run_one(command, echo="  ▸ ")
            printed.append(text)
            QtWidgets.QApplication.processEvents()   # show each as it finishes
            return ok

        def skipped(note: str) -> None:
            self.log("  ▹ " + note)
            printed.append(note + "\n")

        try:
            ok = command_line.run(line, one, skipped)
        except _Leave:
            self.close()
            return True, "".join(printed)
        return ok, "".join(printed)

    def _run_one(self, line: str, echo: str = "") -> tuple[bool, str]:
        """One command, shown in the session as typed (or after `echo`, when
        it is one of several on a line)."""
        import shlex

        line = line.strip()
        if line.startswith("harwood "):
            line = line[len("harwood "):].lstrip()
        if not line:
            return True, ""
        if echo:
            self.log(echo + line)
        else:
            self.log(line, prompt=True)

        if line in ("exit", "quit", "bye"):
            self.close()
            return True, ""

        if line.startswith(":"):
            try:
                answer = examine.colon(self.examine_settings, line)
            except ValueError as error:
                fix = getattr(error, "fix", "")
                answer = f"{error}" + (f" -- {fix}" if fix else "")
                self.log("  " + answer)
                return False, answer
            self.log("  " + answer)
            return True, answer

        try:
            arguments = shlex.split(line)
        except ValueError as error:
            self.log(f"  {error} -- unbalanced quotes?")
            return False, f"{error} -- unbalanced quotes?"

        stream = _Stream()
        if _on_window_thread(arguments):
            ok = _invoke(arguments, stream)
        else:
            ok = self._beside(arguments, stream)
        self._flush(stream, final=True)
        return ok, stream.everything()

    def _beside(self, arguments: list[str], stream: _Stream) -> bool:
        """Run a command on a thread of its own and keep the window alive
        meanwhile: drawing, showing what the task prints as it prints it, and
        a clock. The caller still gets the answer only when it is done -- the
        form and a chain of commands wait, exactly as before; the window does
        not."""
        import threading
        import time

        outcome: dict[str, bool] = {}
        thread = threading.Thread(
            target=lambda: outcome.update(ok=_invoke(arguments, stream)),
            name=f"harwood {arguments[0]}", daemon=True)
        loop = QtCore.QEventLoop()
        tick = QtCore.QTimer()
        tick.setInterval(200)
        started = time.monotonic()

        def check() -> None:
            self._flush(stream)
            self._show_clock(arguments, started)
            if not thread.is_alive():
                loop.quit()

        tick.timeout.connect(check)
        self._beside_running = True
        thread.start()
        tick.start()
        try:
            loop.exec()
        finally:
            tick.stop()
            thread.join()
            self._beside_running = False
            self.clock.hide()
        return outcome.get("ok", False)

    def _flush(self, stream: _Stream, final: bool = False) -> None:
        for each in stream.take(final):
            if each.strip():
                self.log(f"  {each}")

    def _show_clock(self, arguments: list[str], started: float) -> None:
        """⏳ ccdproc · 0:12 · about 0:28 left (from 5 earlier runs on this computer)"""
        import time

        from ..core import timing

        elapsed = time.monotonic() - started
        if elapsed < 0.6:              # a quick command never flashes a clock
            return
        running = timing.current
        name = running.task if running else arguments[0]
        expected = running.estimate if running else None
        bar = self.clock_bar
        if self.work.count() and self.work_dock.isVisible():
            # A panel is open and the task is waiting for a person: counting
            # that as the task being slow would be the clock telling a lie.
            panel = self.work.tabText(self.work.currentIndex())
            self.clock_label.setText(f"⏸ {name} is waiting for you in {panel}; "
                                     "press q there when you are done")
            bar.hide()
            self.clock.show()
            return
        still = theme.current().preferences.reduce_motion
        if expected and elapsed <= expected.seconds * 1.25 + 2:
            left = max(expected.seconds - elapsed, 0)
            text = (f"⏳ {name} · {timing.clock(elapsed)} · about {timing.clock(left)} "
                    f"left ({expected.basis()})")
            bar.setRange(0, 1000)
            bar.setValue(int(1000 * min(elapsed / max(expected.seconds, 1e-6), 0.99)))
            bar.show()
        else:
            if expected:
                text = (f"⏳ {name} · {timing.clock(elapsed)} · longer than the "
                        f"{timing.about(expected.seconds).removeprefix('about ')} "
                        "it usually takes here")
            elif running:
                text = (f"⏳ {name} · {timing.clock(elapsed)} · no estimate yet: "
                        f"{name} has not finished a run on this computer before")
            else:
                text = f"⏳ {name} · {timing.clock(elapsed)}"
            bar.setRange(0, 0)         # moving, because there is no fraction to show
            bar.setVisible(not still)
        self.clock_label.setText(text)
        self.clock.show()

    def eventFilter(self, obj, event) -> bool:
        """Keys over the image are the examine keys."""
        if event.type() == QtCore.QEvent.Type.KeyPress:
            return self._key(event.text())
        return super().eventFilter(obj, event)

    def log(self, text: str, prompt: bool = False) -> None:
        self.session.appendPlainText(("hw> " if prompt else "") + text)
        height = theme.current().line_height
        if height != 100.0:
            cursor = QtGui.QTextCursor(self.session.document().lastBlock())
            block = QtGui.QTextBlockFormat()
            block.setLineHeight(
                height, QtGui.QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
            cursor.mergeBlockFormat(block)
        self.session.verticalScrollBar().setValue(
            self.session.verticalScrollBar().maximum())


class _Leave(Exception):
    """`exit` among several commands on a line."""


class _Stream:
    """What a command prints, collected from whichever thread prints it and
    handed to the session a line at a time."""

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._pending = ""
        self._all: list[str] = []

    def write(self, text: str) -> int:
        with self._lock:
            self._pending += text
            self._all.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def take(self, final: bool = False) -> list[str]:
        """The complete lines so far; with `final`, the unfinished one too."""
        with self._lock:
            text, self._pending = self._pending, ""
            if not final and "\n" in text:
                text, self._pending = text.rsplit("\n", 1)
            elif not final:
                text, self._pending = "", text
        return text.splitlines()

    def everything(self) -> str:
        with self._lock:
            return "".join(self._all)


def _on_window_thread(arguments: list[str]) -> bool:
    """Whether a command has to run on the window's thread: one that works in
    the window itself, rather than doing its sums and reporting."""
    from ..core.registry import exists, get_class

    name = next((word for word in arguments if not word.startswith("-")), "")
    if name in ("gui", "edit", "docs"):
        return True
    return exists(name) and get_class(name).needs_window


def _invoke(arguments: list[str], stream: _Stream) -> bool:
    """One command through the CLI itself, its output into `stream`."""
    import contextlib
    import warnings

    import click

    from ..cli.main import app

    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), \
                warnings.catch_warnings():
            # A 1987 header writes its date as 01/09/87 and astropy says so
            # every time it is read. It is true and it is not actionable, and
            # a session log full of it is a session log nobody reads.
            warnings.filterwarnings("ignore", message=".*datfix.*")
            warnings.filterwarnings("ignore", message=".*FITSFixedWarning.*")
            app.main(args=arguments, prog_name="harwood", standalone_mode=False)
    except click.ClickException as error:
        stream.write(f"{error.format_message()}\n")
        return False
    except click.exceptions.Exit as error:
        return not error.exit_code
    except SystemExit as error:
        return error.code in (None, 0)
    except Exception as error:               # a failure must not end the session
        stream.write(f"{type(error).__name__}: {error}\n")
        return False
    return True


def _hint(text: str) -> QtWidgets.QLabel:
    """An explanation under a field -- section 5: the editor explains, it does
    not merely accept. Sized so it can never be clipped, because a truncated
    explanation is worse than none: it looks like the whole answer."""
    label = QtWidgets.QLabel(text)
    label.setObjectName("hint")
    label.setWordWrap(True)
    policy = QtWidgets.QSizePolicy.Policy
    label.setSizePolicy(policy.Preferred, policy.Minimum)
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


def _arcsec_per_pixel(frame: Frame) -> float | None:
    if frame.wcs is None:
        return None
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        return float(np.mean(proj_plane_pixel_scales(frame.wcs.celestial)) * 3600)
    except Exception:
        return None


def _line_height(edit, percent: float) -> None:
    """Space between the lines of a text pane (wider for the dyslexia mode)."""
    cursor = QtGui.QTextCursor(edit.document())
    cursor.select(QtGui.QTextCursor.SelectionType.Document)
    block = QtGui.QTextBlockFormat()
    proportional = QtGui.QTextBlockFormat.LineHeightTypes.ProportionalHeight.value
    block.setLineHeight(percent, proportional)
    cursor.mergeBlockFormat(block)


def _translucent(colour: str, alpha: int) -> QtGui.QColor:
    value = QtGui.QColor(colour)
    value.setAlpha(alpha)
    return value


def save_plot(plot, path) -> Any:
    """A pyqtgraph plot widget to PNG (at twice its size on screen), SVG or
    PDF (vectors, as drawn), or CSV (its numbers)."""
    from pathlib import Path

    import pyqtgraph.exporters as exporters

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    item = plot.getPlotItem() if hasattr(plot, "getPlotItem") else plot
    suffix = path.suffix.lower()
    if suffix == ".csv":
        exporters.CSVExporter(item).export(str(path))
    elif suffix in (".svg", ".pdf"):
        # The widget drawn onto Qt's own vector devices, at the screen's
        # resolution: at any other one the axes (in points) and the data (in
        # pixels) come apart. pyqtgraph's SVG exporter cannot parse the paths
        # of the Qt it runs on.
        size, dpi = plot.size(), plot.logicalDpiX()
        if suffix == ".svg":
            from PySide6 import QtSvg

            device = QtSvg.QSvgGenerator()
            device.setFileName(str(path))
            device.setSize(size)
            device.setViewBox(QtCore.QRect(QtCore.QPoint(0, 0), size))
            device.setResolution(dpi)
            device.setTitle(path.stem)
        else:
            device = QtGui.QPdfWriter(str(path))
            device.setResolution(dpi)
            inches = QtCore.QSizeF(size.width() / dpi, size.height() / dpi)
            device.setPageSize(QtGui.QPageSize(inches, QtGui.QPageSize.Unit.Inch))
            device.setPageMargins(QtCore.QMarginsF(0, 0, 0, 0))
        painter = QtGui.QPainter(device)
        plot.render(painter)
        painter.end()
    else:
        exporter = exporters.ImageExporter(item)
        exporter.parameters()["width"] = int(item.sceneBoundingRect().width() * 2)
        exporter.export(str(path))
    return path
