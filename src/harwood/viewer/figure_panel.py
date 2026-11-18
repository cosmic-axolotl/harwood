"""A figure in a tab of its own: what `display` does with a picture instead of an
image -- the .eps IRAF's :.snap eps and stdplot leave behind (the course's
sgi*.eps), PostScript, PDF, SVG, PNG, JPEG...

Pictures are not data: no pixel values, no zscale, no frame. They are looked at,
zoomed and compared with what harwood draws, so the tab is a viewer and nothing
else -- fitted to the tab at first; + and - (or Ctrl and the wheel) zoom, 0 fits
again, 1 is the picture's own size. PostScript, EPS and PDF are drawn by
Ghostscript (`gs`, the program IRAF's own EPS go through on any desktop); the
others by Qt.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.display import FIGURES as SUFFIXES  # noqa: F401 -- said in one place
from ..core.display import is_figure  # noqa: F401

#: What Ghostscript draws; Qt reads the rest by itself.
GHOSTSCRIPT = (".eps", ".epsi", ".epsf", ".ps", ".pdf")
#: Ghostscript's resolution: sharp at twice the zoom of a page on the screen.
DPI = 200


def ghostscript() -> str | None:
    names = ("gswin64c", "gswin32c", "gs") if sys.platform == "win32" else ("gs",)
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


class FigureError(Exception):
    """A picture that cannot be drawn, with what to do about it."""

    def __init__(self, message: str, fix: str) -> None:
        super().__init__(message)
        self.fix = fix


def render(path: Path | str) -> QtGui.QImage:
    """The picture as an image: the first page, cropped to its bounding box."""
    path = Path(path)
    if not path.exists():
        raise FigureError(f"{path} does not exist", "check the name and the folder")
    suffix = path.suffix.lower()
    if suffix in GHOSTSCRIPT:
        return _ghostscript(path, suffix)
    if suffix == ".svg":
        from PySide6 import QtSvg

        renderer = QtSvg.QSvgRenderer(str(path))
        if not renderer.isValid():
            raise FigureError(f"{path.name} is not an SVG Qt can read",
                              "open it in a browser to see what is wrong with it")
        size = renderer.defaultSize() * 2
        image = QtGui.QImage(size, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor("white"))
        painter = QtGui.QPainter(image)
        renderer.render(painter)
        painter.end()
        return image
    image = QtGui.QImage(str(path))
    if image.isNull():
        raise FigureError(f"{path.name} could not be read as a picture",
                          "check that the file is whole and its suffix says what it is")
    return image


def _ghostscript(path: Path, suffix: str) -> QtGui.QImage:
    gs = ghostscript()
    if gs is None:
        raise FigureError(
            f"{path.name}: PostScript, EPS and PDF are drawn by Ghostscript, which is "
            "not installed",
            "install it (Linux: sudo apt install ghostscript; macOS: brew install "
            "ghostscript; Windows: ghostscript.com) and display the file again")
    with tempfile.TemporaryDirectory(prefix="hrwd-figure-") as where:
        out = Path(where) / "page.png"
        command = [gs, "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m",
                   f"-r{DPI}", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
                   "-dFirstPage=1", "-dLastPage=1", f"-sOutputFile={out}"]
        if suffix in (".eps", ".epsi", ".epsf"):
            command.append("-dEPSCrop")
        try:
            done = subprocess.run([*command, str(path)], capture_output=True, text=True,
                                  timeout=60)
        except subprocess.TimeoutExpired as error:
            raise FigureError(f"{path.name}: Ghostscript took over a minute",
                              "the file may be very large: open it in a PDF viewer") \
                from error
        image = QtGui.QImage(str(out)) if out.exists() else QtGui.QImage()
    if image.isNull():
        said = (done.stderr or done.stdout).strip().splitlines()
        raise FigureError(f"{path.name}: Ghostscript could not draw it"
                          + (f" ({said[-1]})" if said else ""),
                          "check that the file is whole PostScript (it should start "
                          "with %!PS)")
    return image


class FigurePanel(QtWidgets.QWidget):
    """The picture, fitted to the tab until it is zoomed."""

    finished = QtCore.Signal()
    keep_after_quit = True
    STEP = 1.25

    def __init__(self, path: Path | str, image: QtGui.QImage | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        from .lang import tr

        self.path = Path(path)
        self.image = image if image is not None else render(self.path)
        self.pixmap = QtGui.QPixmap.fromImage(self.image)
        self.zoom: float | None = None            # None: fitted to the tab
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.label)
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.scroll.viewport().installEventFilter(self)
        self.status = QtWidgets.QLabel()
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.hint = tr("+ and − zoom (or Ctrl and the wheel) · 0 fits it to the tab · "
                       "1 is its own size")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll, stretch=1)
        layout.addWidget(self.status)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._draw()

    # ---- what is shown
    def scale(self) -> float:
        if self.zoom is not None:
            return self.zoom
        area = self.scroll.viewport().size()
        w, h = max(self.pixmap.width(), 1), max(self.pixmap.height(), 1)
        return max(min((area.width() - 8) / w, (area.height() - 8) / h, 1.0), 0.02)

    def _draw(self) -> None:
        scale = self.scale()
        size = QtCore.QSize(max(1, round(self.pixmap.width() * scale)),
                            max(1, round(self.pixmap.height() * scale)))
        self.label.setPixmap(self.pixmap.scaled(
            size, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation))
        self.scroll.setWidgetResizable(self.zoom is None)
        if self.zoom is not None:
            self.label.resize(size)
        self.status.setText(f"{self.path.name}   {self.image.width()}×"
                            f"{self.image.height()}   {scale:.0%}      {self.hint}")

    def set_zoom(self, zoom: float | None) -> None:
        self.zoom = None if zoom is None else min(max(zoom, 0.02), 16.0)
        self._draw()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.zoom is None:
            self._draw()

    # ---- keys and the wheel
    def keyPressEvent(self, event) -> None:
        key, text = event.key(), event.text()
        if text in ("+", "=") or key == QtCore.Qt.Key.Key_Plus:
            self.set_zoom(self.scale() * self.STEP)
        elif text in ("-", "−") or key == QtCore.Qt.Key.Key_Minus:
            self.set_zoom(self.scale() / self.STEP)
        elif text == "0":
            self.set_zoom(None)
        elif text == "1":
            self.set_zoom(1.0)
        elif text == "q":
            self.finished.emit()            # its own window closes; a tab stays
        else:
            super().keyPressEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if (event.type() == QtCore.QEvent.Type.Wheel
                and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
            step = self.STEP if event.angleDelta().y() > 0 else 1 / self.STEP
            self.set_zoom(self.scale() * step)
            return True
        return False

    # ---- the window's save button
    def save_picture(self, parent=None) -> Path | None:
        """The picture as drawn, to PNG (or JPEG, TIFF...): an .eps to put in a
        report or a slide, where PostScript is not read."""
        from .lang import tr

        suggested = str(self.path.with_suffix(".png").name)
        name, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent, tr("Save the figure"), suggested,
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif)")
        if not name:
            return None
        if not self.image.save(name):
            return None
        return Path(name)
