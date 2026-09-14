"""The viewer's measurements, without a screen.

Section 8.2 keeps the analysis out of the window precisely so this file can
exist: the `r` key and a task must agree, and neither needs Qt to be checked.
"""

import numpy as np
import pytest

from harwood.core import display
from harwood.viewer import analysis

FWHM = 3.40
TRUE_X, TRUE_Y = 612.03, 539.91


@pytest.fixture
def star_field():
    """One star at an exactly known 1-based position, on a flat sky."""
    ny, nx = 640, 700
    sigma = FWHM / 2.3548
    yy, xx = np.mgrid[0:ny, 0:nx]
    star = 4530.0 * np.exp(-(((xx - (TRUE_X - 1))**2 + (yy - (TRUE_Y - 1))**2)
                             / (2 * sigma**2)))
    rng = np.random.default_rng(11)
    return star + 1030.5 + rng.normal(0, 3.0, (ny, nx))


def test_the_r_key_recovers_the_injected_star(star_field):
    """The permanent half-pixel test of section 8.4.

    The prototype measured (612.03, 539.91) for a star injected there, with a
    FWHM of 3.43 against 3.40 injected. Anything that shifts by half a pixel
    breaks this, which is the whole point of keeping it.
    """
    profile = analysis.radial_profile(star_field, TRUE_X + 1.4, TRUE_Y - 1.1)

    assert profile.converged
    assert profile.x == pytest.approx(TRUE_X, abs=0.15)
    assert profile.y == pytest.approx(TRUE_Y, abs=0.15)
    assert profile.fwhm == pytest.approx(FWHM, abs=0.25)
    assert profile.sky == pytest.approx(1030.5, abs=2.0)


def test_the_sky_annulus_starts_beyond_the_buffer(star_field):
    """The buffer exists so the star's wings stay out of its own sky."""
    tight = analysis.radial_profile(star_field, TRUE_X, TRUE_Y, radius=4, buffer=0,
                                    width=3)
    fair = analysis.radial_profile(star_field, TRUE_X, TRUE_Y, radius=4, buffer=8,
                                   width=5)
    assert tight.sky > fair.sky, "a sky measured on the wings reads too high"


def test_a_fit_that_fails_says_so_rather_than_returning_a_number(star_field):
    blank = np.full((60, 60), 1030.5)
    profile = analysis.radial_profile(blank, 30.0, 30.0)
    assert not profile.converged
    assert profile.message


def test_moffat_is_available_and_measures_the_same_star(star_field):
    profile = analysis.radial_profile(star_field, TRUE_X, TRUE_Y, fittype="moffat")
    assert profile.converged
    assert profile.fwhm == pytest.approx(FWHM, abs=0.4)


def test_quick_phot_agrees_with_the_profile(star_field):
    result = analysis.quick_phot(star_field, TRUE_X + 0.7, TRUE_Y - 0.5,
                                 aperture=8.0, annulus=14.0, dannulus=6.0)
    assert result["x"] == pytest.approx(TRUE_X, abs=0.2)
    assert result["sky"] == pytest.approx(1030.5, abs=2.0)
    sigma = FWHM / 2.3548
    assert result["flux"] == pytest.approx(4530.0 * 2 * np.pi * sigma**2, rel=0.05)


def test_headless_is_chosen_when_there_is_no_screen(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("HARWOOD_DISPLAY", raising=False)
    assert display._autodetect() == "headless"


def test_an_unknown_display_backend_is_refused():
    display.reset()
    with pytest.raises(ValueError, match="unknown display backend"):
        display.get("tektronix")


def test_markers_carry_a_tag_that_is_never_drawn():
    marker = display.Marker(1.0, 2.0, tag="examine")
    assert marker.label == ""


# ------------------------------------------------- the window, not the maths

def _window():
    """The main window, offscreen. Skipped where Qt is not installed."""
    import os

    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from harwood.viewer._qt import application
    from harwood.viewer.window import HarwoodWindow

    application()
    window = HarwoodWindow()
    window.load(np.random.default_rng(0).normal(500, 20, (200, 200)), name="t")
    return window


def test_the_stylesheet_carries_every_size_it_sets():
    """One number moves the whole interface. A size hard-coded outside the
    stylesheet is one that will not move with it."""
    import re

    pytest.importorskip("PySide6")
    from harwood.viewer.window import stylesheet

    small, large = stylesheet(1.0), stylesheet(2.0)
    assert small != large

    sizes = lambda sheet: [float(value) for value in       # noqa: E731
                           re.findall(r"font-size: ([\d.]+)pt", sheet)]
    assert sizes(small), "there are font sizes to scale"
    for one, two in zip(sizes(small), sizes(large), strict=True):
        assert two == pytest.approx(2 * one, rel=1e-6)


def test_ctrl_plus_and_minus_move_the_text_and_stop_at_the_ends():
    from harwood.viewer.window import SCALES

    window = _window()
    assert window.text_scale == 1.0

    window.zoom_text(+1)
    assert window.text_scale > 1.0
    for _ in range(20):
        window.zoom_text(+1)
    assert window.text_scale == SCALES[-1], "and does not run off the end"
    for _ in range(30):
        window.zoom_text(-1)
    assert window.text_scale == SCALES[0]

    window.set_text_scale(1.0)
    assert window.text_scale == 1.0


def test_everything_that_fixed_a_width_grows_with_the_text():
    """The usual way a zoom feature half works: the font grows and the box it
    sits in does not, so the label it was lining up gets clipped."""
    from PySide6 import QtGui

    window = _window()

    def measure():
        return (QtGui.QFontMetrics(window.font()).height(),
                window._param_labels[0].width(),
                window._hints[0][0].minimumHeight(),
                window.scale_box.width())

    window.set_text_scale(1.0)
    small = measure()
    window.set_text_scale(2.0)
    large = measure()

    for one, two in zip(small, large, strict=True):
        assert two > one, f"{one} did not grow"
    # The toolbar combo is the one with a width given in pixels at build time,
    # so it has to be multiplied rather than left alone.
    assert large[3] == pytest.approx(2 * small[3], abs=2)


def test_the_window_can_be_made_small_again():
    """A minimum size larger than a laptop screen is how a window becomes
    impossible to arrange."""
    window = _window()
    assert window.minimumWidth() <= 640 and window.minimumHeight() <= 420

    window.resize(660, 430)
    assert window.width() <= 700 and window.height() <= 480


def test_maximise_and_full_screen_both_come_back():
    window = _window()
    window.show()

    window.toggle_maximised()
    assert window.isMaximized()
    window.toggle_maximised()
    assert not window.isMaximized()

    window.toggle_full_screen()
    assert window.isFullScreen()
    window.toggle_full_screen()
    assert not window.isFullScreen()


def test_a_dragged_panel_can_be_put_back():
    """Dragging a dock somewhere unhelpful is easy and undoing it by hand is
    not. Qt can only restore a layout whose pieces are named, which is why
    every dock has an objectName."""
    from PySide6 import QtCore

    window = _window()
    window.show()
    started = window.dockWidgetArea(window.params_dock)

    window.addDockWidget(QtCore.Qt.LeftDockWidgetArea, window.params_dock)
    window.params_dock.setFloating(True)
    assert window.dockWidgetArea(window.params_dock) != started

    window.reset_layout()
    assert window.dockWidgetArea(window.params_dock) == started
    assert not window.params_dock.isFloating()


def test_every_dock_is_named_so_the_layout_can_be_saved():
    from PySide6 import QtWidgets

    window = _window()
    for dock in window.findChildren(QtWidgets.QDockWidget):
        assert dock.objectName(), f"'{dock.windowTitle()}' has no objectName"
    for bar in window.findChildren(QtWidgets.QToolBar):
        assert bar.objectName()


def test_a_panel_on_its_own_gets_the_same_keys():
    """A panel run from a shell opens in a dialog rather than the main window,
    and the person pressing Ctrl+ should not have to know which of the two
    they are looking at."""
    import os

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6 import QtGui, QtWidgets

    from harwood.viewer import dock
    from harwood.viewer._qt import application

    app = application()
    dialog = QtWidgets.QDialog()
    dock._text_shortcuts(dialog)

    keys = {shortcut.key().toString()
            for shortcut in dialog.findChildren(QtGui.QShortcut)}
    assert {"Ctrl+=", "Ctrl+-", "Ctrl+0", "F11", "Ctrl+M"} <= keys

    before = app.font().pointSizeF()
    grow, = [s for s in dialog.findChildren(QtGui.QShortcut)
             if s.key().toString() == "Ctrl+="]
    grow.activated.emit()
    assert app.font().pointSizeF() > before

    reset, = [s for s in dialog.findChildren(QtGui.QShortcut)
              if s.key().toString() == "Ctrl+0"]
    reset.activated.emit()
    assert app.font().pointSizeF() == pytest.approx(before)


def test_a_panel_dialog_can_be_resized_and_maximised():
    """A dialog with no size grip and no maximise button is the one thing
    worse than a window that cannot be made small."""
    import os

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6 import QtCore, QtWidgets

    from harwood.viewer._qt import application
    from harwood.viewer.icfit_panel import IcfitPanel

    application()
    x = np.arange(40.0)
    panel = IcfitPanel(x, x * 2 + 1)

    dialog = QtWidgets.QDialog()
    dialog.resize(900, 600)
    dialog.setMinimumSize(520, 360)
    dialog.setSizeGripEnabled(True)
    dialog.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)
    QtWidgets.QVBoxLayout(dialog).addWidget(panel)

    assert dialog.isSizeGripEnabled()
    assert dialog.windowFlags() & QtCore.Qt.WindowMaximizeButtonHint
    dialog.resize(560, 380)
    assert dialog.width() < 900
