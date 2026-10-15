"""lineid's plot, driven without a screen."""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _Aperture:
    """A calibrated spectrum with H-alpha in absorption."""

    def __init__(self):
        self.pixel = np.arange(1.0, 801.0)
        self.wavelength = 6400.0 + 0.4 * (self.pixel - 1)
        self.flux = 1000.0 - 600.0 * np.exp(-0.5 * ((self.wavelength - 6562.8) / 1.2) ** 2)
        self.title = "test star"


@pytest.fixture
def panel():
    from harwood.viewer._qt import application
    from harwood.viewer.species_panel import SpeciesPanel

    application()
    return SpeciesPanel(_Aperture())


def test_the_keys_switch_what_is_marked(panel):
    everything = len(panel._wanted())
    assert everything > 0
    panel.key("s")                        # stellar lines off
    assert len(panel._wanted()) < everything
    panel.key("s")
    panel.key("f")                        # only lines with a feature under them
    detected = panel._wanted()
    assert detected and all(mark.detected for mark in detected)
    assert any("H" in mark.line.name for mark in detected), "H-alpha is there"


def test_prominence_stays_between_one_and_three(panel):
    for _ in range(5):
        panel.key("+")
    assert panel.prominence == 3
    for _ in range(5):
        panel.key("-")
    assert panel.prominence == 1


def test_a_velocity_that_is_not_a_number_is_refused(panel):
    panel.key("v")
    panel.entry.setText("fast")
    panel._took_velocity()
    assert panel.velocity == 0.0 and "expected a number" in panel.status.text()
    panel.key("v")
    panel.entry.setText("-120")
    panel._took_velocity()
    assert panel.velocity == -120.0


def test_q_in_the_main_window_ends_the_panel_and_not_harwood(panel):
    """q used to close self.window() -- which, docked in harwood gui, is the
    whole application."""
    from PySide6 import QtCore, QtGui, QtWidgets

    main = QtWidgets.QMainWindow()
    main.setCentralWidget(panel)
    main.show()
    ended = []
    panel.finished.connect(lambda: ended.append(True))
    panel.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress,
                                        QtCore.Qt.Key.Key_Q,
                                        QtCore.Qt.KeyboardModifier.NoModifier, "q"))
    assert ended, "the panel said it was done"
    assert main.isVisible(), "and harwood's window is still open"
