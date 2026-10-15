"""The interactive identify, driven without a screen.

The one step of a spectroscopic night that needs a person (the 1992 recipe
says so), and until now the one with no test at all. These press its keys
and type its wavelengths as a person would, and check what it hands back --
which is what identify writes to the database and dispcor then trusts.
"""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from harwood.core import arc, icfit  # noqa: E402

#: A made-up arc: lines at these pixels, with wavelength = 6500 + 0.2 * pixel.
PIXELS = [80.0, 210.0, 395.0, 520.0, 700.0, 880.0]


def _wave(pixel):
    return 6500.0 + 0.2 * pixel


@pytest.fixture
def panel():
    from harwood.viewer._qt import application
    from harwood.viewer.identify_panel import IdentifyPanel

    application()
    x = np.arange(1.0, 1001.0)
    flux = 10 + sum(500 * np.exp(-0.5 * ((x - p) / 1.5) ** 2) for p in PIXELS)
    peaks = arc.find_peaks(flux, threshold=10.0)
    catalogue = [_wave(p) for p in PIXELS]
    return IdentifyPanel(flux, peaks, catalogue,
                         icfit.FitSpec(function="chebyshev", order=2))


def _mark(panel, pixel, wavelength):
    panel._cursor = pixel + 0.7          # a person points near, not on, the line
    panel.key("m")
    panel.command.setText(f"{wavelength}")
    panel._run_command()


def test_quitting_without_marking_hands_back_no_solution(panel):
    """It used to hand back the placeholder line the panel starts from, and
    identify wrote that to the database as the night's wavelength scale."""
    fit, pixel, wavelength = panel.result()
    assert fit.model is None
    assert pixel.size == wavelength.size == 0


def test_lines_marked_by_hand_give_the_dispersion(panel):
    for p in PIXELS[:4]:
        _mark(panel, p, _wave(p))
    fit, pixel, _ = panel.result()
    assert pixel.size == 4
    assert fit(np.array([600.0]))[0] == pytest.approx(_wave(600.0), abs=0.01)


def test_deleting_back_below_two_lines_forgets_the_old_fit(panel):
    for p in PIXELS[:3]:
        _mark(panel, p, _wave(p))
    assert panel.result()[0].model is not None
    for p in PIXELS[:2]:
        panel._cursor = p
        panel.key("d")
    fit, pixel, _ = panel.result()
    assert pixel.size == 1 and fit.model is None


def test_marking_a_line_again_corrects_it_instead_of_doubling_it(panel):
    for p in PIXELS[:3]:
        _mark(panel, p, _wave(p))
    _mark(panel, PIXELS[1], _wave(PIXELS[1]) + 50)       # a slip ...
    _mark(panel, PIXELS[1], _wave(PIXELS[1]))            # ... corrected
    _, pixel, wavelength = panel.result()
    assert pixel.size == 3
    assert wavelength[1] == pytest.approx(_wave(PIXELS[1]))


def test_l_places_the_rest_of_the_list_once_two_lines_fix_the_scale(panel):
    _mark(panel, PIXELS[0], _wave(PIXELS[0]))
    _mark(panel, PIXELS[-1], _wave(PIXELS[-1]))
    panel.key("l")
    _, pixel, _ = panel.result()
    assert pixel.size == len(PIXELS)


def test_a_colon_command_drops_a_line_waiting_for_its_wavelength(panel):
    panel._cursor = PIXELS[0]
    panel.key("m")
    panel.command.setText(":order 1")
    panel._run_command()
    assert panel.spec.order == 1
    panel.command.setText("6516")
    panel._run_command()
    assert panel.result()[1].size == 0, "the number did not attach to the old line"


def test_a_word_where_a_wavelength_belongs_is_refused(panel):
    panel._cursor = PIXELS[0]
    panel.key("m")
    panel.command.setText("H-alpha")
    panel._run_command()
    assert "not a wavelength" in panel.status.text()
    assert panel.result()[1].size == 0
