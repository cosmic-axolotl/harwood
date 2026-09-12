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
