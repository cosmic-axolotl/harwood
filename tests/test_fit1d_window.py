"""fit1d and background with --interactive from the window: the "Fit line ="
question is asked there (it was asked at a terminal the window does not have, so
the task went on as if answered with nothing, and no fit opened), and the icfit
window opens on the line given."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def window_open(monkeypatch):
    pytest.importorskip("PySide6")
    from harwood.viewer import _qt

    monkeypatch.setattr(_qt, "window", lambda: object())


def test_the_question_goes_to_the_window(window_open, monkeypatch):
    from harwood.tasks import imfit
    from harwood.viewer import ask

    said, given = [], iter(["275", ""])
    monkeypatch.setattr(ask, "line", lambda prompt, title="harwood":
                        (said.append(prompt), next(given))[1])
    assert imfit._answer([], "ngc.fits: Fit line =") == "275"
    assert imfit._answer([], "ngc.fits: Fit line =") is None, "empty is the end"
    assert said[0].startswith("ngc.fits: Fit line =") and "270 280" in said[0]
    assert imfit._answer(["12"], "x") == "12", "--lines still answers first"


def test_background_opens_the_fit_on_the_line_asked(window_open, monkeypatch, tmp_path):
    from astropy.io import fits

    import harwood
    from harwood.core import icfit
    from harwood.viewer import ask, icfit_panel

    monkeypatch.chdir(tmp_path)
    yy, xx = np.mgrid[0:40, 0:60]
    sky = 50 + 0.5 * xx
    fits.writeto("slit.fits", (sky + 300 * np.exp(-((xx - 30) / 4.0) ** 2)).astype("f4"))
    given = iter(["20", None])
    monkeypatch.setattr(ask, "line", lambda prompt, title="harwood": next(given))
    opened = []

    class Session:
        def __init__(self, spec):
            self.spec = spec
            self.fitted_wts = np.ones(60)

    def make(x, y, spec, title="", weights=None):
        opened.append(title)
        return Session(icfit.FitSpec(**{**spec.__dict__, "sample": "1:15 45:60"}))

    monkeypatch.setattr(icfit_panel, "make", make)
    harwood.run("background", input=["slit.fits"], output=["sub.fits"], order=2,
                interactive=True)
    assert opened and "Fit line = 20 - 20" in opened[0]
    out = fits.getdata("sub.fits")
    assert abs(float(np.median(out[:, 2:12]))) < 1.0, "the sky set in the fit is gone"
