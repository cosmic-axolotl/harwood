"""display of a picture rather than an image -- the .eps IRAF's :.snap eps and
stdplot leave (the course's sgi*.eps) -- in a tab of its own."""

from __future__ import annotations

import os
import shutil

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from harwood.viewer import _qt

    return _qt.application()

EPS = """%!PS-Adobe-3.0 EPSF-3.0
%%BoundingBox: 0 0 200 100
newpath 10 10 moveto 190 90 lineto 4 setlinewidth stroke
showpage
"""


def test_display_knows_a_figure_from_an_image():
    from harwood.core.display import is_figure

    assert is_figure("sgi13360.eps") and is_figure("fig.PNG") and is_figure("a.pdf")
    assert not is_figure("ngc660.fits") and not is_figure("mask.pl")
    assert not is_figure("ngc660.fits[1:10,*]")


@pytest.mark.skipif(shutil.which("gs") is None, reason="needs Ghostscript")
def test_an_eps_opens_in_its_own_tab_and_zooms(qapp, tmp_path):
    from harwood.viewer import window as winmod
    from harwood.viewer.figure_panel import FigurePanel

    eps = tmp_path / "sgi1.eps"
    eps.write_text(EPS)
    from harwood.viewer import _qt

    win = winmod.HarwoodWindow()
    panel = win.show_figure(eps)
    assert isinstance(panel, FigurePanel)
    assert win.views.currentWidget() is panel
    assert win.views.tabText(win.views.currentIndex()) == "sgi1.eps"
    # cropped to the bounding box, at 200 dpi: 200 x 100 points
    assert abs(panel.image.width() - 556) < 6 and abs(panel.image.height() - 278) < 6
    panel.set_zoom(1.0)
    assert panel.scale() == 1.0
    panel.set_zoom(None)
    assert panel.zoom is None
    again = win.show_figure(eps)
    assert sum(isinstance(win.views.widget(i), FigurePanel)
               for i in range(win.views.count())) == 1, "the same file is the same tab"
    assert win.views.currentWidget() is again
    _qt.set_window(None)
    win.close()


def test_a_missing_or_broken_figure_says_what_to_do(tmp_path):
    from harwood.viewer.figure_panel import FigureError, render

    with pytest.raises(FigureError) as error:
        render(tmp_path / "nothing.eps")
    assert error.value.fix
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    with pytest.raises(FigureError):
        render(bad)


@pytest.mark.skipif(shutil.which("gs") is None, reason="needs Ghostscript")
def test_the_class_figure_draws(qapp):
    from pathlib import Path

    from harwood.viewer.figure_panel import render

    found = sorted(Path.home().glob("iraf/exercises/7.1/sgi*.eps"))
    if not found:
        pytest.skip("the course's figures are not on this machine")
    image = render(found[0])
    assert image.width() > 500 and image.height() > 300
