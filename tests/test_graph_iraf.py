"""imhistogram's plot and fitsky's histplot and radplot against IRAF 2.18.

tests/data/iraf_graphs holds what IRAF drew, from its graphics metacode: each plot's
window, its text, and its curves and marks. A log axis with data at or under zero is on
IRAF's elog scale, a straight line through zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import field

from harwood.core import apphot, fitsky, gio
from harwood.core.registry import get_task
from harwood.tasks import imutil

DATA = Path(__file__).parent / "data" / "iraf_graphs"
GKI_UNITS = 4

HISTOGRAMS = {
    "line": {"plot_type": "line"}, "box": {"plot_type": "box"},
    "log": {"plot_type": "line"}, "nolog": {"plot_type": "box", "logy": False},
    "cum": {"hist_type": "cumulative", "plot_type": "line"},
    "dif": {"hist_type": "difference", "plot_type": "line"},
    "sec": {"hist_type": "second_difference", "plot_type": "box"},
    "bins": {"z1": 100.0, "z2": 140.0, "nbins": 40, "plot_type": "box"},
    "binw": {"z1": 100.0, "z2": 140.0, "binwidth": 2.0, "plot_type": "line"},
    "closed": {"z1": 100.0, "z2": 140.0, "nbins": 8, "top_closed": True, "plot_type": "line"}}


def _on_screen(window, x, y, plot=None):
    wx1, wx2, wy1, wy2, sx1, sx2, sy1, sy2 = window
    y = np.asarray(y, dtype=float)
    with np.errstate(all="ignore"):
        if plot is not None and plot.elog:
            part = (gio.elog(y) - gio.elog(wy1)) / (gio.elog(wy2) - gio.elog(wy1))
        elif plot is not None and plot.logy:
            part = (np.log10(y) - np.log10(wy1)) / (np.log10(wy2) - np.log10(wy1))
        else:
            part = (y - wy1) / (wy2 - wy1)
    x = np.asarray(x, dtype=float)
    return np.c_[(sx1 + (x - wx1) / (wx2 - wx1) * (sx2 - sx1)) * 32767,
                 (sy1 + part * (sy2 - sy1)) * 32767]


def _held(window, mine, theirs, margin=0):
    """The points on the plot, where IRAF drew them: it clips the rest."""
    mine = mine[np.isfinite(mine).all(axis=1)]
    left, right = window[4] * 32767 + margin, window[5] * 32767 - margin
    bottom, top = window[6] * 32767 - 1, window[7] * 32767 + 1
    inside = mine[(mine[:, 0] >= left) & (mine[:, 0] <= right)
                  & (mine[:, 1] >= bottom) & (mine[:, 1] <= top)]
    for point in inside:
        assert np.abs(theirs - point).max(axis=1).min() <= GKI_UNITS, point
    return len(inside)


def _window(frame, window, tolerance=1e-4):
    wanted = frame["windows"][0]
    width, height = abs(wanted[1] - wanted[0]), abs(wanted[3] - wanted[2])
    assert window[:2] == pytest.approx(wanted[:2], abs=tolerance * width)
    assert window[2:] == pytest.approx(wanted[2:4], abs=tolerance * height)
    return wanted


@pytest.mark.parametrize("case", list(HISTOGRAMS))
def test_imhistogram_draws_what_iraf_draws(case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fits.PrimaryHDU(field(), fits.Header({"OBJECT": "a field"})).writeto("field.fits")
    drawn = []
    monkeypatch.setattr(imutil, "_plot", lambda plot, *args: drawn.append(plot))
    get_task("imhistogram").run(image="field.fits", device="stdgraph", **HISTOGRAMS[case])
    frame = json.loads((DATA / f"imh_{case}.json").read_text(encoding="utf-8"))[0]
    plot = drawn[0]
    window = _window(frame, plot.window)
    assert plot.title[0].removeprefix("harwood -- ") in frame["text"]
    assert plot.title[1] in frame["text"]
    lines = np.array([p for _, points in frame["lines"] for p in points])
    assert _held(window, _on_screen(window, *plot.curves[0], plot), lines) > 0


@pytest.mark.parametrize(("case", "algorithm", "smooth", "cursor"), [
    ("sky_h", "histplot", False, "hcur"), ("sky_hs", "histplot", True, "hcur"),
    ("sky_r", "radplot", False, "rcur")])
def test_fitsky_draws_what_iraf_draws(case, algorithm, smooth, cursor):
    drawn = []

    class Recording(fitsky.Cursor):
        def events(self, graph, start):
            drawn.append(graph)
            yield from super().events(graph, start)

    marks = Recording(DATA / cursor)
    marks.image = "field.fits"
    pars = apphot.SkyPars(algorithm, 10.0, 10.0, khist=3.0, binsize=0.1, smooth=smooth)
    for x, y in ((40.0, 46.0), (111.0, 52.0)):
        fitsky.fitsky(field(), x, y, pars, apphot.DataPars(), marks, None)
    frames = json.loads((DATA / f"{case}.json").read_text(encoding="utf-8"))[:2]
    assert len(drawn) == len(frames)
    for graph, frame in zip(drawn, frames, strict=True):
        window = _window(frame, graph.window)
        for text in (*graph.title, graph.xlabel, graph.ylabel):
            assert text in frame["text"], text
        lines = np.array([p for _, points in frame["lines"] for p in points])
        for x, y in graph.curves:
            assert _held(window, _on_screen(window, x, y), lines) > 0
        if graph.marks is not None:
            drawn_marks = np.array([m[1:3] for m in frame["marks"]])
            # A mark on the edge of the plot is cut up, and is not read as a mark; the
            # one highest is on the top of it.
            mine = _on_screen(window, *graph.marks)
            mine = mine[mine[:, 1] < window[7] * 32767 - 300]
            assert _held(window, mine, drawn_marks, 250) > 0


def test_elog_goes_straight_through_zero():
    assert gio.elog([0.0, 5.0, -5.0, 100.0, -1000.0]) == pytest.approx(
        [0.0, 0.5, -0.5, 2.0, -3.0])
    assert gio.decades(3.0, 250.0) == (1.0, 1000.0)
    assert gio.decades(0.0, 250.0) == (0.0, 1000.0)
    assert gio.decades(-250.0, 3.0) == (-1000.0, 10.0)


def test_the_panel_has_the_window_iraf_gives():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from harwood.core.graph import Graph
    from harwood.viewer import _qt
    from harwood.viewer.graph_panel import GraphPanel

    _qt.application()
    plot = Graph("a", logy=True).line([1.0, 2.0, 3.0], [0.0, 50.0, 900.0]).scale_to(1.0, 3.0)
    panel = GraphPanel(plot)
    assert plot.elog and plot.window == (1.0, 3.0, 0.0, 1000.0)
    (low, high) = panel.plot.getViewBox().viewRange()[1]
    assert low == pytest.approx(0.0) and high == pytest.approx(3.0)
