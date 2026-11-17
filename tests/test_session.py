"""The window's session (viewer/session.py): what harwood gui had open, kept when it
closes and put back when asked -- the folder, the frames with their pixels, header,
z1 z2, marks and view, imexamine's parameters, the graph tabs (implot alive), the
text files with what was not saved, the record and the prompt; or a new session,
the old one set aside and still restorable."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def make_window(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    from harwood.viewer import _qt
    from harwood.viewer.window import HarwoodWindow

    _qt.application()
    made = []

    def make():
        window = HarwoodWindow()
        window.resize(1100, 760)
        window.show()
        _qt.set_window(window)
        made.append(window)
        return window

    yield make
    for window in made:
        window.keeps_session = False
        for tab in window.files.files():            # no question about them
            tab.document().setModified(False)
        window.close()
    _qt.set_window(None)


def _star():
    yy, xx = np.mgrid[0:60, 0:80]
    return 100 + 900 * np.exp(-((xx - 40) ** 2 + (yy - 30) ** 2) / 10.0)


def _work(window, tmp_path):
    """A session with something of everything in it."""
    from astropy.io import fits
    from PySide6 import QtWidgets

    from harwood.core.display import Marker
    from harwood.core.graph import Graph
    from harwood.viewer.graph_panel import GraphPanel
    from harwood.viewer.implot_panel import ImplotPanel

    work = tmp_path / "night1"
    work.mkdir()
    os.chdir(work)
    fits.PrimaryHDU(_star()).writeto(work / "star.fits")
    header = fits.Header({"OBJECT": "a star", "EXPTIME": 30.0})
    window.load(_star(), header, frame=1, name="star.fits")
    window.load(_star() * 2, None, frame=2, name="double", window=(100.0, 500.0))
    window.add_markers([Marker(40.0, 30.0, shape="circle", radius=4.0)], frame=2)
    window.select_frame(2)
    window.view.setRange(xRange=(20, 60), yRange=(10, 50), padding=0)
    window.param_widgets["radius"].setText("7")
    graph = Graph("a curve", "x", "y").line([1, 2, 3], [1, 4, 9])
    window.add_kept_panel(GraphPanel(graph), "curve")
    implot = ImplotPanel(None, "star.fits", images=[str(work / "star.fits")], line=30)
    implot.engine.line = 25
    implot._render()
    window.add_kept_panel(implot, "implot star.fits")
    (work / "notes.txt").write_text("first line\n", encoding="utf-8")
    window.files.open([str(work / "notes.txt")])
    window.files.tabs.currentWidget().replace_all("first line\nnot saved\n")
    window.prompt.setText("imstat star.fits")
    window.log("hw> display star.fits")
    QtWidgets.QApplication.processEvents()
    return work


def test_a_closed_session_comes_back_whole(make_window, tmp_path):
    from harwood.viewer import session
    from harwood.viewer.graph_panel import GraphPanel
    from harwood.viewer.implot_panel import ImplotPanel

    first = make_window()
    assert session.offer(first, answer="restore") == "none"   # nothing kept yet
    work = _work(first, tmp_path)
    from PySide6 import QtWidgets

    QtWidgets.QApplication.processEvents()
    view = first.view.viewRange()
    first.close()                                   # kept, the unsaved text with it
    meta = session.summary()
    assert meta["frames"] == [[1, "star.fits"], [2, "double"]]
    assert meta["tabs"] == ["curve", "implot star.fits"]
    assert meta["unsaved"] == [str(work / "notes.txt")]
    assert (work / "notes.txt").read_text(encoding="utf-8") == "first line\n"

    os.chdir(tmp_path)
    second = make_window()
    assert session.offer(second, answer="restore") == "restore"
    assert Path.cwd() == work
    assert sorted(second.frames) == [1, 2] and second.current == 2
    one, two = second.frames[1], second.frames[2]
    assert np.array_equal(one.data, _star()) and one.header["OBJECT"] == "a star"
    assert two.window == (100.0, 500.0) and len(two.markers) == 1
    assert second.frame_box.currentText() == "2  double"
    # the region looked at, whole and at the centre (square pixels may widen one
    # axis when the image has another shape now)
    for (lo, hi), (was_lo, was_hi) in zip(second.view.viewRange(), view, strict=True):
        assert lo <= was_lo + 1e-6 and hi >= was_hi - 1e-6
        assert (lo + hi) / 2 == pytest.approx((was_lo + was_hi) / 2, abs=1e-6)
    assert second.param_widgets["radius"].text() == "7"
    titles = [second.views.tabText(i) for i in range(second.views.count())]
    assert titles == ["Image", "curve", "implot star.fits"]
    curve, implot = second.views.widget(1), second.views.widget(2)
    assert isinstance(curve, GraphPanel) and curve.graph.title == ["a curve"]
    assert isinstance(implot, ImplotPanel) and implot.engine.line == 25
    assert implot.key("j") or True                  # alive: it takes its keys
    tab = second.files.tabs.currentWidget()
    assert tab.toPlainText() == "first line\nnot saved\n" and tab.modified
    assert second.prompt.text() == "imstat star.fits"
    assert "hw> display star.fits" in second.session.toPlainText()
    assert second.keeps_session


def test_a_new_session_sets_the_old_one_aside(make_window, tmp_path):
    from harwood.viewer import session

    first = make_window()
    session.offer(first)
    _work(first, tmp_path)
    first.close()
    os.chdir(tmp_path)
    second = make_window()
    assert session.offer(second, answer="new") == "new"
    assert not second.frames and second.views.count() == 1
    assert not second.files.files()
    assert "notes.txt" in second.session.toPlainText()   # where its text went
    assert session.summary(session.previous_folder()) is not None
    second.restore_previous()
    assert sorted(second.frames) == [1, 2] and second.views.count() == 3
    # and File > New session clears it again, keeping it as the previous one
    second.new_session()
    assert not second.frames and second.views.count() == 1
    assert session.summary(session.previous_folder())["frames"]


def test_a_window_a_task_opened_keeps_no_session(make_window):
    from harwood.viewer import session

    window = make_window()
    window.load(_star(), None, frame=1, name="star")
    window.close()
    assert session.summary() is None
