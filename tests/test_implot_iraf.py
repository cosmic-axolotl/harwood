"""implot against IRAF 2.18, and its two-point keys.

IRAF's implot was run on made-up frames with a cursor file (its coords parameter,
the graphics cursor list) and a metacode device instead of a screen. The space bar's
lines after l, c, j, k, m, n, :a, :c, :l, :step and :w are in
tests/data/iraf_implot.json. In tests/data/iraf_implot, cases that also give a, e, p
and s their second position, through CL's gcur, which is where they read it, not the
cursor file: what is printed, and every frame's window, labels, curves and marks.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

import harwood
from harwood.core import implot

REF = json.loads((Path(__file__).parent / "data" / "iraf_implot.json").read_text())
GRAPHS = Path(__file__).parent / "data" / "iraf_implot"


def _frames(where: Path) -> None:
    from astropy.io import fits

    yy, xx = np.indices((60, 80))
    a = (500 + 2 * xx + 3 * yy + 40 * np.sin(xx / 4.0) * np.cos(yy / 5.0)
         + (xx * 7 + yy * 13) % 9).astype(np.float32)
    b = (100 + xx * yy / 10.0 + 5 * np.cos(xx / 3.0)).astype(np.float32)
    fits.PrimaryHDU(a, fits.Header({"LTV1": -2.0, "LTV2": -1.0, "OBJECT": "synthetic a"})
                    ).writeto(where / "sa.fits")
    fits.PrimaryHDU(b, fits.Header({"OBJECT": "synthetic b"})).writeto(where / "sb.fits")


def test_implot_prints_what_iraf_prints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _frames(tmp_path)
    (tmp_path / "scur").write_text(REF["cursor"])
    result = harwood.run("implot", image=["sa.fits", "sb.fits"], line=20, coords="scur")
    assert result.data["text"] == REF["printed"]


CASES = yaml.safe_load((GRAPHS / "CASES.yaml").read_text())
GKI_UNITS = 2.5


def _on_screen(frame, x, y, log=False):
    wx1, wx2, wy1, wy2, sx1, sx2, sy1, sy2 = frame["window"]
    y = np.asarray(y, dtype=float)
    if log:
        with np.errstate(all="ignore"):
            part = (np.log10(y) - np.log10(wy1)) / (np.log10(wy2) - np.log10(wy1))
    else:
        part = (y - wy1) / (wy2 - wy1)
    x = np.asarray(x, dtype=float)
    return np.c_[(sx1 + (x - wx1) / (wx2 - wx1) * (sx2 - sx1)) * 32767,
                 (sy1 + part * (sy2 - sy1)) * 32767]


def _inside(frame, points):
    """Those of the points on the plot: IRAF clips the rest."""
    left, right = min(frame["window"][4:6]) * 32767 - 1, max(frame["window"][4:6]) * 32767 + 1
    bottom, top = frame["window"][6] * 32767 - 1, frame["window"][7] * 32767 + 1
    points = points[np.isfinite(points).all(axis=1)]
    return points[(points[:, 0] >= left) & (points[:, 0] <= right)
                  & (points[:, 1] >= bottom) & (points[:, 1] <= top)]


def _unmatched(points, others):
    points, others = np.reshape(points, (-1, 2)), np.reshape(others, (-1, 2))
    if not len(others):
        return points
    return np.array([p for p in points if np.abs(others - p).max(axis=1).min() > GKI_UNITS]
                    ).reshape(-1, 2)


def _session(case):
    """The engine through a case's cursor files, and each plot it started."""
    plots = []

    class Recording(implot.Implot):
        def _draw(self, erase_last=False):
            over = self.overplot and bool(self.plot.curves)
            super()._draw(erase_last)
            if not over:
                plots.append(self.plot)

    settings = CASES[case]
    engine = Recording(settings["images"], line=settings.get("line", 20),
                       wcs=settings.get("wcs", "logical"))
    second = GRAPHS / f"{case}.2nd"
    positions = iter(second.read_text().splitlines() if second.exists() else [])
    for line in (GRAPHS / f"{case}.cur").read_text().splitlines():
        x, y, _, key = line.split(None, 3)
        key = key.replace("\\040", " ")
        if key == "q":
            break
        if key.startswith(":"):
            engine.colon(key)
            continue
        engine.key(key, float(x), float(y))
        if engine.pending:
            x, y, _, key = next(positions).split(None, 3)
            engine.key(key, float(x), float(y))
    return engine, plots


@pytest.fixture
def frames_here(tmp_path, monkeypatch):
    subprocess.run([sys.executable, str(GRAPHS / "make_frames.py"), str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("case", sorted(CASES))
def test_implot_prints_and_draws_what_iraf_does(frames_here, case):
    engine, plots = _session(case)
    # What IRAF has in the second place of a 1-D image's world is memory; and p gives
    # all four lines at once, where / walks IRAF's through them.
    said = iter(engine.messages)
    for line in (GRAPHS / f"{case}.out").read_text().replace("again:", "").splitlines():
        if line in ("", "\a"):
            continue
        if case == "spec":
            line = re.sub(r"world=\[([^,]*),[^\]]*\]", r"world=[\1,1.]", line)
        assert any(line == mine for mine in said), line
    frames = json.loads((GRAPHS / f"{case}.json").read_text())
    assert len(plots) == len(frames)
    for number, (plot, frame) in enumerate(zip(plots, frames, strict=True)):
        where = f"frame {number}"
        window = frame["window"]
        width, height = abs(window[1] - window[0]), abs(window[3] - window[2])
        assert plot.window[:2] == pytest.approx(window[:2], abs=1e-4 * width), where
        assert plot.window[2:] == pytest.approx(window[2:4], abs=1e-4 * height), where
        words = {text for text in frame["text"] if not re.fullmatch(r"[-.0-9E+]+", text)}
        assert words == {text for text in (plot.xlabel, plot.title, plot.subtitle)
                         if text}, where

        theirs = [p for style, points in frame["lines"] if style != 0 for p in points]
        mine = _inside(frame, np.vstack([_on_screen(frame, x, y, plot.logy)
                                         for x, y, _ in plot.curves]))
        assert not len(_unmatched(mine, theirs)), where
        dashed = [p for style, points in frame["lines"] if style == 2 for p in points]
        for x1, y1, x2, y2 in plot.segments:
            ends = _inside(frame, _on_screen(frame, [x1, x2], [y1, y2], plot.logy))
            assert not len(_unmatched(ends, dashed)), where
        if window[0] < window[1]:           # the mark is cut up otherwise, and undecoded
            at = (_on_screen(frame, *zip(*plot.marks, strict=True), plot.logy)
                  if plot.marks else np.zeros((0, 2)))
            assert len(at) == len(frame["plus"]), where
            assert not len(_unmatched(at, frame["plus"])), where


def test_a_curve_over_the_plot_leaves_the_window_as_it_was():
    e = _engine(np.arange(1, 201, dtype=np.float32).reshape(10, 20))
    window = e.window()
    e.key("o")
    e.colon(":l 9")
    assert e.window() == window, "an overplot is drawn on the plot as it stands"
    assert len(e.plot.curves) == 2 and len(e.plot.marks) == 2


def test_the_panel_draws_the_right_axis_and_a_log_plot_in_its_window():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from harwood.viewer import _qt
    from harwood.viewer.implot_panel import ImplotPanel

    _qt.application()
    panel = ImplotPanel(np.arange(1, 401, dtype=np.float32).reshape(20, 20))
    _, _, y1, y2 = panel.engine.window()
    view = panel.plot.getViewBox().viewRange()[1]
    assert panel.plot.getAxis("right").range == pytest.approx(
        [1 + (v - y1) / (y2 - y1) * 19 for v in view])
    panel.command.setText(":log+")
    panel._run_command()
    panel.key("r")
    low, high = panel.plot.getViewBox().viewRange()[1]
    assert low < 2 < 3 < high < 3.2, "the view of a log plot is in decades"
    panel.command.setText(":log-")
    panel._run_command()
    panel.key("r")
    panel.command.setText(":x 15 5")
    panel._run_command()
    assert panel.plot.getViewBox().xInverted()


def _engine(data):
    return implot.Implot(["x"], reader=lambda _: (data, None))


def test_statistics_between_two_points():
    data = np.tile(np.arange(1, 21, dtype=np.float32), (5, 1))
    e = _engine(data)
    e.key("s", 4.2, 0)
    e.key("s", 8.0, 0)
    assert e.messages[-1] == "median=6., mean=6., rms=1.581139, sum=30., npix=5"


def test_profile_finds_the_peak_and_its_width():
    x = np.arange(1, 101, dtype=np.float32)
    line = (10 + 0.05 * x + 200 * np.exp(-0.5 * ((x - 47.3) / 3.0) ** 2)).astype(np.float32)
    e = _engine(np.tile(line, (3, 1)))
    e.key("p", 30, float(line[29]))
    e.key("p", 65, float(line[64]))
    profile, moments, half, background = e.messages[-4:]
    centre = float(profile.split("Center=")[1].split(",")[0])
    width = float(profile.split("Width=")[1].split(",")[0])
    assert centre == pytest.approx(47.3, abs=0.02)
    assert width == pytest.approx(2.35482 * 3.0, rel=0.01)
    assert float(moments.split("Centroid=")[1].split(",")[0]) == pytest.approx(47.3,
                                                                                  abs=0.05)
    assert background.startswith("[4/4] Background: (")


def test_a_averages_a_marked_range_and_e_expands():
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    e = _engine(data)
    e.key("a", 2, 0)
    e.key("a", 5, 0)                    # a range in x of a line plot: columns 2..5
    assert not e.lineplot and e.navg == 4
    assert e.plot.title == "Average of columns 2 to 5 of x"
    e.key("e", 3, 20)
    e.key("e", 7, 60)
    assert e.plot.xrange == (3, 7) and e.plot.yrange == (20, 60)


def test_colon_commands_fix_ranges_and_log():
    e = _engine(np.arange(1, 101, dtype=np.float32).reshape(10, 10))
    e.colon(":x 2 8")
    assert e.window()[:2] == (2.0, 8.0)
    e.colon(":x")
    assert e.fixed_x is None
    e.colon(":log+")
    assert not e.plot.logy, "IRAF takes it up at the next redraw"
    e.key("r")
    assert e.plot.logy
    e.colon(":step 3")
    e.key("k")
    assert e.line == e.nlines // 2 + 3 or e.line == (e.nlines + 1) // 2 + 3
