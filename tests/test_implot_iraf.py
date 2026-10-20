"""implot against IRAF 2.18, and its two-point keys.

IRAF's implot was run on two made-up frames with a cursor file (its coords
parameter, the graphics cursor list) and a metacode device instead of a
screen: the space bar's lines after l, c, j, k, m, n, :a, :c, :l, :step and
:w are in tests/data/iraf_implot.json, and harwood prints the same ones. s,
p, a and e read their second position from the terminal in IRAF, whatever
the cursor file says, so they are checked here on their own arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import harwood
from harwood.core import implot

REF = json.loads((Path(__file__).parent / "data" / "iraf_implot.json").read_text())


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
    assert e.plot.logy
    e.colon(":step 3")
    e.key("k")
    assert e.line == e.nlines // 2 + 3 or e.line == (e.nlines + 1) // 2 + 3
