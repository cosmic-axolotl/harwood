"""imexamine's b x y w t u v , . o f against IRAF 2.18: what it prints, the sections it writes,
and the curves it draws, to the 2 in 32767 the metacode holds.

tests/data/iraf_imexam_keys holds them (run.py, iek_cases.py).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import examine, images
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_imexam_keys"
GOLD = json.loads((HERE / "iek.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).parent))
from iek_cases import CASES  # noqa: E402
from iraf_field import field  # noqa: E402


def session(case):
    """The keys of a case, the way IRAF goes through them, with its autoredraw."""
    Path("keys.cur").write_text(CASES[case], encoding="utf-8")
    settings = examine.Settings(logname="out.log", ncoutput=21, nloutput=21, image="f.fits")
    fits.writeto("f.fits", np.asarray(field(), "f4"), overwrite=True)
    data, settings.header = images.read("f.fits")
    data = np.asarray(data, "f4").astype(float)
    text, plots = [], []
    for x, y, _, key in examine.read_cursor("keys.cur"):
        if key.startswith(":"):
            settings.redraw = False
            answer = examine.colon(settings, key)
            if "does not apply" in answer:
                text.append(answer)
            if settings.redraw and settings.graph and settings.last_position:
                plots.append(examine.examine(data, "f", x, y, settings).plot)
            continue
        result = examine.examine(data, key, x, y, settings)
        if key == "m":
            text.append(None)
        elif key in ("o", "l", "c") or result.line.startswith("["):
            pass
        else:
            text.extend(result.line.splitlines())
        if result.plot is not None:
            plots.append(result.plot)
    return text, plots


def theirs(case):
    """What IRAF printed, without its own log-opening line, its complaint about a printf of
    its own in the t key, and the m key's, whose text is another test's."""
    lines, out, skip = GOLD[case]["text"].splitlines(), [], False
    for number, line in enumerate(lines):
        if number == 0 and line.startswith("Log file"):
            continue
        if line.startswith(("Warning: Incomplete", "(Old format", "(New format")):
            continue
        if line.startswith("#            SECTION"):
            skip = True
            out.append(None)
            continue
        if skip:
            skip = False
            continue
        out.append(line.replace("\x07", ""))
    return out


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("case", list(GOLD))
def test_it_prints_what_iraf_prints(work, case):
    text, _ = session(case)
    assert text == theirs(case)


def _off(x, y, points) -> float:
    """How far a curve is from what was drawn, in metacode units, each axis scaled freely."""
    worst = 0.0
    for ours, drawn in ((x, points[:, 0]), (y, points[:, 1])):
        if np.ptp(ours) > 0:
            design = np.vstack([ours, np.ones(ours.size)]).T
            scale = np.linalg.lstsq(design, drawn, rcond=None)[0]
            worst = max(worst, float(np.abs(design @ scale - drawn).max()))
    return worst


@pytest.mark.parametrize("case", [c for c in GOLD if GOLD[c]["frames"]])
def test_it_draws_the_curves_iraf_draws(work, case):
    _, plots = session(case)
    frame = -1
    for plot in plots:
        if not plot.get("append"):
            frame += 1
        x, y = (plot["fit_x"], plot["fit_y"]) if "fit_y" in plot else (plot["x"], plot["y"])
        curves = GOLD[case]["frames"][frame]
        n = len(y)
        clipped = not any(len(curve) // 2 == n for curve in curves)
        if clipped:                                 # a peak over the window, its start cut
            n -= 1
        same = [curve for curve in curves if len(curve) // 2 == n]
        chosen = same[-1] if plot.get("append") else same[0]
        points = np.array(chosen).reshape(-1, 2).astype(float)
        keep = n - 10 if clipped else n
        x, y = np.asarray(x, float)[-keep:], np.asarray(y, float)[-keep:]
        assert _off(x, y, points[-keep:]) < 2.5


def test_the_sections_are_what_iraf_writes(work):
    session("section")
    for name, (shape, sha) in GOLD["section"]["sections"].items():
        data = np.asarray(fits.getdata(name)).astype(">f4")
        assert list(data.shape) == shape
        assert hashlib.sha256(data.tobytes()).hexdigest() == sha
        assert "CRPIX1" not in fits.getheader(name)


def test_the_log_stops_and_starts_with_w(work):
    Path("keys.cur").write_text(CASES["log"], encoding="utf-8")
    fits.writeto("f.fits", np.asarray(field(), "f4"), overwrite=True)
    result = get_task("imexamine").run(input=["f.fits"], imagecur="keys.cur",
                                       logfile="out.log")
    assert len(result.data["rows"]) == 5
    log = Path("out.log").read_text(encoding="utf-8")
    assert "110.00" not in log, "written while the log was closed"
    assert "20.00" in log and "41.00" in log


def test_a_second_position_can_be_abandoned(work):
    settings = examine.Settings()
    data = np.asarray(field(), float)
    assert examine.examine(data, "b", 10.0, 10.0, settings).line == "again:"
    assert examine.examine(data, "x", 20.0, 20.0, settings).ok
    assert examine.examine(data, "b", 30.0, 30.0, settings).line == "again:"


def test_d_needs_the_window(work):
    result = examine.examine(np.asarray(field(), float), "d", 10.0, 10.0, examine.Settings())
    assert not result.ok and "window" in result.line


#: run_edges.py: what t said at each place, and the columns of the image that are the image's
EDGES = {
    "2 15": ("f.fits[1:5,13:17] -> t.001.fits", (1, 5, 13, 17)),
    "39 15": ("f.fits[36:40,13:17] -> t.001.fits", (36, 40, 13, 17)),
    "20 3": ("f.fits[17:23,1:5] -> t.001.fits", (17, 23, 1, 5)),
    "20 28": ("f.fits[17:23,26:30] -> t.001.fits", (17, 23, 26, 30)),
    "20 2": ("Warning: Pixels out of bounds", None),
    "20 29": ("Warning: Pixels out of bounds", None),
    "2 2": ("Warning: Pixels out of bounds", None),
    "39 29": ("Warning: Pixels out of bounds", None),
    "60 15": ("outside", None),
    "20 40": ("outside", None),
}


@pytest.mark.parametrize("at", list(EDGES))
def test_t_at_the_edges_of_an_image(work, at):
    said, kept = EDGES[at]
    y, x = np.mgrid[1:31, 1:41]
    data = (1000.0 * y + x).astype("f4")
    settings = examine.Settings(ncoutput=7, nloutput=5, image="f.fits", output="t")
    column, line = (float(word) for word in at.split())
    result = examine.examine(data.astype(float), "t", column, line, settings)
    assert said in result.line
    if kept is None:
        assert not Path("t.001.fits").exists()
    else:
        x1, x2, y1, y2 = kept
        assert np.array_equal(fits.getdata("t.001.fits"), data[y1 - 1:y2, x1 - 1:x2])


@pytest.fixture
def screen(monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from harwood.core import display
    from harwood.viewer._qt import application
    from harwood.viewer.window import HarwoodWindow

    display.reset()
    monkeypatch.setenv("HARWOOD_DISPLAY", "native")
    application()
    window = HarwoodWindow()
    window.show()
    display.get().show(np.asarray(field(), float), None, frame=1, name="first")
    yield window
    display.reset()


def _dialogs(monkeypatch, name, frame, accepted=True):
    from PySide6 import QtWidgets

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        lambda *args, **kw: (name, "FITS images"))
    monkeypatch.setattr(QtWidgets.QInputDialog, "getInt",
                        lambda *args, **kw: (frame, accepted))


def test_d_loads_the_image_it_is_given_into_the_frame_it_is_given(work, screen, monkeypatch):
    fits.writeto("other.fits", np.asarray(field(), "f4") + 1, overwrite=True)
    _dialogs(monkeypatch, "other.fits", 3)
    assert screen.examine_at("d", 10.0, 10.0) is None
    assert sorted(screen.frames) == [1, 3]
    assert screen.current == 3
    assert screen.frames[3].name == "other.fits"
    assert np.array_equal(screen.frames[3].data, np.asarray(field(), "f4") + 1)


@pytest.mark.parametrize("name, accepted", [("", True), ("other.fits", False)])
def test_d_can_be_abandoned(work, screen, monkeypatch, name, accepted):
    fits.writeto("other.fits", np.asarray(field(), "f4"), overwrite=True)
    _dialogs(monkeypatch, name, 3, accepted)
    assert screen.examine_at("d", 10.0, 10.0) is None
    assert sorted(screen.frames) == [1]
    assert screen.current == 1
