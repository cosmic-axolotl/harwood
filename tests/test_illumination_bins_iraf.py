"""illumination by hand against IRAF 2.18's, from files of cursor reads and answers:
the bins' graph (il_gsetbins) -- its window, title, the spectrum and each bin's mark
where IRAF's metacode has them -- the questions asked before each image and each bin,
what the keys print, and the illumination written, pixel for pixel.

tests/data/iraf_longslit holds the sessions (il_cases.py) and what IRAF left in each
(run_il.py, il_collect.py -> illum_bins.json); the images are ls_inputs.py's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import harwood
from harwood.core import images, longslit

HERE = Path(__file__).parent / "data" / "iraf_longslit"
GOLD = json.loads((HERE / "illum_bins.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from il_cases import CASES  # noqa: E402
from ls_inputs import write  # noqa: E402

GKI = 32767


@pytest.fixture
def start(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(start, monkeypatch, case):
    names, params, cursor, answers = CASES[case]
    (start / "cursor").write_text(cursor, encoding="utf-8")
    (start / "answers").write_text(answers, encoding="utf-8")
    sessions = []
    made = longslit.BinsSession.__init__

    def keep(self, *args, **kwargs):
        sessions.append(self)
        made(self, *args, **kwargs)

    monkeypatch.setattr(longslit.BinsSession, "__init__", keep)
    params = dict(params)
    outputs = [f"{name}.fits" for name in params.pop("output", "o").split(",")]
    data = harwood.run("illumination", images=names.split(","),
                       illuminations=outputs, cursor="cursor", answers="answers",
                       **params).data
    return data, sessions


def _gki(values, a, b, v1, v2):
    return (v1 + (np.asarray(values, float) - a) / (b - a) * (v2 - v1)) * GKI


def _compare(ours: longslit.BinsFrame, gold: dict):
    window = gold["window"]
    x1, x2, y1, y2, vx1, vx2, vy1, vy2 = window
    for mine, theirs in zip(ours.window, window[:4], strict=True):
        assert mine == pytest.approx(theirs, rel=2e-6, abs=6e-8)
    assert gold["text"][-len(ours.title):] == ours.title
    # IRAF labels the axis with the other one's name: Column for lines, Line for columns
    wrong = {"Line": "Column", "Column": "Line"}[ours.xlabel]
    assert gold["text"][-len(ours.title) - 1] == wrong

    def inside(gx, gy):
        return ((gx > vx1 * GKI + 3) & (gx < vx2 * GKI - 3)
                & (gy > vy1 * GKI + 3) & (gy < vy2 * GKI - 3))

    vertices = np.array([p for line in gold["lines"] for p in line], float)
    gx, gy = _gki(ours.x, x1, x2, vx1, vx2), _gki(ours.y, y1, y2, vy1, vy2)
    for x, y in zip(gx[inside(gx, gy)], gy[inside(gx, gy)], strict=True):
        assert np.abs(vertices - (x, y)).max(axis=1).min() <= 2
    pairs = [np.array(line, float) for line in gold["lines"] if len(line) == 2]
    h1, h2, h3 = ours.heights

    def has(xa, ya, xb, yb):
        px, py = _gki([xa, xb], x1, x2, vx1, vx2), _gki([ya, yb], y1, y2, vy1, vy2)
        if not inside(px, py).all():
            return None
        ends = np.column_stack([px, py])
        return any(np.abs(p - ends).max() <= 2 or np.abs(p[::-1] - ends).max() <= 2
                   for p in pairs)

    for xa, xb, tick_a, tick_b in ours.drawn:
        assert has(xa, h3, xb, h3) in (True, None)
        for x, tick in ((xa, tick_a), (xb, tick_b)):
            assert has(x, h1, x, h2) in ((True, None) if tick else (False, None))


@pytest.mark.parametrize("case", list(CASES))
def test_the_session_asks_says_and_writes_what_iraf_does(start, monkeypatch, case):
    data, _ = _run(start, monkeypatch, case)
    gold = GOLD[case]
    assert data.get("stdout", "") == gold["said"]
    assert gold["warned"] == ""
    assert sorted(p.stem for p in Path().glob("o*.fits")) == sorted(gold["images"])
    for name, theirs in gold["images"].items():
        pixels, header = images.read(f"{name}.fits")
        mine = np.asarray(pixels, np.float32)
        assert list(mine.shape) == theirs["shape"], name
        assert mine.view(np.uint32).ravel().tolist() == theirs["bits"], name
        assert header.get("CCDMEAN") == 1.0


@pytest.mark.parametrize("case", list(CASES))
def test_the_bins_graph_is_iraf_s(start, monkeypatch, case):
    _, sessions = _run(start, monkeypatch, case)
    gold = GOLD[case]["frames"]
    drawn = [f for s in sessions for f in s.frames]
    assert len(drawn) == len(gold)
    for ours, theirs in zip(drawn, gold, strict=True):
        _compare(ours, theirs)


def test_the_panel_is_the_session(start, monkeypatch):
    """The window's keys are the session's: the fresh case pressed in the panel leaves
    IRAF's bins and its illumination."""
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from harwood.core import igsfit
    from harwood.viewer import bins_panel, icfit_panel
    from harwood.viewer._qt import application

    application()
    reads = list(igsfit.reads(CASES["fresh"][2]))
    stop = next(i for i, read in enumerate(reads) if read[3] == "q")
    panels, asked = [], []

    def press(session, title=""):
        panel = bins_panel.BinsPanel(session)
        panels.append(panel)
        ended = []
        panel.finished.connect(lambda: ended.append(True))
        for x, y, _, key, rest in reads[:stop + 1]:
            panel._cursor = (x, y)
            assert panel.key(key, rest)
        assert ended == [True]

    monkeypatch.setattr(bins_panel, "run", press)
    monkeypatch.setattr("harwood.viewer.fitcoords_panel.ask_answer",
                        lambda question, current, title="": asked.append(question) or "NO")
    monkeypatch.setattr(icfit_panel, "make", lambda *a, **k: pytest.fail("no fit by hand"))
    harwood.run("illumination", images=["sky2"], illuminations=["o.fits"],
                interactive=True)
    assert panels[0].session.bins == "50:10 70:140 "
    assert asked == ["Fit the illumination by hand for sky2 at bin 1?"]
    for ours, theirs in zip(panels[0].session.frames, GOLD["fresh"]["frames"], strict=True):
        _compare(ours, theirs)
    theirs = GOLD["fresh"]["images"]["o"]
    mine = np.asarray(images.read("o.fits")[0], np.float32)
    assert mine.view(np.uint32).ravel().tolist() == theirs["bits"]
