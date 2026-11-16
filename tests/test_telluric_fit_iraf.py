"""telluric and skytweak by hand -- IRAF's twk_fit -- against IRAF 2.18's, from files of
cursor reads and answers: everything printed (the prompts, the colon commands, the
bells), the pixels written, and every graph drawn: the windows and viewports of the
candidates' graph and of the one below, the title and label, each curve where IRAF's
metacode puts it, and the sample's marks.

tests/data/iraf_telluric holds the sessions (twkfit_cases.py) and what IRAF left in
each (run_fit.py, twkfit_collect.py -> twkfit.json); the spectra are twk_inputs.py's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import harwood
from harwood.core import gtools, telluric

HERE = Path(__file__).parent / "data" / "iraf_telluric"
GOLD = json.loads((HERE / "twkfit.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from twk_inputs import spectra  # noqa: E402
from twkfit_cases import CASES  # noqa: E402

GKI = 32767


@pytest.fixture
def start(tmp_path, monkeypatch):
    spectra(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(start, monkeypatch, case):
    task, params, cursor, answers = CASES[case]
    (start / "cursor").write_text(cursor, encoding="utf-8")
    (start / "answers").write_text(answers, encoding="utf-8")
    sessions = []
    made = telluric.FitSession.__init__

    def keep(self, *args, **kwargs):
        sessions.append(self)
        made(self, *args, **kwargs)

    monkeypatch.setattr(telluric.FitSession, "__init__", keep)
    data = harwood.run(task, cursor="cursor", answers="answers", **params).data
    return data["text"], sessions


def _gki(values, a, b, v1, v2, log):
    values = np.asarray(values, float)
    if log:
        values = np.array([gtools.elog(v) for v in values])
        a, b = gtools.elog(a), gtools.elog(b)
    return (v1 + (values - a) / (b - a) * (v2 - v1)) * GKI


def _on_screen(xs, ys, window, log):
    x1, x2, y1, y2, vx1, vx2, vy1, vy2 = window
    gx, gy = _gki(xs, x1, x2, vx1, vx2, log[0]), _gki(ys, y1, y2, vy1, vy2, log[1])
    inside = ((gx > vx1 * GKI + 3) & (gx < vx2 * GKI - 3)
              & (gy > vy1 * GKI + 3) & (gy < vy2 * GKI - 3))
    return gx, gy, inside


def _drawn(gx, gy, inside, vertices):
    """Each of our points inside the viewport is a vertex of IRAF's lines."""
    for x, y in zip(gx[inside], gy[inside], strict=True):
        near = np.abs(vertices - (x, y)).max(axis=1)
        assert near.min() <= 2, (x, y)


def _compare(ours: telluric.FitFrame, gold: dict):
    assert len(ours.windows) == len(gold["windows"])
    logs = gold["logs"] or [[False, False]] * len(ours.windows)
    assert [list(log) for log in ours.logs[:len(ours.windows)]] == logs
    for mine, port, theirs in zip(ours.windows, ours.viewports, gold["windows"],
                                  strict=True):
        for a, b in zip(mine, theirs[:4], strict=True):
            assert a == pytest.approx(b, rel=2e-6, abs=6e-8)
        assert port == pytest.approx(theirs[4:], abs=1e-4)
    # the label is the last thing written, under the lowest graph, when there is one
    assert (gold["text"][-1] == ours.xlabel) if ours.xlabel else \
        (gold["text"][-1] in ours.title or not ours.title)
    for line in ours.title:
        assert line in gold["text"]
    vertices = np.array([p for line in gold["lines"] for p in line], float)
    whole = [np.array(line, float) for line in gold["lines"] if len(line) == ours.x.size]
    # the candidates, as gt_plot draws them: turned under transpose
    window, log = gold["windows"][0], logs[0]
    count = 0
    for curve in ours.curves:
        xs, ys = (curve, ours.x) if ours.transposed[0] else (ours.x, curve)
        gx, gy, inside = _on_screen(xs, ys, window, log)
        _drawn(gx, gy, inside, vertices)
        count += bool(inside.all())
    if ours.below is not None:
        xs, ys = (ours.below, ours.x) if ours.transposed[1] else (ours.x, ours.below)
        gx, gy, inside = _on_screen(xs, ys, gold["windows"][1], logs[1])
        _drawn(gx, gy, inside, vertices)
        count += bool(inside.all())
    assert len(whole) >= count
    # the sample's marks, which rg_gxmarkr draws in the window's own coordinates
    pairs = [np.array(line, float) for line in gold["lines"] if len(line) == 2]
    y1, y2, y3 = ours.heights

    def has(xa, ya, xb, yb):
        gx, gy, inside = _on_screen([xa, xb], [ya, yb], window, log)
        if not inside.all():
            return None
        ends = np.column_stack([gx, gy])
        return any(np.abs(p - ends).max() <= 2 or np.abs(p[::-1] - ends).max() <= 2
                   for p in pairs)

    for xa, xb, tick_a, tick_b in ours.regions:
        assert has(xa, y3, xb, y3) in (True, None)
        for x, tick in ((xa, tick_a), (xb, tick_b)):
            assert has(x, y1, x, y2) in ((True, None) if tick else (False, None))


@pytest.mark.parametrize("case", list(CASES))
def test_the_session_says_and_writes_what_iraf_does(start, monkeypatch, case):
    text, _ = _run(start, monkeypatch, case)
    gold = GOLD[case]
    said = "\n".join(line for line in text if not line.startswith("Warning: "))
    assert said == gold["said"].rstrip("\n")
    assert [line for line in text if line.startswith("Warning: ")] == \
        gold["warned"].splitlines()
    assert sorted(p.stem for p in Path().glob("c*.fits")) == sorted(gold["images"])
    from test_telluric_iraf import _pixels

    for name, theirs in gold["images"].items():
        mine = _pixels(name + ".fits")
        assert list(mine.shape) == theirs["shape"], name
        assert mine.view(np.uint32).ravel().tolist() == theirs["bits"], name


@pytest.mark.parametrize("case", list(CASES))
def test_the_graphs_are_iraf_s(start, monkeypatch, case):
    _, sessions = _run(start, monkeypatch, case)
    gold = GOLD[case]["frames"]
    drawn = [f for s in sessions for f in s.frames]
    assert len(drawn) == len(gold)
    for ours, theirs in zip(drawn, gold, strict=True):
        _compare(ours, theirs)


def test_an_invalid_sample_is_refused(start):
    """IRAF means to refuse it (ifnoerr ... else a warning) and does not: the sample
    becomes the text typed, no pixel is in it, the RMS is INDEF and the search finds
    nothing from then on. Here the warning is given and the sample kept."""
    (start / "cursor").write_text("0 0 1 :sample foo\n0 0 1 :sample\n0 0 1 q\n0 0 1 q\n",
                                  encoding="utf-8")
    text = harwood.run("telluric", input="s", output="c", cal="t", cursor="cursor").data[
        "text"]
    assert "s[*,1]: sample *" in text
    assert any(line.startswith("Warning: Range syntax error: the sample is kept")
               for line in text)
    assert "  Tweak: shift = 0.70, scale = 1.012, normalization = 0.9735" in text


def test_the_panel_is_the_session(start, monkeypatch):
    """The window's keys are the session's: the choose case's first spectrum pressed in
    the panel, the mouse over each candidate, leaves IRAF's graphs and shift and scale."""
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from harwood.core import igsfit
    from harwood.viewer import tweak_panel
    from harwood.viewer._qt import application

    application()
    reads = list(igsfit.reads(CASES["choose"][2]))
    stop = next(i for i, read in enumerate(reads) if read[3] == "q")
    panels = []

    def press(session, title=""):
        panel = tweak_panel.TweakPanel(session)
        panels.append(panel)
        mine = reads[:stop + 1] if len(panels) == 1 else [(0.0, 0.0, 1, "q", "")]
        ended = []
        panel.finished.connect(lambda: ended.append(True))
        for x, y, wcs, key, rest in mine:
            panel._cursor, panel._wcs = (x, y), wcs
            assert panel.key(key, rest)
        assert ended == [True]

    asked = []
    monkeypatch.setattr(tweak_panel, "run", press)
    monkeypatch.setattr("harwood.viewer.fitcoords_panel.ask_answer",
                        lambda question, current, title="": asked.append(question) or "yes")
    text = harwood.run("telluric", input="s", output="c", cal="t", interactive=True).data[
        "text"]
    gold = GOLD["choose"]
    first = panels[0].session
    assert len(panels) == 2
    assert asked == ["Tweak s[*,2] by eye?"]    # the first is what --interactive asked
    for ours, theirs in zip(first.frames, gold["frames"], strict=False):
        _compare(ours, theirs)
    assert len(first.frames) == stop + 1
    line = next(t for t in gold["said"].splitlines() if t.startswith("  Tweak"))
    assert line in text
    assert len(panels[0].labels()) == 3 and panels[0].labels()[1].startswith("scale")
