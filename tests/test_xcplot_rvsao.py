"""xcsao's graph and cursor -- RVSAO's xcplot -- against RVSAO 2.8.5's, from files of
cursor reads and answers: everything printed, the header written, and every graph:
the spectrum's and the correlation's windows and viewports, the spectrum, the part
of the correlation inside its window, the range correlated, the points at -pkfrac
of the peak, the titles and labels, and the summary.

tests/data/iraf_xcsao holds the sessions (xp_cases.py) and what RVSAO left in each
(run_plot.py, xp_collect.py -> xcplot.json); the spectra are xc_inputs.py's. The
keys that go to emsao's graph or edit the spectrum (dispmode 2 and 3, l, m to 2,
e, p) are not ported yet: their cases are left out here (NOT_YET).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from harwood.core import igsfit, xcsao

HERE = Path(__file__).parent / "data" / "iraf_xcsao"
GOLD = json.loads((HERE / "xcplot.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from xc_inputs import write  # noqa: E402
from xp_cases import CASES  # noqa: E402

GKI = 32767
STAMP = re.compile(r"^(IRAF|harwood) rvsao\.xcsao 2\.8\.5 .*\n", re.M)
TIME = re.compile(r" \d\d-[A-Z][a-z][a-z]-\d{4} \d\d:\d\d")
NOT_YET = {"dispmode2", "dispmode3", "mode_lines", "mode_lines_quit", "lines",
           "lines_quit"}
SHOWN = [c for c in CASES if c not in NOT_YET]
#: c and v as harwood has them: what they set reaches the correlation, which RVSAO's
#: does not; the correlation after them is RVSAO's with those parameters from the
#: start (the case named), the rest of the session RVSAO's own.
DIRECT = {"peak_params": "peak_params_direct", "velocity_limits": "velocity_direct"}
REPORT = re.compile(r"\n(?=\S+ Object: )")
RVSAO_CARDS = re.compile(r"^(VELOCITY|Z|CZERR|ZERR|VELQUAL|BCV|CZXC|ZXC|CZXCERR|CZXCR|"
                         r"BESTTEMP|CZPARAMS|XCTEMP\d+|APVEL\d+|APVXC\d+|XT\d+_\d+|XCSAO|"
                         r"HISTORY)\s*=?")


def run(tmp_path, monkeypatch, case):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    spectra, params, cursor, answers = CASES[case]
    session = xcsao.Xcsao({"displot": True, "curmode": True, "logfiles": "STDOUT",
                           **params})
    reads = igsfit.reads(cursor)
    session.read_key = lambda: next(((x, y, w, k) for x, y, w, k, _ in reads), None)
    lines = iter(answers.splitlines())
    session.read_line = lambda: next(lines, None)
    session.run(spectra.split(","))
    return session


def _plain(text: str) -> str:
    return STAMP.sub("STAMP\n", text)


@pytest.mark.parametrize("case", SHOWN)
def test_xcplot_says_what_rvsao_says(tmp_path, monkeypatch, case):
    session = run(tmp_path, monkeypatch, case)
    mine, theirs = _plain(session.out), _plain(GOLD[case]["said"])
    if case in DIRECT:
        mine, theirs = REPORT.split(mine), REPORT.split(theirs)
        assert mine[:2] == theirs[:2]
        assert mine[2:] == REPORT.split(_plain(GOLD[DIRECT[case]]["said"]))[1:]
        return
    assert mine == theirs


@pytest.mark.parametrize("case", [c for c in SHOWN if "save_vel" in CASES[c][1]])
def test_xcplot_writes_what_rvsao_writes(tmp_path, monkeypatch, case):
    run(tmp_path, monkeypatch, case)
    for name, theirs in GOLD[case]["spectra"].items():
        raw = Path(name).read_bytes()
        mine = []
        for k in range(0, len(raw), 80):
            card = raw[k:k + 80].decode("ascii")
            if card.startswith("END "):
                break
            mine.append(card)

        def cards(images):
            return [TIME.sub(" TIME", c.rstrip()) for c in images if RVSAO_CARDS.match(c)]
        assert cards(mine) == cards(theirs["cards"]), name


def _gki(panel, xs, ys):
    nx, ny = panel.ndc(np.asarray(xs, float), np.asarray(ys, float))
    return nx * GKI, ny * GKI


def _inside(panel, gx, gy):
    v1, v2, v3, v4 = (v * GKI for v in panel.viewport)
    return (gx > v1 + 3) & (gx < v2 - 3) & (gy > v3 + 3) & (gy < v4 - 3)


def _vertices(gold):
    return np.array([p for line in gold["lines"]
                     for p in np.array(line[1:], float).reshape(-1, 2)], float)


def _curve(panel, gold):
    """Every point of ours inside the viewport is a vertex of IRAF's lines."""
    gx, gy = _gki(panel, panel.x, panel.y)
    inside = _inside(panel, gx, gy)
    vertices = _vertices(gold)
    for x, y in zip(gx[inside], gy[inside], strict=True):
        near = np.abs(vertices - (x, y)).max(axis=1)
        assert near.min() <= 2, (x, y)
    return int(inside.sum())


def _compare(ours, gold):
    windows = gold["windows"]
    panels = [ours.spectrum] + ([ours.correlation] if ours.correlation else [])
    for panel, window in zip(panels, windows, strict=False):
        for a, b in zip(panel.window, window[:4], strict=True):
            assert a == pytest.approx(b, rel=2e-6, abs=6e-8)
        assert panel.viewport == pytest.approx(window[4:8], abs=1e-4)
    assert windows[len(panels)][4:8] == pytest.approx([0.67, 1.0, 0.0, 1.0])
    for panel in panels:
        assert _curve(panel, gold) > 0
    # the range correlated: our dashes inside the spectrum's viewport are IRAF's
    pairs = np.array([line[1:] for line in gold["lines"] if len(line) == 5], float)
    for a, b, c, d in ours.spectrum.marks:
        gx, gy = _gki(ours.spectrum, [a, c], [b, d])
        if _inside(ours.spectrum, gx, gy).all():
            end = np.array([gx[0], gy[0], gx[1], gy[1]])
            assert np.abs(pairs - end).max(axis=1).min() <= 2
    # the points at -pkfrac of the peak: a circle (a closed polyline) each
    circles = [np.array(line[1:], float).reshape(-1, 2) for line in gold["lines"]
               if len(line) > 20 and line[1:3] == line[-2:]]
    if ours.correlation is not None:
        cx, cy = _gki(ours.correlation, *zip(*ours.correlation.circles, strict=True)) \
            if ours.correlation.circles else (np.array([]), np.array([]))
        inside = _inside(ours.correlation, cx, cy) if len(cx) else np.array([], bool)
        assert len(circles) == int(inside.sum())
        for x, y, circle in zip(cx[inside], cy[inside], circles, strict=True):
            assert abs(circle[:, 0].mean() - x) <= 3 and abs(circle[:, 1].mean() - y) <= 3
    # the texts other than the axes' numbers
    texts = [t for t in gold["texts"] if not re.fullmatch(r"-?[\d.]+(E-?\d+)?", t[2])]
    words = [t[2] for t in texts]
    for panel in panels:
        assert panel.xlabel in words
        if panel.title:
            assert panel.title in words
    rest = [t for t in texts if t[2] not in [p.xlabel for p in panels]
            and t[2] not in [p.title for p in panels if p.title]]
    assert len(rest) == len(ours.texts), [t[2] for t in rest]
    for text, theirs in zip(ours.texts, rest, strict=True):
        if text.text.startswith("rvsao.xcsao"):
            assert TIME.sub("", text.text) == TIME.sub("", theirs[2])
        else:
            assert text.text == theirs[2]
        assert abs(text.x * GKI - theirs[0]) <= 2 and abs(text.y * GKI - theirs[1]) <= 2
        assert round(text.size * 100) == theirs[4], text.text


@pytest.mark.parametrize("case", SHOWN)
def test_xcplot_draws_what_rvsao_draws(tmp_path, monkeypatch, case):
    session = run(tmp_path, monkeypatch, case)
    drawn = [f for plot in session.plots for f in plot.frames]
    gold = GOLD[case]["frames"]
    if case in DIRECT:
        gold = gold[:1] + GOLD[DIRECT[case]]["frames"]
    assert len(drawn) == len(gold)
    for ours, theirs in zip(drawn, gold, strict=True):
        _compare(ours, theirs)


@pytest.mark.parametrize("case", ["template", "replot", "mode", "good"])
def test_the_panel_is_the_session(tmp_path, monkeypatch, case):
    """The window's keys are the session's: a case's keys pressed in the panel, the
    questions answered in their dialogs, leave what RVSAO printed and drew."""
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from harwood.viewer import ask, xcplot_panel
    from harwood.viewer._qt import application

    application()
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    spectra, params, cursor, answers = CASES[case]
    reads = list(igsfit.reads(cursor))
    replies = iter(answers.splitlines())
    monkeypatch.setattr(ask, "line", lambda prompt, title="": next(replies, None))

    def press(plot):
        assert plot.start()
        panel = xcplot_panel.XcplotPanel(plot)
        ended = []
        panel.finished.connect(lambda: ended.append(True))
        while reads and not ended:
            x, y, _, key, _ = reads.pop(0)
            panel._cursor = (x, y)
            assert panel.key(key)

    session = xcsao.Xcsao({"displot": True, "curmode": True, "logfiles": "STDOUT",
                           **params})
    session.show = press
    session.run(spectra.split(","))
    assert _plain(session.out) == _plain(GOLD[case]["said"])
    drawn = [f for plot in session.plots for f in plot.frames]
    assert len(drawn) == len(GOLD[case]["frames"])
    for ours, theirs in zip(drawn, GOLD[case]["frames"], strict=True):
        _compare(ours, theirs)


def test_the_task_reads_a_cursor_file_and_writes_its_hard_copies(tmp_path, monkeypatch):
    import harwood

    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "keys").write_text("5000 0.5 1 t\n5000 0.5 1 @\n5000 0.5 1 y\n"
                                   "5000 0.5 1 q\n", encoding="utf-8")
    (tmp_path / "answers").write_text("2\n", encoding="utf-8")
    result = harwood.run("xcsao", spectra=["g2500.fits"], templates="tg.fits,ta.fits",
                         cursor="keys", answers="answers", save_vel=True,
                         logfiles="STDOUT", hardcopy=True)
    pictures = sorted(p.name for p in result.outputs if p.suffix == ".svg")
    assert pictures == ["xcsao_g2500_1_1.svg", "xcsao_g2500_1_2.svg"]
    assert any(line.endswith("Making hard copy on stdplot") for line in result.data["text"])
    from astropy.io import fits

    head = fits.getheader("g2500.fits")
    assert head["VELQUAL"] == "Q"
    assert head["BESTTEMP"].strip() == "ta.fits"      # t chose the second by R
