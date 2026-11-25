"""emsao's graph and cursor -- RVSAO's emplot -- against RVSAO 2.8.5's, from files of
cursor reads and of answers to its questions: everything printed, the header cards
and the pixels written, and every graph: its window and viewport, the spectrum, each
line's mark (solid or dashed) and label, the title, and the summary page beside it.

tests/data/iraf_emsao holds the sessions (ep_cases.py) and what RVSAO left in each
(run_plot.py, ep_collect.py -> emplot.json); the spectra are ep_inputs.py's. The
cases that delete part of the spectrum ran on build_fixed.sh's RVSAO, whose d does
not end in a segmentation violation.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from harwood.core import emsao, igsfit

HERE = Path(__file__).parent / "data" / "iraf_emsao"
GOLD = json.loads((HERE / "emplot.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from ep_cases import CASES  # noqa: E402
from ep_inputs import write_all  # noqa: E402

GKI = 32767
STAMP = re.compile(r"^(IRAF|harwood) rvsao\.emsao 2\.8\.5 .*\n", re.M)
TIME = re.compile(r" \d\d-[A-Z][a-z][a-z]-\d{4} \d\d:\d\d")
RVSAO_CARDS = re.compile(r"^(VELOCITY|Z|CZERR|ZERR|VELQUAL|BCV|VELSET|CZEM|ZEM|CZEMERR|"
                         r"CZEMNL|CZEMNLF|EMLINE\d+|EL\d+_\d+|APVEL\d+|APVEM\d+|EMSAO|QPLOT|"
                         r"HISTORY)\s*=?")


def _params(params: dict) -> dict:
    out = {}
    for key, value in params.items():
        out[key] = (value == "yes") if value in ("yes", "no") else value
    return out


def run(tmp_path, monkeypatch, case):
    write_all(tmp_path)
    monkeypatch.chdir(tmp_path)
    spectra, params, cursor, answers = CASES[case]
    session = emsao.Emsao({"displot": True, "curmode": True, "logfiles": "STDOUT",
                           **_params(params)})
    reads = igsfit.reads(cursor)
    session.read_key = lambda: next(((x, y, w, k) for x, y, w, k, _ in reads), None)
    lines = iter(answers.splitlines())
    session.read_line = lambda: next(lines, None)
    session.run(spectra.split(","))
    return session


# IRAF's FITS kernel writes back the header it holds in memory, RA, DEC, HA and ST
# of 00:00:00 that getimage gave it and today's DATE among them: the spectrum read
# after the velocity was saved has a position and a date. harwood writes RVSAO's
# own cards only (see Xcsao.save), and these cases differ there.
KERNEL = {"save_ms", "save_twice"}
KERNEL_LINES = re.compile(r"^(Object: .*|Observed .*| \(No observation date\).*)$")


def _plain(text: str) -> str:
    return STAMP.sub("STAMP\n", text)


def _kernel_free(text: str) -> str:
    """Every report but the first without the lines the kernel's header changes."""
    blocks = text.split("\nFile: ")
    out = blocks[:2]
    for block in blocks[2:]:
        out.append("\n".join(line for line in block.split("\n")
                             if not KERNEL_LINES.match(line)))
    return "\nFile: ".join(out)


@pytest.mark.parametrize("case", list(CASES))
def test_emplot_says_what_rvsao_says(tmp_path, monkeypatch, case):
    session = run(tmp_path, monkeypatch, case)
    mine, theirs = _plain(session.out), _plain(GOLD[case]["said"])
    if case in KERNEL:
        mine, theirs = _kernel_free(mine), _kernel_free(theirs)
    assert mine == theirs


def _cards(cards: list[str]) -> list[str]:
    return [TIME.sub(" TIME", c.rstrip()) for c in cards if RVSAO_CARDS.match(c)]


@pytest.mark.parametrize("case", [c for c in CASES if GOLD[c]["spectra"]
                                  or "save_vel" in CASES[c][1]])
def test_emplot_writes_what_rvsao_writes(tmp_path, monkeypatch, case):
    run(tmp_path, monkeypatch, case)
    from astropy.io import fits

    for name, theirs in GOLD[case]["spectra"].items():
        raw = Path(name).read_bytes()
        mine = []
        for k in range(0, len(raw), 80):
            card = raw[k:k + 80].decode("ascii")
            if card.startswith("END "):
                break
            mine.append(card)
        assert _cards(mine) == _cards(theirs["cards"]), name
        with fits.open(name, do_not_scale_image_data=True) as hdul:
            data = np.asarray(hdul[0].data, np.float32)
        assert list(data.shape) == theirs["shape"]
        assert data.view(np.uint32).ravel().tolist() == theirs["bits"], name


def _gki(frame, x, y, ndc=False):
    nx, ny = (x, y) if ndc else frame.ndc(x, y)
    return nx * GKI, ny * GKI


def _compare(ours, gold, kernel=False):
    window = gold["windows"][0]
    for a, b in zip(ours.window, window[:4], strict=True):
        assert a == pytest.approx(b, rel=2e-6, abs=6e-8)
    assert ours.viewport == pytest.approx(window[4:8], abs=1e-4)
    if ours.summary:
        assert gold["windows"][1][4:8] == pytest.approx([0.67, 1.0, 0.0, 1.0])
    # the spectrum: every vertex of its polyline (GIO sends 1023 points at a time,
    # each piece starting where the one before ended)
    joined = []
    for line in gold["lines"]:
        points = [tuple(line[k:k + 2]) for k in range(1, len(line), 2)]
        if joined and len(joined[-1]) % 1022 == 1 and len(joined[-1]) > 1000 \
                and points[0] == joined[-1][-1]:
            joined[-1] += points[1:]
        else:
            joined.append(points)
    curve = [line for line in joined if len(line) == ours.x.size]
    assert len(curve) == 1
    points = np.array(curve[0], float)
    gx, gy = _gki(ours, ours.x.astype(float), ours.y.astype(float))
    assert np.abs(points[:, 0] - gx).max() <= 2
    assert np.abs(points[:, 1] - gy).max() <= 2
    # the lines' marks: the two-point lines inside the viewport
    v1, v2 = window[4] * GKI + 3, window[5] * GKI - 3
    pairs = np.array([line[1:] for line in gold["lines"] if len(line) == 5
                      and v1 < line[1] < v2 and line[1] == line[3]], float)
    mine = np.array([[*_gki(ours, a, b), *_gki(ours, c, d)] for a, b, c, d in ours.marks],
                    float).reshape(-1, 4)
    # a mark wholly below or above the viewport is clipped away by GIO; one across
    # its edge is cut there: those are compared where they stand across
    y1, y2 = window[6] * GKI, window[7] * GKI
    low, high = mine[:, [1, 3]].min(axis=1), mine[:, [1, 3]].max(axis=1)
    mine = mine[(high >= y1) & (low <= y2)]
    assert len(mine) == len(pairs)
    if len(mine):
        whole = (mine[:, [1, 3]].min(axis=1) >= y1) & (mine[:, [1, 3]].max(axis=1) <= y2)
        assert np.abs(mine[whole] - pairs[whole]).max(initial=0) <= 2
        assert np.abs(mine[:, 0] - pairs[:, 0]).max() <= 2
    # the texts other than the axes' numbers, in the order drawn
    texts = [t for t in gold["texts"] if not re.fullmatch(r"-?[\d.]+(E-?\d+)?", t[2])]
    labels = [(t, _gki(ours, t.x, t.y)) for t in ours.labels]
    summary = [(t, _gki(ours, t.x, t.y, ndc=True)) for t in ours.summary]
    words = [t[2] for t in texts]
    assert ours.xlabel in words
    assert (ours.title in words) if ours.title else True
    rest = [t for t in texts if t[2] not in (ours.xlabel, ours.title)]
    assert len(rest) == len(labels) + len(summary), [t[2] for t in rest]
    for (text, (x, y)), theirs in zip(labels + summary, rest, strict=True):
        if text.text.startswith("rvsao."):
            assert TIME.sub("", text.text) == TIME.sub("", theirs[2])
        elif kernel and (text.text.startswith(("File:", "Object:"))
                         or (text.ndc and text.y == 0.96 and text.x > 0.6)):
            pass                       # the header IRAF's kernel wrote: see KERNEL
        else:
            assert text.text == theirs[2]
        assert abs(x - theirs[0]) <= 2 and abs(y - theirs[1]) <= 2, text.text
        assert round(text.size * 100) == theirs[4], text.text
        assert (theirs[3] == 180) == text.up


@pytest.mark.parametrize("case", list(CASES))
def test_emplot_draws_what_rvsao_draws(tmp_path, monkeypatch, case):
    session = run(tmp_path, monkeypatch, case)
    drawn = [f for plot in session.plots for f in plot.frames]
    gold = GOLD[case]["frames"]
    assert len(drawn) == len(gold)
    for k, (ours, theirs) in enumerate(zip(drawn, gold, strict=True)):
        _compare(ours, theirs, kernel=case in KERNEL and k > 0)


@pytest.mark.parametrize("case", ["zoom", "emission", "labels", "set_typed"])
def test_the_panel_is_the_session(tmp_path, monkeypatch, case):
    """The window's keys are the session's: a case's keys pressed in the panel, with
    the mouse where the cursor file puts it and the questions answered in their
    dialogs, leave what RVSAO printed and drew."""
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from harwood.viewer import ask, emplot_panel
    from harwood.viewer._qt import application

    application()
    write_all(tmp_path)
    monkeypatch.chdir(tmp_path)
    spectra, params, cursor, answers = CASES[case]
    reads = list(igsfit.reads(cursor))
    replies = iter(answers.splitlines())
    asked = []

    def answer(prompt, title=""):
        asked.append(prompt)
        if prompt.rstrip().endswith(("none:", "velocity:", "s>et:")):
            x, y, _, key, _ = reads.pop(0)
            return key
        return next(replies, None)

    monkeypatch.setattr(ask, "line", answer)

    def press(plot, title=""):
        assert plot.start()
        panel = emplot_panel.EmplotPanel(plot)
        ended = []
        panel.finished.connect(lambda: ended.append(True))
        while reads and not ended:
            x, y, _, key, _ = reads.pop(0)
            panel._cursor = (x, y)
            assert panel.key(key)
        if not ended:                   # the cursor file's end: the window closed
            plot.finish()

    session = emsao.Emsao({"displot": True, "curmode": True, "logfiles": "STDOUT",
                           **_params(params)})
    session.show = press
    session.run(spectra.split(","))
    assert _plain(session.out) == _plain(GOLD[case]["said"])
    drawn = [f for plot in session.plots for f in plot.frames]
    assert len(drawn) == len(GOLD[case]["frames"])
    for ours, theirs in zip(drawn, GOLD[case]["frames"], strict=True):
        _compare(ours, theirs)
