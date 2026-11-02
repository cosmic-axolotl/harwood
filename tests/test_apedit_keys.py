"""apedit's keys and graph against IRAF 2.18.

tests/data/iraf_apedit holds cursor files IRAF ran on tests/iraf_field.spectra_frame(),
with what its prompts read: what it printed and, from its graphics metacode, each frame's
window, title, apertures and background band, and the graphs of b's icfit between them.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import numpy as np
import pytest
from iraf_field import spectra_frame

from harwood.core import apeditsession, icfitsession

KEYS = Path(__file__).parent / "data" / "iraf_apedit"
CASES = sorted(path.stem for path in KEYS.glob("*.cur"))
GKI_UNITS = 4
LOG = re.compile(r"\w{3} \d+ \d\d:\d\d: DATABASE.*")


def _session():
    return apeditsession.ApeditSession(spectra_frame(), [], None, 10, "frame",
                                       settings={"nfind": 2})


@functools.cache
def _run(case):
    drawn = []
    classes = apeditsession.ApeditSession, icfitsession.IcfitSession
    originals = [cls.graph for cls in classes]

    def recording(original):
        def graph(self):
            made = original(self)
            drawn.append(made)
            return made
        return graph

    try:
        for cls, original in zip(classes, originals, strict=True):
            cls.graph = recording(original)
        session = _session()
        session.run_commands((KEYS / f"{case}.cur").read_text(encoding="utf-8"),
                             (KEYS / f"{case}.ans").read_text(encoding="utf-8"))
    finally:
        for cls, original in zip(classes, originals, strict=True):
            cls.graph = original
    return session, drawn


def _same(ours: str, theirs: str) -> bool:
    """The same words, and numbers as printed, to the last figure."""
    if ours == theirs:
        return True
    pattern = r"-?\d+\.?\d*(?:[eE][-+]?\d+)?|[^\d\s]+"
    a, b = re.findall(pattern, ours), re.findall(pattern, theirs)
    if len(a) != len(b):
        return False
    for u, v in zip(a, b, strict=True):
        if u != v:
            try:
                if abs(float(u) - float(v)) > 0.0101:
                    return False
            except ValueError:
                return False
    return True


def _on_screen(window, x, y):
    wx1, wx2, wy1, wy2, sx1, sx2, sy1, sy2 = window
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return np.c_[(sx1 + (x - wx1) / (wx2 - wx1) * (sx2 - sx1)) * 32767,
                 (sy1 + (y - wy1) / (wy2 - wy1) * (sy2 - sy1)) * 32767]


@pytest.mark.parametrize("case", CASES)
def test_the_keys_say_what_iraf_s_say(case):
    session, _ = _run(case)
    theirs = [LOG.sub("", line) for line in (KEYS / f"{case}.out").read_text(
            encoding="utf-8").splitlines()
              if line not in ("Searching aperture database ...", "Finding apertures ...")
              and not LOG.fullmatch(line)]
    ours = session.stdout.splitlines()
    while theirs and not theirs[-1]:
        theirs.pop()
    assert len(ours) == len(theirs)
    for mine, iraf_s in zip(ours, theirs, strict=True):
        assert _same(mine, iraf_s), (mine, iraf_s)


@pytest.mark.parametrize("case", CASES)
def test_the_graphs_are_what_iraf_draws(case):
    _, drawn = _run(case)
    frames = json.loads((KEYS / f"{case}.json").read_text(encoding="utf-8"))
    assert len(frames) == len(drawn)
    for number, (frame, graph) in enumerate(zip(frames, drawn, strict=True)):
        where = f"frame {number}"
        window = frame["windows"][0]
        width, height = abs(window[1] - window[0]), abs(window[3] - window[2])
        assert graph.window[:2] == pytest.approx(window[:2], abs=1e-4 * width), where
        assert graph.window[2:] == pytest.approx(window[2:4], abs=2e-4 * height), where
        lines = np.array([p for _, points in frame["lines"] for p in points]).reshape(-1, 2)
        if isinstance(graph, icfitsession.Graph):
            words = [t for t in frame["text"] if not re.fullmatch(r"[-.0-9E+]+", t)]
            assert graph.title in words and graph.params[0] in words, (where, words)
            continue
        assert graph.title in frame["text"], where
        dy = 0.025 * height
        level = window[3] - 4 * dy
        for mark in graph.marks:
            assert str(mark.number) in frame["text"], where
            for x, y in ((mark.centre, level + 2 * dy), (mark.centre, level - 2 * dy),
                         (mark.low, level + dy), (mark.high, level - dy)):
                point = _on_screen(window, [x], [y])[0]
                if window[4] * 32767 < point[0] < window[5] * 32767:
                    assert np.abs(lines - point).max(axis=1).min() <= GKI_UNITS, (where, mark)
        assert bool(graph.sample) == (len(graph.marks) == 1), where
        low = window[2] + 0.02 * height
        for xl, xu in graph.sample:
            ends = _on_screen(window, [xl, xu], [low, low])
            for end in ends:
                assert np.abs(lines - end).max(axis=1).min() <= GKI_UNITS, where


def test_the_panel_is_the_session():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from harwood.viewer import _qt
    from harwood.viewer.aperture_panel import AperturePanel

    _qt.application()
    panel = AperturePanel(_session())
    seen = []
    panel.finished.connect(lambda: seen.append(True))
    panel._cursor = (35.0, 0.0)
    assert panel.key("m") and len(panel.session.apertures) == 1
    assert panel.session.apertures[0].centre == pytest.approx(34.59, abs=0.02)
    assert panel.key("b") and panel.stack.currentWidget() is panel.fit_panel
    panel.fit_panel.key("q")
    assert panel.stack.currentWidget() is panel.page
    panel._cursor = (35.0, 0.0)
    assert panel.key("i") and panel.session.state == "number"
    panel.command.setText("7")
    panel._run_command()
    assert panel.session.apertures[0].number == 7 and "aperture = 7" in panel.status.text()
    assert not panel.key("%")
    assert panel.key("q") and seen
