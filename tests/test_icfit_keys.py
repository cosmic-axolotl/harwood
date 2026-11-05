"""icfit's keys and graphs against IRAF 2.18, in fit1d, which is icfit on one line.

tests/data/iraf_icfit holds cursor files IRAF ran, with what its prompts read: what it
printed on both streams, the file colon commands wrote, and from its graphics metacode
each frame's window, title, labels, marks and curves, where w and the /commands put
them: of what a window leaves out, GIO draws nothing, so only what is inside it is held
to what was drawn. IRAF's fit there is in real and
harwood's in double, so a number is held to about the sixth figure. An exactly fitted
few points are not here: there IRAF's real arithmetic gives a large rms, and harwood 0.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from harwood.core import gtools, icfit, icfitsession

DATA = Path(__file__).parent / "data"
KEYS = DATA / "iraf_icfit"
CASES = sorted(path.stem for path in KEYS.glob("*.cur"))
Y = np.array(json.loads((DATA / "iraf_icfit_titles.json").read_text(encoding="utf-8"))["y"])
X = np.arange(1.0, Y.size + 1.0)
START = {"function": "spline3", "order": 3, "sample": "*", "naverage": 1, "low_reject": 0.0,
         "high_reject": 0.0, "niterate": 1, "grow": 0.0}
GKI_UNITS = 3.5
NOW = re.compile(r"^# \w{3} \d\d:\d\d:\d\d .*")


class Recording(icfitsession.IcfitSession):
    def graph(self):
        graph = super().graph()
        self.__dict__.setdefault("drawn", []).append(graph)
        return graph


def _session(case, cursor=None):
    session = Recording(X, Y, icfit.FitSpec(**START), title="o.fits[*,2]\n", xlabel="X",
                        xrange=(1.0, float(X.size)))
    session.run_commands(cursor or (KEYS / f"{case}.cur").read_text(encoding="utf-8"),
                         (KEYS / f"{case}.ans").read_text(encoding="utf-8"))
    return session


def _same(ours: str, theirs: str) -> bool:
    """The same words, and numbers to what a real fit and a double one share."""
    if ours == theirs:
        return True
    pattern = r"-?\d+\.?\d*(?:[eE][-+]?\d+)?|[^\d\s]+"
    a, b = re.findall(pattern, ours), re.findall(pattern, theirs)
    if len(a) != len(b):
        return False
    for u, v in zip(a, b, strict=True):
        if u != v:
            try:
                if abs(float(u) - float(v)) > 2e-4 * max(abs(float(v)), 1e-3) + 1e-4:
                    return False
            except ValueError:
                return False
    return True


def _text(text: str) -> list[str]:
    return [NOW.sub("# when", line) for line in text.splitlines()]


def _on_screen(window, x, y, logs=(False, False)):
    """GKI units, through elog on a logarithmic axis."""
    wx1, wx2, wy1, wy2, sx1, sx2, sy1, sy2 = window
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if logs[0]:
        x, (wx1, wx2) = _elog(x), _elog([wx1, wx2])
    if logs[1]:
        y, (wy1, wy2) = _elog(y), _elog([wy1, wy2])
    return np.c_[(sx1 + (x - wx1) / (wx2 - wx1) * (sx2 - sx1)) * 32767,
                 (sy1 + (y - wy1) / (wy2 - wy1) * (sy2 - sy1)) * 32767]


def _elog(values):
    return np.array([gtools.elog(v) for v in np.atleast_1d(values)])


def _inside(window, points, margin=0.0):
    """The points GIO leaves whole: inside the viewport, by `margin` GKI units."""
    sx1, sx2, sy1, sy2 = np.array(window[4:8]) * 32767
    points = np.reshape(points, (-1, 2))
    return points[(points[:, 0] >= sx1 + margin - 2) & (points[:, 0] <= sx2 - margin + 2)
                  & (points[:, 1] >= sy1 + margin - 2) & (points[:, 1] <= sy2 - margin + 2)]


def _unmatched(points, others, within=GKI_UNITS):
    points, others = np.reshape(points, (-1, 2)), np.reshape(others, (-1, 2))
    if not len(others):
        return points
    return np.array([p for p in points if np.abs(others - p).max(axis=1).min() > within]
                    ).reshape(-1, 2)


@pytest.mark.parametrize("case", CASES)
def test_the_keys_say_what_iraf_s_say(case, tmp_path):
    cursor = (KEYS / f"{case}.cur").read_text(encoding="utf-8")
    kept = tmp_path / "keep.txt"
    session = _session(case, cursor.replace("keep.txt", str(kept)))
    ours, theirs = _text(session.stdout), _text((KEYS / f"{case}.out").read_text(
            encoding="utf-8"))
    assert len(ours) == len(theirs)
    for mine, iraf_s in zip(ours, theirs, strict=True):
        assert _same(mine, iraf_s), (mine, iraf_s)
    assert session.stderr == (KEYS / f"{case}.err").read_text(encoding="utf-8")
    if (KEYS / f"{case}.keep").exists():
        ours, theirs = _text(kept.read_text(
                encoding="utf-8")), _text((KEYS / f"{case}.keep").read_text(
                encoding="utf-8"))
        assert len(ours) == len(theirs)
        for mine, iraf_s in zip(ours, theirs, strict=True):
            assert _same(mine, iraf_s), (mine, iraf_s)


@pytest.mark.parametrize("case", CASES)
def test_the_graphs_are_what_iraf_draws(case):
    session = _session(case)
    frames = json.loads((KEYS / f"{case}.json").read_text(encoding="utf-8"))
    drawn = []
    for graph in session.__dict__.get("drawn", []):
        if graph.overplot and drawn:
            drawn[-1].append(graph)
        else:
            drawn.append([graph])
    assert len(frames) == len(drawn)
    for number, (frame, graphs) in enumerate(zip(frames, drawn, strict=True)):
        where = f"frame {number}"
        graph = graphs[0]
        window = frame["windows"][0]
        width, height = abs(window[1] - window[0]), abs(window[3] - window[2])
        assert graph.window[:2] == pytest.approx(window[:2], abs=1e-4 * width), where
        assert graph.window[2:] == pytest.approx(window[2:4], abs=2e-4 * height), where
        logs = frame.get("logs", [[False, False]])[0]
        assert tuple(graph.logs) == tuple(logs), where

        def screen(points, logs=logs, window=window):
            return _on_screen(window, *points, logs)

        words = [text for text in frame["text"] if not re.fullmatch(r"[-.0-9E+]+", text)]
        said = [*graph.params, graph.xlabel, graph.ylabel, *graph.title.strip().split("\n")]
        for text in said:
            assert not text or any(_same(text, word) for word in words), (where, text, words)
        assert len(words) == len([text for text in said if text]), (where, words)
        lines = np.array([p for _, points in frame["lines"] for p in points]).reshape(-1, 2)
        dashed = np.array([p for style, points in frame["lines"] if style == 2
                           for p in points]).reshape(-1, 2)
        marks = {}
        for mark in frame["marks"]:
            marks.setdefault(mark[0], []).append(mark[1:3])
        whole = 371 / 4 * max(3, *graph.size) + GKI_UNITS
        for name, part in (("cross", "deleted"), ("plus", "averaged"),
                           ("diamond", "rejected")):
            mine = np.vstack([np.zeros((0, 2))] + [screen(getattr(g, part)) for g in graphs])
            if name == "plus":          # as wide as the group, which may cross the edge
                mine = _inside(window, mine)
            mine = _inside(window, mine, whole)
            assert not len(_unmatched(mine, marks.get(name, []))), (where, name)
        kept = _inside(window, screen(graph.kept), whole)
        if graph.kind == "line":
            assert not len(_unmatched(kept, lines)), where
        elif graph.kind == "mark":
            assert not len(_unmatched(kept, marks.get(graph.mark, []))), where
        for one in graphs:
            inside = _inside(window, screen(one.fit))
            # IRAF's grid is in real, which tells on the steep edge of a singular spline.
            assert len(_unmatched(inside, dashed, 8)) <= 0.03 * len(inside), where
            for x1, y1, x2, y2 in one.sample:
                ends = _inside(window, screen(([x1, x2], [y1, y2])))
                assert not len(_unmatched(ends, lines)), where
        assert (not dashed.size) == all(not len(_inside(window, screen(g.fit)))
                                        for g in graphs), where


def test_nothing_changes_until_f_and_q_keeps_the_last_fit():
    session = icfitsession.IcfitSession(X, Y, icfit.FitSpec(**START))
    first = session.fit
    session.key("d", float(X[20]), float(Y[20]))
    session.key(":", rest="order 5")
    assert session.stale and session.result() is first
    session.key("q")
    assert session.done and session.result() is first
    assert first.spec.order == 3


def test_the_panel_is_the_session():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from harwood.viewer import _qt
    from harwood.viewer.icfit_panel import IcfitPanel

    _qt.application()
    panel = IcfitPanel(X, Y, icfit.FitSpec(**START), title="a line", xlabel="X", ylabel="Y")
    seen = []
    panel.finished.connect(lambda: seen.append(True))
    panel._cursor = (float(X[20]), float(Y[20]))
    fit = panel.fit
    assert panel.key("d") and "Changed since the last fit" in panel.status.text()
    assert panel.fit is fit, "as in IRAF a deletion waits for f"
    assert panel.key("f") and panel.fit is not fit
    assert "deleted=1" in panel.status.text()
    assert panel.key("j") and panel.plot.getPlotItem().getAxis("left").labelText == "residuals"
    assert panel.key("h") and panel.key("a") and panel.session.state == "weight"
    panel.command.setText("3")
    panel._run_command()
    assert panel.session.state == "fit" and panel.session.x.size == X.size + 1
    assert not panel.key("%")
    assert panel.key("q") and seen


def test_the_panel_draws_and_takes_what_gtools_has(tmp_path, monkeypatch):
    """w and its key, :/ commands and :. ones at the panel: the view is the window
    gtools gives, flipped, on elog, and nearest is nearest in it."""
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from harwood.viewer import _qt
    from harwood.viewer.icfit_panel import IcfitPanel

    monkeypatch.chdir(tmp_path)
    _qt.application()
    panel = IcfitPanel(X, Y, icfit.FitSpec(**START), title="a line", xlabel="X", ylabel="Y")
    session, box = panel.session, panel.plot.getViewBox()
    panel._cursor = (float(X[20]), float(Y[20]))
    assert panel.key("w") and session.pending == "window"
    assert panel.key("z") and not session.pending
    (x0, x1), _ = box.viewRange()
    assert x0 < X[20] < x1 and x1 - x0 == pytest.approx(
        session.window[1] - session.window[0], rel=1e-6)
    panel.command.setText(":/xflip yes")
    panel._run_command()
    assert panel.key("r") and box.xInverted() and session.window[0] > session.window[1]
    assert panel.key("c") and "x = 21.  y" in session.stdout.splitlines()[-1]
    panel.command.setText(":/type mark")
    panel._run_command()
    panel.command.setText(":/mark box")
    panel._run_command()
    graph = session.graph()
    assert (graph.kind, graph.mark) == ("mark", "box")
    panel.command.setText(":/xflip no")
    panel._run_command()
    panel.command.setText(":/xtransform log")
    panel._run_command()
    panel.key("w")
    panel.key("a")
    ticks = panel.plot.getAxis("bottom")._tickLevels
    assert ticks and [label for _, label in ticks[0]][-2:] == ["100", "1000"]
    panel.command.setText(":.snap graph.png")
    panel._run_command()
    assert (tmp_path / "graph.png").stat().st_size > 0
    panel.command.setText(":.zero")
    panel._run_command()
    assert "cursor-mode command harwood does not have" in panel.status.text()
    assert ":.zero" not in session.stdout, "cursor mode is the panel's, not the task's"


def test_a_sample_range_is_two_ticks_and_a_bar():
    session = icfitsession.IcfitSession(X, Y, icfit.FitSpec(**{**START, "sample": "20:60"}))
    graph = session.graph()
    assert len(graph.sample) == 3
    x1, y1, x2, y2 = graph.sample[2]
    assert (x1, x2) == (20.0, 60.0) and y1 == y2
