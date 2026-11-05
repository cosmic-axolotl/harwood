"""identify's keys and graphs against IRAF 2.18.

tests/data/iraf_identify holds cursor files IRAF ran on tests/iraf_field.arc_spectrum(),
with what its prompts read and what f's icfit was told: what it printed, the record it
wrote, and from its graphics metacode each frame's window, title, label, spectrum and
ticks, and icfit's graphs between them; where w and the /commands put the window, what
GIO clipped is not held to anything.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import numpy as np
import pytest
from iraf_field import arc_spectrum

from harwood.core import gtools, icfit, icfitsession, identifysession
from harwood.core import identify as idn

DATA = Path(__file__).parent / "data"
KEYS = DATA / "iraf_identify"
CASES = sorted(path.stem for path in KEYS.glob("*.cur"))
NOW = re.compile(r"^(# )?\w{3} \d\d:\d\d:\d\d .*|^Community.*|^harwood$")
GKI_UNITS = 4


def _session():
    ident = idn.Identify(
        arc_spectrum(), idn.read_linelist(DATA / "arc_lines.dat"),
        icfit.FitSpec(function="spline3", order=1, sample="*", naverage=1, niterate=0,
                      low_reject=3.0, high_reject=3.0, grow=0.0),
        fwidth=4.0, ftype="emission", cradius=5.0, threshold=0.0, minsep=2.0, match=-3.0,
        maxfeatures=50)
    return identifysession.IdentifySession(ident, "arc", "identify arc - Ap 1",
                                           "arc_lines.dat", "database", section=" - Ap 1")


@functools.cache
def _run(case):
    """The session once for each case, and what it drew: some keys search for minutes."""
    drawn = []
    classes = identifysession.IdentifySession, icfitsession.IcfitSession
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


def _text(text: str) -> list[str]:
    return [NOW.sub("(when)", line) for line in text.splitlines()]


def _same(ours: str, theirs: str) -> bool:
    """IRAF's fit is in double here too, but the rest of the print in real can differ
    in the last figure a %10.8g shows, and a pixel by the .01 centring is held to."""
    if ours == theirs:
        return True
    pattern = r"-?\d+\.?\d*(?:[eE][-+]?\d+)?|[^\d\s]+"
    a, b = re.findall(pattern, ours), re.findall(pattern, theirs)
    if len(a) != len(b):
        return False
    for u, v in zip(a, b, strict=True):
        if u != v:
            try:
                if abs(float(u) - float(v)) > 1.5e-5 * max(abs(float(v)), 1) + 0.0051:
                    return False
            except ValueError:
                return False
    return True


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


def _within(window, points, margin=2.0):
    """The points well inside the viewport: not where GIO cut a line at its edge."""
    sx1, sx2, sy1, sy2 = np.array(window[4:8]) * 32767
    points = np.reshape(points, (-1, 2))
    return points[(points[:, 0] > sx1 + margin) & (points[:, 0] < sx2 - margin)
                  & (points[:, 1] > sy1 + margin) & (points[:, 1] < sy2 - margin)]


@pytest.mark.parametrize("case", CASES)
def test_the_keys_say_and_write_what_iraf_s_do(case):
    session, _ = _run(case)
    ours, theirs = _text(session.stdout), _text((KEYS / f"{case}.out").read_text(
            encoding="utf-8"))
    assert len(ours) == len(theirs)
    for mine, iraf_s in zip(ours, theirs, strict=True):
        assert _same(mine, iraf_s), (mine, iraf_s)
    assert session.stderr == (KEYS / f"{case}.err").read_text(encoding="utf-8")
    record = KEYS / f"{case}.db"
    assert session.write == record.exists()
    if record.exists():
        rows = [line.split() for line in record.read_text(encoding="utf-8").splitlines()
                if line.startswith("\t ")]
        rows = [row for row in rows if len(row) >= 6 and re.fullmatch(r"[-\d.]+", row[0])
                and row[4] in ("1", "2")]
        assert len(rows) == len(session.features)
        for row, feature in zip(rows, session.features, strict=True):
            assert float(row[0]) == pytest.approx(feature.pix, abs=0.0051)
            assert float(row[1]) == pytest.approx(feature.fit, abs=2e-3)
            if row[2] == "INDEF":
                assert feature.user is None
            else:
                assert float(row[2]) == pytest.approx(feature.user, abs=2e-3)
            assert float(row[5]) == feature.wts


@pytest.mark.parametrize("case", CASES)
def test_the_graphs_are_what_iraf_draws(case):
    _, drawn = _run(case)
    frames = json.loads((KEYS / f"{case}.json").read_text(encoding="utf-8"))
    assert len(frames) == len(drawn)
    for number, (frame, graph) in enumerate(zip(frames, drawn, strict=True)):
        where = f"frame {number}"
        window = frame["windows"][0]
        width, height = abs(window[1] - window[0]), abs(window[3] - window[2])
        if height > 1e-3:
            assert graph.window[:2] == pytest.approx(window[:2], abs=1e-4 * width), where
            assert graph.window[2:] == pytest.approx(window[2:4], abs=2e-4 * height), where
        lines = np.array([p for _, points in frame["lines"] for p in points]).reshape(-1, 2)
        if isinstance(graph, identifysession.Graph):
            logs = frame.get("logs", [[False, False]])[0]
            assert tuple(graph.logs) == tuple(logs), where
            said = [*graph.title.split("\n"), graph.xlabel, graph.ylabel,
                    *(t.label for t in graph.ticks)]
            for text in said:
                assert not text or text in frame["text"], (where, text)
            for tick in graph.ticks:
                base = _on_screen(window, [tick.x], [tick.y], logs)[0]
                sign = 1 if tick.upward else -1
                top = base[1] + sign * 0.05 * 32767
                if window[6] * 32767 < top < window[7] * 32767:
                    end = np.array([base[0], top])
                    assert np.abs(lines - end).max(axis=1).min() <= GKI_UNITS, (where, tick)
            mine = _on_screen(window, graph.x, graph.y, logs)
            if graph.kind == "mark":
                marks = np.array([m[1:3] for m in frame["marks"]]).reshape(-1, 2)
                for point in _within(window, mine, 371):
                    assert np.abs(marks - point).max(axis=1).min() <= GKI_UNITS, where
            elif graph.kind == "line":
                # The spectrum, in as many pieces as the window cut it into; off the
                # edges, where the axes' ticks are.
                pieces = [p for _, p in frame["lines"] if len(p) >= 3]
                spectrum = np.array([q for p in pieces for q in p[1:-1]]).reshape(-1, 2)
                inside = _within(window, spectrum, 1000)
                assert len(inside), where
                for point in inside[::40]:
                    assert np.abs(mine - point).max(axis=1).min() <= GKI_UNITS, where
        else:
            words = [t for t in frame["text"] if not re.fullmatch(r"[-.0-9E+]+", t)]
            for text in (*graph.params, graph.xlabel, graph.ylabel):
                assert not text or any(_same(text, word) for word in words), (where, text)
            if height > 1e-3 and len(graph.kept[0]):
                marks = np.array([m[1:3] for m in frame["marks"]]).reshape(-1, 2)
                for point in _on_screen(window, *graph.kept):
                    assert np.abs(marks - point).max(axis=1).min() <= GKI_UNITS, where


def test_a_line_marked_twice_loses_its_coordinate_to_the_list():
    session = _session()
    session.key("m", 100.0)
    session.answer("4043.3948")
    assert session.features[0].user == pytest.approx(4043.3948)
    session.key("m", 100.5)
    assert len(session.features) == 1
    assert session.features[0].user is None, "nothing fitted yet, and the list has none near"
