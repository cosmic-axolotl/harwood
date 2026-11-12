"""wcslab against IRAF 2.18's: every text it writes -- the hms and dms labels and
their units, the titles -- with its place on the page (to two GKI units of 32767),
its up vector (to the degree), size and justification, in IRAF's order; and its
lines: every vertex IRAF draws (the box, the grid, the ticks, the pole's plus)
within two GKI units of one of harwood's polylines of the same line type, and the
other way round.

tests/data/iraf_wcslab holds the images (wcl_inputs.py: only their WCS and size
matter), the cases (wcl_cases.py) and what IRAF did (run.py, wcl_collect.py ->
wcslab.json): RA/Dec in every orientation the course meets and more -- rotated,
transposed, across 0h, at and near both poles, a degree and a milliarcsecond
field, ARC and ZEA, linear and no WCS, wcspars' own WCS and none at all --, and
wlpars: ticks in and out, labels inside, rotation, full labels, sides, titles,
intervals given (all or one), minor grids, line types.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from harwood.core import wcslab

HERE = Path(__file__).parent / "data" / "iraf_wcslab"
GOLD = json.loads((HERE / "wcslab.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from wcl_cases import CASES  # noqa: E402
from wcl_inputs import write  # noqa: E402

YES = {"yes": True, "no": False}
MAXNDC = 32767
#: Where harwood differs from IRAF on purpose: a linear WCS with a rotation goes
#: through mw_v2trand's LRO branch, which drops the input's y (p2[1] for p1[2]), so
#: IRAF's lines of constant axis 1 collapse each to a point and nothing is labelled.
DIFFERENT = {"linear_rotated"}
SAME = [c for c in CASES if c not in DIFFERENT]


def _run(tmp_path, monkeypatch, case):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    image, params = CASES[case]
    p = {k: YES.get(v, v) for k, v in params.items()}
    return wcslab.labelled(image, p)


def _gki(v, rounded=False):
    """NDC to GKI: gtext's nint, and gadraw's truncation for polylines."""
    v = np.asarray(v, float) * MAXNDC
    return np.clip(np.rint(v) if rounded else np.floor(v), 0, MAXNDC)


@pytest.mark.parametrize("case", SAME)
def test_wcslab_writes_what_iraf_writes(tmp_path, monkeypatch, case):
    d, messages = _run(tmp_path, monkeypatch, case)
    gold = GOLD[case]
    assert gold["err"] == "" and messages == []
    ours = [[float(_gki(t.x, True)), float(_gki(t.y, True)), t.text, round(t.up),
             int(np.float32(t.size) * np.float32(100)), t.h, t.v] for t in d.texts]
    theirs = gold["texts"]
    assert [t[2] for t in ours] == [t[2] for t in theirs]
    for a, b in zip(ours, theirs, strict=True):
        assert abs(a[0] - b[0]) <= 2 and abs(a[1] - b[1]) <= 2, (a, b)
        # the up vector is the integer part of the angle: where the angle is a whole
        # number of degrees (a polar label, each on its own line, symmetric about
        # the pole), IRAF's and harwood's last digits decide between n - 1 and n
        assert abs(a[3] - b[3]) <= 1 and a[4:] == b[4:], (a, b)


@pytest.mark.parametrize("case", SAME)
def test_wcslab_draws_what_iraf_draws(tmp_path, monkeypatch, case):
    d, _ = _run(tmp_path, monkeypatch, case)
    gold = GOLD[case]
    for ltype in sorted({lt for lt, _ in gold["lines"]} | {s.ltype for s in d.strokes}):
        theirs = np.array([q for lt, line in gold["lines"] if lt == ltype for q in line],
                          float).reshape(-1, 2)
        ours = [np.column_stack([_gki(s.x), _gki(s.y)]) for s in d.strokes
                if s.ltype == ltype]
        assert theirs.size and ours, f"line type {ltype}"
        ours_pts = np.vstack(ours)
        # their vertices on our polylines: near a vertex, or on a segment of ours
        dense = np.vstack([_densify(o) for o in ours])
        far = cKDTree(dense).query(theirs)[0]
        assert far.max() <= 2.5, (ltype, theirs[np.argmax(far)])
        dense = np.vstack([_densify(np.array(line, float)) for lt, line in gold["lines"]
                           if lt == ltype])
        far = cKDTree(dense).query(ours_pts)[0]
        assert far.max() <= 2.5, (ltype, ours_pts[np.argmax(far)])


def test_a_rotated_linear_grid_is_drawn_where_iraf_collapses_it(tmp_path, monkeypatch):
    gold = GOLD["linear_rotated"]
    collapsed = [line for _, line in gold["lines"] if len(line) > 100
                 and len({tuple(q) for q in line}) == 1]
    assert len(collapsed) >= 5                 # IRAF's: each line a single point
    assert [t[2] for t in gold["texts"]] == ["linrot.fits: linear, rotated"]
    d, _ = _run(tmp_path, monkeypatch, "linear_rotated")
    long = [s for s in d.strokes if s.x.size > 100]
    assert long and all(np.ptp(s.x) + np.ptp(s.y) > 0.05 for s in long)
    labels = [t.text for t in d.texts if t.text.strip()]
    assert "100." in labels and "-100." in labels


def _densify(line):
    """A polyline's points every GKI unit along it."""
    out = [line[:1]]
    for a, b in zip(line[:-1], line[1:], strict=True):
        n = max(1, int(np.hypot(*(b - a))))
        t = np.linspace(0, 1, n + 1)[1:, None]
        out.append(a + (b - a) * t)
    return np.vstack(out)
