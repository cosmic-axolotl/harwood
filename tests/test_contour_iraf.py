"""contour against IRAF 2.18's: what it says, its title and the line of its levels,
the labels of the major levels, the high and low marks (nhi 0) and the values of
nhi 1 -- exactly -- and NCAR's drawing: every polyline CONREC puts into the
metacode (traced, smoothed, dashed, clipped), point for point in GKI units, and
every text it writes there (the labels in their gaps, the marks), where IRAF puts
it.

tests/data/iraf_contour holds the images (ct_inputs.py), the cases (ct_cases.py)
and what IRAF did (run.py, ct_collect.py -> contour.json).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from harwood.core import contour, images

HERE = Path(__file__).parent / "data" / "iraf_contour"
GOLD = json.loads((HERE / "contour.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from ct_cases import CASES  # noqa: E402
from ct_inputs import write  # noqa: E402

YES = {"yes": True, "no": False}


def _params(case):
    image, params = CASES[case]
    return image, {k: YES.get(v, v) for k, v in params.items()}


def _run(tmp_path, monkeypatch, case):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    image, p = _params(case)
    data, header = images.read(image)
    return contour.contour(
        data, image, p.get("title", "imtitle"), str(header.get("OBJECT", "")).strip(),
        p.get("floor"), p.get("ceiling"), p.get("zero", 0.0), p.get("ncontours", 0),
        p.get("interval", 0.0), p.get("nhi", -1), 528, p.get("label", True),
        p.get("xres", 64), p.get("yres", 64), p.get("subsample", False),
        p.get("preserve", True), p.get("perimeter", True)), data


def _inner_texts(gold):
    """The texts CONREC wrote, inside the curves' box: not the perimeter's."""
    pts = np.array([q for line in gold["lines"] for q in line], float)
    lo, hi = pts.min(0), pts.max(0)
    return [t for x, y, t in gold["texts"]
            if lo[0] < x < hi[0] and lo[1] < y < hi[1]]


@pytest.mark.parametrize("case", list(CASES))
def test_contour_says_what_iraf_says(tmp_path, monkeypatch, case):
    gold = GOLD[case]
    fatal = re.search(r'ERROR \(\d+, "(.*)"\)', gold["err"])
    if fatal:
        with pytest.raises(contour.ContourError) as error:
            _run(tmp_path, monkeypatch, case)
        assert str(error.value) == fatal.group(1).rstrip("\a")
        return
    m, _ = _run(tmp_path, monkeypatch, case)
    assert m.message == gold["err"]
    texts = [t for _, _, t in gold["texts"]]
    if m.perimeter:
        assert m.title in texts
        assert m.caption in texts
    else:
        assert m.caption not in texts


@pytest.mark.parametrize("case", [c for c in CASES if GOLD[c]["lines"]
                                  and _params(c)[1].get("nhi", -1) < 0])
def test_contour_labels_the_levels_iraf_labels(tmp_path, monkeypatch, case):
    m, _ = _run(tmp_path, monkeypatch, case)
    theirs = set(_inner_texts(GOLD[case]))
    ours = {lev.label.strip() for lev in m.levels if lev.major}
    assert theirs <= ours
    if not _params(case)[1].get("label", True):
        assert not ours and not theirs


@pytest.mark.parametrize("case", [c for c in CASES if _params(c)[1].get("nhi", -1) >= 0])
def test_contour_marks_what_iraf_marks(tmp_path, monkeypatch, case):
    m, _ = _run(tmp_path, monkeypatch, case)
    inner = _inner_texts(GOLD[case])
    if _params(case)[1]["nhi"] == 0:
        theirs = Counter(t for t in inner if t in ("H", "L"))
        ours = Counter(kind for _, _, kind, _ in m.marks)
        assert ours == theirs
        values = {t.strip() for t in inner if t not in ("H", "L")}
        assert {text.strip() for *_, text in m.marks} <= values
    else:
        assert Counter(text.strip() for *_, text in m.marks) <= Counter(
            t.strip() for t in inner)


@pytest.mark.parametrize("case", [c for c in CASES if GOLD[c]["lines"]])
def test_contour_draws_iraf_s_metacode(tmp_path, monkeypatch, case):
    """CONREC's polylines come first in the metacode, then t_contour's perimeter; its
    texts, the labels and MINMAX's, come before the perimeter's."""
    m, _ = _run(tmp_path, monkeypatch, case)
    d = m.drawing
    gold = GOLD[case]
    assert [[list(p) for p in line] for line in d.lines] == gold["lines"][:len(d.lines)]
    assert [[x, y, t] for x, y, t, *_ in d.texts] == gold["texts"][:len(d.texts)]
    rest = gold["lines"][len(d.lines):]
    if m.perimeter:
        # what follows is the perimeter's: on or outside CONREC's viewport
        x1, x2, y1, y2 = (v * 32767 for v in d.viewport)
        inside = [p for line in rest for p in line
                  if x1 + 2 < p[0] < x2 - 2 and y1 + 2 < p[1] < y2 - 2]
        assert not inside
    else:
        assert not rest


def test_the_curves_are_in_the_image_s_pixels(tmp_path, monkeypatch):
    m, _ = _run(tmp_path, monkeypatch, "section")
    pts = np.concatenate([c for lev in m.levels for c in lev.curves])
    # within the frame, to the GKI unit the metacode truncates to
    unit = m.ncols / 32767 / (m.drawing.viewport[1] - m.drawing.viewport[0])
    assert pts[:, 0].min() >= 1 - 2 * unit and pts[:, 0].max() <= m.ncols + 2 * unit
    assert pts[:, 1].min() >= 1 - 2 * unit and pts[:, 1].max() <= m.nlines + 2 * unit
    labels = [t for t in m.texts if t[2].strip()]
    assert labels and all(1 <= x <= m.ncols and 1 <= y <= m.nlines
                          for x, y, *_ in labels)
