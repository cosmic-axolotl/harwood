"""contour against IRAF 2.18's: what it says, its title and the line of its levels,
the labels of the major levels, the high and low marks (nhi 0) and the values of
nhi 1 -- exactly -- and its curves: every point IRAF draws, taken back to the
image's pixels through the perimeter's tick labels, lies on one of harwood's levels
(to a small fraction of the interval: NCAR smooths its lines with splines, harwood
traces them straight between the grid's edges).

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
from scipy.ndimage import map_coordinates

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


@pytest.mark.parametrize("case", [c for c in CASES if GOLD[c]["lines"]
                                  and _params(c)[1].get("perimeter", True)])
def test_iraf_draws_its_curves_on_harwoods_levels(tmp_path, monkeypatch, case):
    m, data = _run(tmp_path, monkeypatch, case)
    _, p = _params(case)
    gold = GOLD[case]
    pts = np.array([q for line in gold["lines"] for q in line], float)
    lo = pts.min(0)
    bottom = [(int(t), x) for x, y, t in gold["texts"] if t.isdigit() and y < lo[1]]
    left = [(int(t), y) for x, y, t in gold["texts"] if t.isdigit() and x < lo[0]]
    ax = np.polyfit([x for _, x in bottom], [v for v, _ in bottom], 1)
    ay = np.polyfit([y for _, y in left], [v for v, _ in left], 1)
    px, py = np.polyval(ax, pts[:, 0]), np.polyval(ay, pts[:, 1])
    inside = (px > 1.5) & (px < m.ncols - 0.5) & (py > 1.5) & (py < m.nlines - 0.5)
    z, _ = contour.getdata(np.asarray(data, np.float32), p.get("subsample", False),
                           p.get("preserve", True), p.get("xres", 64), p.get("yres", 64))
    z = z.astype(float) - float(np.float32(p.get("zero", 0.0)))
    gx = 1 + (px[inside] - 1) * (m.nx - 1) / (m.ncols - 1)
    gy = 1 + (py[inside] - 1) * (m.ny - 1) / (m.nlines - 1)
    values = map_coordinates(z, [gy - 1, gx - 1], order=1)
    levels = np.array([lev.value for lev in m.levels])
    off = np.min(np.abs(values[:, None] - levels[None, :]), axis=1) / abs(m.interval)
    assert np.median(off) < 0.05
    assert np.percentile(off, 90) < 0.35
