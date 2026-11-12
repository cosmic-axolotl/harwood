"""surface against IRAF 2.18's: what it says (surf_limits' four messages and the
block average's), its title, and where every point of the grid lands -- NCAR's
EZSRFC and TRN32S ported, nearly every vertex IRAF draws is within one unit of
NCAR's 1..1024 frame of one of harwood's projected points (the few others are
where a line disappears behind a ridge: NCAR's floating horizon, which harwood
replaces by painting the cells back to front).

tests/data/iraf_surface holds the cases (srf_cases.py, on iraf_contour's images)
and what IRAF did (run.py, srf_collect.py -> surface.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from harwood.core import images, surface

HERE = Path(__file__).parent / "data" / "iraf_surface"
GOLD = json.loads((HERE / "surface.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "iraf_contour"))
from ct_inputs import write  # noqa: E402
from srf_cases import CASES  # noqa: E402

YES = {"yes": True, "no": False}


def _run(tmp_path, monkeypatch, case):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    image, params = CASES[case]
    p = {k: YES.get(v, v) for k, v in params.items()}
    data, header = images.read(image)
    try:
        return surface.surface(data, image, p.get("title", "imtitle"),
                               str(header.get("OBJECT", "")).strip(), p.get("floor"),
                               p.get("ceiling"), p.get("angh", -33.0), p.get("angv", 25.0),
                               p.get("xres", 64), p.get("yres", 64),
                               p.get("subsample", False), p.get("preserve", True)), None, p
    except surface.SurfaceError as error:
        from harwood.core.contour import getdata
        _, message = getdata(np.asarray(data, np.float32), p.get("subsample", False),
                             p.get("preserve", True), p.get("xres", 64), p.get("yres", 64))
        return None, message + str(error) + "\n", p


@pytest.mark.parametrize("case", list(CASES))
def test_surface_says_what_iraf_says(tmp_path, monkeypatch, case):
    s, stopped, _ = _run(tmp_path, monkeypatch, case)
    gold = GOLD[case]
    if stopped is not None:
        assert stopped == gold["err"]
        assert not gold["lines"]
        return
    assert s.message == gold["err"]
    assert s.title in [t for _, _, t in gold["texts"]]


@pytest.mark.parametrize("case", [c for c in CASES if GOLD[c]["lines"]])
def test_every_point_lands_where_iraf_puts_it(tmp_path, monkeypatch, case):
    s, _, p = _run(tmp_path, monkeypatch, case)
    pts = np.array([q for line in GOLD[case]["lines"] for q in line], float)
    lo, hi = (0.1, 0.9) if p.get("label") else (0.0, 1.0)
    ours = np.column_stack([s.u.ravel(), s.v.ravel()])
    gki = (lo + (hi - lo) * (ours - 1) / 1023) * 32767
    dist, _ = cKDTree(gki).query(pts)
    near = 33 * (hi - lo)
    if p.get("label"):
        lines, _ = surface.perimeter(s)
        axes = np.vstack(lines)
        axes_gki = (lo + (hi - lo) * (axes - 1) / 1023) * 32767
        dist = np.minimum(dist, cKDTree(axes_gki).query(pts)[0])
        assert (dist < near).mean() > 0.80      # the labels are stroked, not points
    else:
        assert (dist < near).mean() > 0.97
        assert np.median(dist) < 2.0
