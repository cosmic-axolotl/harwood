"""graph against IRAF 2.18's: what it says, the window and transforms of the frame it
draws, its title and labels, and every point of every curve and mark as GKI has
it, from the metacode.

tests/data/iraf_graph holds the inputs (gr_inputs.py), the cases (gr_cases.py) and
what IRAF did (run.py, gr_collect.py -> graph.json): lists of every kind of line,
standard input, images and their sections projected on each axis in logical,
physical and world coordinates, every mark and the sizes read from a list, the
histograms, several curves and their styles, windows given and rounded, log axes
and elog, linear transformations and transposition. gr_gki.py draws harwood's
strokes as GIO would, on IRAF's own frame: what glabax makes of the viewport is
the device's and is not harwood's to draw.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from harwood.core import plotgraph

HERE = Path(__file__).parent / "data" / "iraf_graph"
GOLD = json.loads((HERE / "graph.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from gr_cases import CASES, LISTS  # noqa: E402
from gr_gki import Gio  # noqa: E402
from gr_inputs import write  # noqa: E402

YES = {"yes": True, "no": False}
TRANSFORMS = {"linear": 0, "log": 1, "elog": 2}
#: stdvdm's standard marker size 1 -- greset's quarter of graphcap's character
#: height (ch#.0294) -- in NDC, and its aspect ratio (ar#.77): what an error bar
#: read from a list is as wide (or as high) as.
SZMARKER1, ASPECT = np.float32(0.0294) / np.float32(4.0), np.float32(0.77)


def _run(tmp_path, monkeypatch, case):
    write(tmp_path)
    (tmp_path / "lists").write_text(LISTS, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    operands, params, stdin = CASES[case]
    params = {k: YES.get(v, v) for k, v in params.items()}
    try:
        return plotgraph.graph({"input": operands, **params}, stdin), None
    except plotgraph.GraphError as error:
        return None, error


@pytest.mark.parametrize("case", list(CASES))
def test_graph_says_what_iraf_says(tmp_path, monkeypatch, case):
    plot, error = _run(tmp_path, monkeypatch, case)
    gold = GOLD[case]
    assert gold["out"] == ""
    fatal = re.search(r'ERROR \(\d+, "(.*)"\)', gold["err"], re.S)
    if fatal:
        assert error is not None and str(error) == fatal.group(1).rstrip("\a")
        return
    assert error is None, error
    assert plot.err == gold["err"]


@pytest.mark.parametrize("case", [c for c in CASES if "frame" in GOLD[c]])
def test_graph_frames_as_iraf(tmp_path, monkeypatch, case):
    plot, _ = _run(tmp_path, monkeypatch, case)
    frame = GOLD[case]["frame"]
    assert list(plot.window) == pytest.approx(frame["window"][:4], rel=2e-7, abs=1e-30)
    assert [TRANSFORMS[plot.xtran], TRANSFORMS[plot.ytran]] == frame["transforms"]
    if plot.viewport is not None:
        assert list(plot.viewport) == pytest.approx(frame["window"][4:], abs=1e-6)
    if plot.box:
        for text in (*plot.title_lines, plot.xlabel, plot.ylabel):
            if text.strip():
                assert text in frame["text"], text
    else:
        assert frame["text"] == []


@pytest.mark.parametrize("case", [c for c in CASES if "frame" in GOLD[c]])
def test_graph_draws_every_point_iraf_draws(tmp_path, monkeypatch, case):
    plot, _ = _run(tmp_path, monkeypatch, case)
    frame = GOLD[case]["frame"]
    theirs = [d[:4] for d in frame["drawn"] if d[4] == len(frame["text"])
              and (d[0] == "marker" or d[5] == 100)]
    ours = Gio(frame, SZMARKER1, ASPECT).strokes(plot.strokes)
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs, strict=True):
        # a polymarker's colour is GKI_PMSET's, which gr_collect does not read
        assert a[:3] == b[:3] or a[0] == b[0] == "marker"
        assert a[3] == b[3]
