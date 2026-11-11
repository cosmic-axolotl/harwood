"""fit1d and background against IRAF 2.18's: every pixel, and what was printed.

tests/data/iraf_fit1d holds the cases (f1_cases.py), the images (f1_inputs.py: a
rectified long-slit frame -- a galaxy on a sky that varies along the slit, with
noise and a few cosmic rays, so that some lines reject points and the next ones
are fitted afresh while others are only re-solved -- a 1-D spectrum, a mask, an
image to write into, a 16-bit copy) and what IRAF's tasks wrote with
interactive=no (run.py, f1_collect.py -> fit1d.json): background along both axes
with every function, rejection, growing and averaging; fit1d's three outputs, a
mask, a section into a new image and into an existing one, in place, integers.
And by hand: the lines answered to 'Fit line =' fitted from a cursor file (sample
regions marked with s, :order and :function, t), as the course's background is.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import harwood

HERE = Path(__file__).parent / "data" / "iraf_fit1d"
GOLD = json.loads((HERE / "fit1d.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from f1_cases import CASES, INTERACTIVE  # noqa: E402
from f1_inputs import write  # noqa: E402


@pytest.fixture
def frames(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("case", list(CASES) + list(INTERACTIVE))
def test_fit1d_writes_what_iraf_writes(frames, case):
    task, image, output, params = {**CASES, **INTERACTIVE}[case]
    params = dict(params)
    answers = params.pop("_answers", None)
    cursor = params.pop("_cursor", None)
    if cursor is not None:
        (frames / "cursor").write_text(cursor, encoding="utf-8")
        params.update(interactive=True, cursor=str(frames / "cursor"),
                      lines=answers.splitlines())
    root, _, section = image.partition("[")
    name = f"{root}.fits" + (f"[{section}" if section else "")
    if task == "background":
        result = harwood.run("background", input=[name], output=[f"{output}.fits"],
                             **params)
    else:
        result = harwood.run("fit1d", input=[name], output=[f"{output}.fits"],
                             type=params.pop("type"), **params)
    gold = GOLD[case]
    if cursor is None:      # IRAF's prompts and its icfit's "again:" are not compared
        assert result.data.get("text", []) == gold["said"]
    with fits.open(frames / f"{output}.fits") as hdul:
        ours = np.asarray(next(h.data for h in hdul if h.data is not None))
    theirs = np.frombuffer(base64.b64decode(gold["data"]), dtype=np.dtype(gold["dtype"]))
    theirs = theirs.reshape(gold["shape"])
    assert ours.dtype.itemsize == theirs.dtype.itemsize
    assert ours.dtype.kind == theirs.dtype.kind
    assert list(ours.shape) == gold["shape"]
    native = theirs.dtype.newbyteorder("=")
    diff = ours.astype(native) != theirs.astype(native)
    assert not diff.any(), (int(diff.sum()), np.argwhere(diff)[:5].tolist())
