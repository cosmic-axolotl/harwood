"""ccdtest -- mkimage, artobs, subsection -- against IRAF 2.18's: the pixels of every image
each session makes, to the bit, and the header the tasks write through the instrument's
translation (the demonstration's, demo.dat, or none).

tests/data/iraf_ccdtest holds the sessions (ccdt_cases.py) and what IRAF made (run.py,
ccdt_collect.py -> ct.json). Each session starts as a new IRAF does: mkimage's static seed
at 1.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import harwood
from harwood.core import ccdtest

HERE = Path(__file__).parent / "data" / "iraf_ccdtest"
GOLD = json.loads((HERE / "ct.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from ccdt_cases import CASES  # noqa: E402
from ccdt_collect import SKIP  # noqa: E402

#: What IRAF writes that says nothing, and harwood leaves out: DATAMIN = DATAMAX = 0 on a
#: real image edited in place (limits not computed), and the identity physical system
#: (WCSDIM, LTM, WAT) imcopy writes for a section that starts at [1,1].
SAYS_NOTHING = {"DATAMIN", "DATAMAX", "WCSDIM", "LTM1_1", "LTM2_2", "WAT0_001", "WAT1_001",
                "WAT2_001"}
POSITIONAL = {"mkimage": ("image", "option", "value"),
              "artobs": ("image", "exptime", "ccdtype"),
              "subsection": ("subimage", "image")}


def _image(path: Path) -> dict:
    with fits.open(path, do_not_scale_image_data=True) as hdul:
        raw = np.ascontiguousarray(hdul[0].data)
        header = hdul[0].header
        return {"bitpix": header["BITPIX"], "shape": list(raw.shape),
                "sha": hashlib.sha256(raw.astype(raw.dtype.newbyteorder(">")).tobytes())
                .hexdigest(),
                "cards": {k: header[k] for k in header if k not in SKIP}}


@pytest.mark.parametrize("case", list(CASES))
def test_ccdtest_makes_what_iraf_makes(tmp_path, monkeypatch, case):
    monkeypatch.chdir(tmp_path)
    ccdtest.reset()
    instrument, calls = CASES[case]
    inst = instrument.replace("ccddb$", "").removesuffix(".dat") or None
    for task, args, params in calls:
        given = dict(zip(POSITIONAL[task], args, strict=False))
        if task != "mkimage" and inst:
            given["instrument"] = inst
        harwood.run(task, **given, **params)
    gold = GOLD[case]
    assert sorted(p.name for p in tmp_path.glob("*.fits")) == sorted(gold)
    for name, theirs in gold.items():
        mine = _image(tmp_path / name)
        assert mine["bitpix"] == theirs["bitpix"], name
        assert mine["shape"] == theirs["shape"], name
        assert mine["sha"] == theirs["sha"], name
        for key, value in theirs["cards"].items():
            if key in SAYS_NOTHING:
                continue
            assert mine["cards"].get(key) == pytest.approx(value) if isinstance(
                value, float) else mine["cards"].get(key) == value, (name, key)
