"""mkspec against IRAF 2.18's, to the bit: a flat spectrum, a ramp, and two
blackbodies, one of them a frame of four lines.

tests/data/iraf_mkspec holds the cases (mk_cases.py) and what IRAF wrote (mk.json:
the pixels as their bits, and the title).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import spectrum
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_mkspec"
GOLD = json.loads((HERE / "mk.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from mk_cases import CASES  # noqa: E402


@pytest.mark.parametrize("case", list(CASES))
def test_mkspec_makes_what_iraf_makes(tmp_path, monkeypatch, case):
    monkeypatch.chdir(tmp_path)
    get_task("mkspec").run(image_name=case, **CASES[case])
    theirs = np.array(GOLD[case]["bits"], np.uint32).view(np.float32).reshape(
        GOLD[case]["shape"])
    if theirs.ndim == 1:
        aperture, = spectrum.read(f"{case}.fits")
        mine = aperture.flux.astype(np.float32)
        title = fits.getheader(f"{case}.fits")["OBJECT"]
    else:
        mine = fits.getdata(f"{case}.fits")
        title = fits.getheader(f"{case}.fits")["OBJECT"]
    assert np.array_equal(mine, theirs)
    assert title == GOLD[case]["title"].strip('"')
