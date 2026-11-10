"""boxcar against IRAF 2.18's: every pixel as stored, BITPIX, and what it refuses.

tests/data/iraf_boxcar holds the cases (box_cases.py) and what IRAF's boxcar wrote for
each (run.py, box_collect.py -> boxcar.json), on the images of
tests/data/iraf_fmedian/fmd_inputs.py: median's, one with DATAMIN and DATAMAX that
are not its pixels' limits, and a cube.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import harwood

DATA = Path(__file__).parent / "data"
GOLD = json.loads((DATA / "iraf_boxcar" / "boxcar.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(DATA / "iraf_boxcar"))
sys.path.insert(0, str(DATA / "iraf_fmedian"))
from box_cases import CASES  # noqa: E402
from fmd_inputs import write  # noqa: E402


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    where = tmp_path_factory.mktemp("boxcar")
    write(where)
    return where


def _run(case: str):
    params = dict(CASES[case])
    root, _, section = params.pop("input").partition("[")
    image = f"{root}.fits" + (f"[{section}" if section else "")
    extra = {"verbose": False} if "boxcar" == "fmedian" else {}
    extra.update(params)
    return harwood.run("boxcar", input=[image], output="out.fits", format="iraf",
                       xwindow=extra.pop("xwindow"), ywindow=extra.pop("ywindow"), **extra)


@pytest.mark.parametrize("case", [c for c in CASES if "data" in GOLD[c]])
def test_boxcar_writes_what_iraf_writes(frames, monkeypatch, case):
    monkeypatch.chdir(frames)
    result = _run(case)
    theirs = GOLD[case]
    with fits.open("out.fits", do_not_scale_image_data=True) as hdul:
        header, data = hdul[0].header, np.asarray(hdul[0].data)
    assert header["BITPIX"] == theirs["bitpix"]
    assert list(data.shape) == theirs["shape"]
    expected = np.array([np.nan if v is None else v for v in theirs["data"]])
    assert np.array_equal(data.ravel().astype(np.float64), expected, equal_nan=True)
    said = [line for line in theirs["said"] if line.strip()]
    number = list(CASES).index(case) + 1
    assert result.data.get("text", []) == [line.replace(f"o{number:03d}.fits", "out.fits")
                                           for line in said]


@pytest.mark.parametrize("case", [c for c in CASES if "data" not in GOLD[c]])
def test_boxcar_refuses_what_iraf_refuses(frames, monkeypatch, case):
    monkeypatch.chdir(frames)
    with pytest.raises(ValueError) as error:
        _run(case)
    said = " ".join(GOLD[case]["said"])
    assert str(error.value).split(": ")[-1] in said
    assert getattr(error.value, "fix", "")
