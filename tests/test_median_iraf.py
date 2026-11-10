"""median against IRAF 2.18's: every pixel as stored, BITPIX, what verbose prints,
and what it refuses.

tests/data/iraf_median holds the cases (med_cases.py), the images (med_inputs.py:
real, short and double, narrower and shorter than the windows, 1-D, one column, a
NaN) and what IRAF's median wrote for each (run.py, med_collect.py -> median.json):
every window shape, the four boundaries, the rejection limits, and sections --
among them the narrow images and the subsampled sections where IRAF's own code
gives what a median of the window would not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import harwood

HERE = Path(__file__).parent / "data" / "iraf_median"
GOLD = json.loads((HERE / "median.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from med_cases import CASES  # noqa: E402
from med_inputs import write  # noqa: E402


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    where = tmp_path_factory.mktemp("median")
    write(where)
    return where


def _run(case: str):
    params = dict(CASES[case])
    given = params.pop("input")
    root, _, section = given.partition("[")
    image = given if ".fits" in given else f"{root}.fits" + (f"[{section}" if section else "")
    return harwood.run("median", input=[image], output="out.fits", format="iraf",
                       xwindow=params.pop("xwindow"), ywindow=params.pop("ywindow"),
                       **{"verbose": False, **params})


@pytest.mark.parametrize("case", [c for c in CASES if "data" in GOLD[c]])
def test_median_writes_what_iraf_writes(frames, monkeypatch, case):
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
    mine = result.data.get("text", [])
    assert mine == [line.replace(f"o{number:03d}.fits", "out.fits") for line in said]


@pytest.mark.parametrize("case", [c for c in CASES if "data" not in GOLD[c]])
def test_median_refuses_what_iraf_refuses(frames, monkeypatch, case):
    monkeypatch.chdir(frames)
    with pytest.raises(ValueError) as error:
        _run(case)
    iraf_said = " ".join(GOLD[case]["said"])
    message = str(error.value)
    assert any(part in iraf_said for part in message.split(": ")[-1:] + [message])
    assert getattr(error.value, "fix", "")


def test_the_course_s_lamp(frames, monkeypatch):
    """7.1: median cuar_btpf cuar_median 7 1 -- a 7 x 1 window along the lines;
    without an extension the name is written as typed, as harwood writes names."""
    monkeypatch.chdir(frames)
    harwood.run("median", input=["a.fits"], output="cuar_median", xwindow=7, ywindow=1)
    assert Path("cuar_median").exists()
