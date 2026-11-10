"""response and illumination against IRAF 2.18's: every pixel, CCDMEAN and MKILLUM.

tests/data/iraf_longslit holds the cases (ls_cases.py), the images (ls_inputs.py: a
dome flat and a twilight sky with a star in it, dispersed along the lines and along
the columns) and what IRAF's tasks wrote for each with interactive=no (run.py,
ls_collect.py -> longslit.json): every fitting function, rejection, averaging,
sample ranges and a threshold for response, the calibration as a section and an
existing image written into; for illumination one to seven bins, bins given in
any order or overlapping, the five interpolators and the lower ones IRAF falls
back to.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import harwood
from harwood.core import images

HERE = Path(__file__).parent / "data" / "iraf_longslit"
GOLD = json.loads((HERE / "longslit.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from ls_cases import ILLUMINATION, RESPONSE  # noqa: E402
from ls_inputs import write  # noqa: E402


@pytest.fixture
def frames(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _name(image: str) -> str:
    root, _, section = image.partition("[")
    return f"{root}.fits" + (f"[{section}" if section else "")


def _compare(case: str, path: str) -> None:
    theirs = GOLD[case]
    pixels, header = images.read(path)
    data = np.asarray(pixels, dtype=np.float64)
    assert list(data.shape) == theirs["shape"]
    expected = np.array([np.nan if v is None else v for v in theirs["data"]])
    assert np.array_equal(data.ravel(), expected, equal_nan=True)
    assert header.get("CCDMEAN") == theirs["ccdmean"]
    assert header.get("MKILLUM") == theirs["mkillum"]


@pytest.mark.parametrize("case", list(RESPONSE))
def test_response_writes_what_iraf_writes(frames, case):
    params = dict(RESPONSE[case])
    output = params.pop("output", "o") + ".fits"
    harwood.run("response", calibration=[_name(params.pop("calibration"))],
                normalization=[_name(params.pop("normalization"))], response=[output],
                **params)
    _compare(case, output)


@pytest.mark.parametrize("case", list(ILLUMINATION))
def test_illumination_writes_what_iraf_writes(frames, case):
    params = dict(ILLUMINATION[case])
    harwood.run("illumination", images=[_name(params.pop("images"))],
                illuminations=["o.fits"], **params)
    _compare(case, "o.fits")
