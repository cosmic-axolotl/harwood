"""imsum against IRAF 2.18's: every pixel as stored, BITPIX, BZERO, the title, the
header parameters, what verbose prints, and what it refuses.

tests/data/iraf_imsum holds the cases (ims_cases.py), the images (ims_inputs.py:
twenty short and twenty real frames, one of every other type, 1-D, cubes) and what
IRAF's imsum wrote for each (run.py, ims_collect.py -> imsum.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import harwood

HERE = Path(__file__).parent / "data" / "iraf_imsum"
GOLD = json.loads((HERE / "imsum.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from ims_cases import CASES  # noqa: E402
from ims_inputs import write  # noqa: E402


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    where = tmp_path_factory.mktemp("imsum")
    write(where)
    return where


def _name(image: str) -> str:
    root, _, section = image.partition("[")
    return f"{root}.fits" + (f"[{section}" if section else "")


def _run(params: dict):
    params = dict(params)
    images = [_name(i) for i in params.pop("input")]
    return harwood.run("imsum", input=images, output="out.fits", format="iraf", **params)


@pytest.mark.parametrize("case", [c for c in CASES if "data" in GOLD[c]])
def test_imsum_writes_what_iraf_writes(frames, monkeypatch, case):
    monkeypatch.chdir(frames)
    result = _run(CASES[case])
    theirs = GOLD[case]
    with fits.open("out.fits", do_not_scale_image_data=True) as hdul:
        header, data = hdul[0].header, np.asarray(hdul[0].data)
    assert header["BITPIX"] == theirs["bitpix"]
    assert header.get("BZERO", 0.0) == theirs["bzero"]
    assert list(data.shape) == theirs["shape"]
    expected = np.array([np.nan if v is None else v for v in theirs["data"]])
    assert np.array_equal(data.ravel().astype(np.float64), expected, equal_nan=True)
    assert header.get("OBJECT") == theirs["object"]
    for key, card in theirs["cards"].items():
        assert float(header[key]) == float(fits.Card.fromstring(card).value), key
    said = [line for line in theirs["said"] if line.strip()]
    assert [line for line in result.data.get("text", []) if line.strip()] == \
        [line.replace(f"o{list(CASES).index(case) + 1:03d}.fits", "out.fits")
         for line in said]


@pytest.mark.parametrize("case", [c for c in CASES if "data" not in GOLD[c]])
def test_imsum_refuses_what_iraf_refuses(frames, monkeypatch, case):
    monkeypatch.chdir(frames)
    with pytest.raises(ValueError) as error:
        _run(CASES[case])
    iraf_said = " ".join(GOLD[case]["said"])
    assert str(error.value) in iraf_said
    assert getattr(error.value, "fix", "")


def test_the_course_s_average_of_two_frames(frames, monkeypatch):
    """Exercise 1: imsum im010,s011 aver1 pixt=r calct=r option=average v+ -- the
    letters IRAF takes for the types, and the same pixels as imarith's sum halved."""
    monkeypatch.chdir(frames)
    harwood.run("imsum", input=["r01.fits,r02.fits"], output="aver1.fits",
                pixtype="r", calctype="r", option="average")
    a, b = fits.getdata("r01.fits"), fits.getdata("r02.fits")
    from harwood.core import images

    mine, _ = images.read("aver1.fits")
    assert np.array_equal(mine, ((a + b) / np.float32(2)).astype(np.float32))


#: EXPTIME as written in the file, and as IRAF 2.18's imsum of that one image with
#: hparams=EXPTIME wrote it back: read by ctod, written by imputd with dtoc3 from
#: 15 digits -- neither the nearest double nor its shortest text.
IMPUTD = [("1.234567890123456", "1.23456789012346"), ("62.99999999999999", "63.0000000000001"),
          ("0.30000000000000004", "0.300000000000001"), ("63.0", "63."),
          ("1.0000000000000002", "1."), ("123456.7890123456", "123456.789012346"),
          ("9.87654321098765E-05", "9.87654321098767E-5")]


@pytest.mark.parametrize(("written", "iraf"), IMPUTD)
def test_header_parameters_are_read_and_written_as_iraf_does(written, iraf):
    from harwood.core import imsum

    card = fits.Card.fromstring(f"EXPTIME = {written:>20}")
    assert imsum.hparam_format(imsum.hparam_value(card)) == iraf
