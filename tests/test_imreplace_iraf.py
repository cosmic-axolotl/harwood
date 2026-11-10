"""imreplace against IRAF 2.18's: every pixel as stored, BITPIX and BZERO.

tests/data/iraf_imreplace holds the cases (imr_cases.py), the images (imr_inputs.py:
short, ushort, int, real, double, 8-bit, scaled, 1-D and a cube) and what IRAF's
imreplace left in each, edited in place (run.py, imr_collect.py -> imreplace.json).
IRAF stops on the 8-bit and the scaled images ("File write error") and leaves them
as they were; harwood writes those as the type IRAF reads them as.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import harwood

HERE = Path(__file__).parent / "data" / "iraf_imreplace"
GOLD = json.loads((HERE / "imreplace.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from imr_cases import CASES  # noqa: E402
from imr_inputs import write  # noqa: E402

REFUSED = ("b8", "scaled")


@pytest.fixture
def images_here(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(params: dict, output: str) -> None:
    params = dict(params)
    image = params.pop("image")
    root, _, section = image.partition("[")
    harwood.run("imreplace", images=[f"{root}.fits" + (f"[{section}" if section else "")],
                output=output, format="iraf", **params)


@pytest.mark.parametrize("case", [c for c in CASES if CASES[c]["image"].split("[")[0]
                                  not in REFUSED])
def test_imreplace_stores_what_iraf_stores(images_here, case):
    _run(CASES[case], "out.fits")
    theirs = GOLD[case]
    with fits.open("out.fits", do_not_scale_image_data=True) as hdul:
        header, data = hdul[0].header, np.asarray(hdul[0].data)
    assert header["BITPIX"] == theirs["bitpix"]
    assert header.get("BZERO", 0.0) == theirs["bzero"]
    mine = data.ravel().astype(np.float64)
    theirs_data = np.array([np.nan if v is None else v for v in theirs["data"]])
    assert np.array_equal(mine, theirs_data, equal_nan=True)
    if theirs["negzero"] is not None:
        assert (np.signbit(data.ravel()) & (data.ravel() == 0)).tolist() == theirs["negzero"]


@pytest.mark.parametrize("case", [c for c in CASES
                                  if CASES[c]["image"].split("[")[0] in REFUSED])
def test_iraf_refuses_eight_bit_and_scaled_images_and_harwood_writes_them(images_here,
                                                                          case):
    assert "File write error" in " ".join(GOLD[case]["said"])
    params = CASES[case]
    root = params["image"]
    physical = fits.getdata(f"{root}.fits").astype(np.float64)
    _run(params, "out.fits")
    with fits.open("out.fits") as hdul:
        bitpix, data = hdul[0].header["BITPIX"], np.asarray(hdul[0].data, dtype=np.float64)
    assert bitpix == (16 if root == "b8" else -32)
    # The window on what IRAF reads: short for 8-bit, real for scaled.
    kind = np.int16 if root == "b8" else np.float32
    value = params["value"]
    new = float(np.trunc(np.float32(value))) if root == "b8" else float(np.float32(value))
    low, high = params.get("lower"), params.get("upper")
    if root == "b8":
        def edge(v, up):
            v = np.float32(v)
            if up:
                return int(v)
            return int(v) if float(int(v)) == float(v) else int(np.float32(v + 1))
        low = None if low is None else edge(low, False)
        high = None if high is None else edge(high, True)
    else:
        low = None if low is None else float(np.float32(low))
        high = None if high is None else float(np.float32(high))
    work = physical.astype(kind).astype(np.float64)
    hit = np.ones(work.shape, bool)
    if low is not None:
        hit &= work >= low
    if high is not None:
        hit &= work <= high
    assert np.array_equal(data, np.where(hit, new, work))


def test_in_place_when_output_names_the_input(images_here):
    before = fits.getdata("r32.fits")
    harwood.run("imreplace", images=["r32.fits"], value=0.0, upper=0.0, output="r32.fits",
                format="iraf")
    after = fits.getdata("r32.fits")
    assert np.array_equal(after, np.where(before <= 0, np.float32(0), before))


def test_without_output_the_input_is_kept(images_here):
    before = fits.getdata("s16.fits")
    result = harwood.run("imreplace", images=["s16.fits"], value=0.0, lower=100.0)
    assert np.array_equal(fits.getdata("s16.fits"), before)
    assert result.outputs and Path(result.outputs[0]).name == "s16.fits"
    assert result.outputs[0] != Path("s16.fits")
    assert result.data["rows"][0]["changed"] == int(np.count_nonzero(before >= 100))


def test_radius_iraf_leaves_unwritten_is_said(images_here):
    result = harwood.run("imreplace", images=["one.fits"], value=0.0, lower=150.0,
                         radius=2.0, output="o.fits")
    assert result.data["rows"][0]["changed"] == 0
    assert any("writes nothing" in w.message for w in result.warnings)
    result = harwood.run("imreplace", images=["cube.fits"], value=0.0, lower=150.0,
                         radius=1.5, output="c.fits")
    assert any("cube" in w.message for w in result.warnings)
