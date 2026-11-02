"""imcombine's offsets, outlimits, pixel masks and project, quadrature and nmodel, scales from
files and keywords, and the masks it writes, against IRAF 2.18.

tests/data/iraf_imcombine_geometry holds the sha256 of what IRAF wrote (run.py,
imcgeo_cases.py): the bits of the image and the sigma, and the whole numbers of the masks.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from harwood.core import images
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_imcombine_geometry"
GOLD = json.loads((HERE / "imcgeo.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from imcgeo_cases import ALL, CASES, OUTPUTS  # noqa: E402
from imcgeo_collect import values  # noqa: E402
from imcgeo_inputs import write  # noqa: E402


def run(case, **more):
    params = dict(CASES[case])
    given = params.pop("images", ALL)
    arguments = {"masktype": "none", "exposure_key": "EXPTIME", **OUTPUTS, **more}
    for key, value in params.items():
        if key == "expname":
            arguments["exposure_key"] = value
        elif key in ("mclip", "project"):
            arguments[key] = value == "yes"
        else:
            arguments[key] = value
    return get_task("imcombine").run(input=[given], output="out.fits", **arguments)


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path)


@pytest.mark.parametrize("case", list(GOLD))
def test_imcombine_writes_what_iraf_writes(work, case):
    run(case)
    expected = GOLD[case]
    for file, want in expected["files"].items():
        array = values(file)
        assert list(array.shape) == want["shape"], file
        assert hashlib.sha256(array.tobytes()).hexdigest() == want["sha"], file
    header = images.read_header("out.fits")
    for key, value in expected["cards"].items():
        assert header.get(key, 0.0) == pytest.approx(value, abs=1e-9), key


def test_a_project_loses_the_axis_it_combined(work):
    run("project")
    header = images.read_header("out.fits")
    assert header["NAXIS"] == 2 or "NAXIS3" not in header
    assert "CRPIX3" not in header and "CD3_3" not in header and header["CD2_2"] == 0.25


def test_a_mask_goes_by_pixel_and_not_by_ltv(work):
    # f1's LTV is (3, -2): IRAF only matches through it when pmmatch is set
    run("m-good0")
    assert np.array_equal(values("nrej.pl")[2:5, 4:9] > 0, np.ones((3, 5), bool))


@pytest.mark.parametrize("given, match", [
    (dict(offsets="nofile.txt"), "no type and no file"),
    (dict(masktype="whatever"), "invalid or ambiguous masktype"),
    (dict(masktype="novalue", maskvalue="0"), "maskvalue cannot be 0"),
    (dict(masktype="goodvalue", maskvalue="-1"), "bad mask value"),
    (dict(outlimits="500 600"), "leave no output"),
    (dict(scale="@nofile.txt"), "no such file"),
    (dict(zero="!NOKEY"), "no keyword NOKEY"),
    (dict(rdnoise="NOKEY", reject="ccdclip"), "no keyword NOKEY"),
])
def test_what_it_cannot_do_it_says(work, given, match):
    with pytest.raises(Exception, match=match):
        get_task("imcombine").run(input=[ALL], output="out.fits", **given)


def test_the_output_offsets_show_in_the_explanation(work):
    result = run("off-sigclip")
    actions = [step.action for step in result.steps]
    assert "offsets" in actions and "reject" in actions
    frames = next(step for step in result.steps if step.action == "read").measured["frames"]
    assert frames == 5


def test_masks_written_as_fits_are_fits_too(work):
    from astropy.io import fits

    get_task("imcombine").run(input=[ALL], output="out.fits", masktype="none",
                              offsets="offs.txt", bpmasks="bpm.fits", rejmasks="rej.fits",
                              expmasks="exp.fits")
    assert fits.getdata("bpm.fits").shape == (41, 51)
    assert fits.getdata("rej.fits").shape == (5, 41, 51)
    assert fits.getdata("exp.fits").shape == (41, 51)
