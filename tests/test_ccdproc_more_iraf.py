"""ccdproc against IRAF 2.18, bit by bit: every kind of overscan, trim and fixpix, each
calibration and image type, illumination and fringe, readcor and scancor, and sections of the
chip with calibrations of other sections.

tests/data/iraf_ccdproc holds the sha256 of the images IRAF wrote and their cards (run.py,
ccdp_cases.py). IRAF processes in place; harwood's inputs are left alone.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import images
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_ccdproc"
GOLD = json.loads((HERE / "ccdp.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from ccdp_cases import CASES  # noqa: E402
from ccdp_collect import FLAGS, NUMBER, TEXT, digest  # noqa: E402
from ccdp_inputs import write  # noqa: E402


def arguments(params):
    out = {}
    for key, value in params.items():
        if value in ("yes", "no"):
            value = value == "yes"
        if key in ("biassec", "trimsec") and value == "image":
            continue
        out["bpm" if key == "fixfile" else key] = value
    return out


def run(case):
    image, params = CASES[case]
    return get_task("ccdproc").run(input=[image], output="result.fits", logfile="",
                                   **arguments(params))


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path)
    return tmp_path


@pytest.mark.parametrize("case", list(GOLD))
def test_ccdproc_writes_what_iraf_writes(work, case):
    image, _ = CASES[case]
    before = (work / image).read_bytes()
    run(case)
    assert (work / image).read_bytes() == before, "the input is untouched"
    expected = GOLD[case]
    shape, sha = digest(fits.getdata("result.fits"))
    assert shape == expected["shape"]
    assert sha == expected["sha"]
    header = images.read_header("result.fits")
    assert [key for key in FLAGS if key in header] == expected["flags"]
    assert {key: header[key] for key in TEXT if key in header} == expected["text"]
    for key in NUMBER:
        assert header.get(key, 0.0) == pytest.approx(expected["number"].get(key, 0.0),
                                                     abs=1e-9), key


@pytest.mark.parametrize("case, flag", [("illum-noflag", "MKILLUM"),
                                        ("fringe-noflag", "MKFRINGE")])
def test_with_no_flag_the_correction_is_left_out_with_a_warning(work, case, flag):
    result = run(case)
    assert any(flag in str(warning) for warning in result.warnings)
    assert flag in GOLD[case]["error"]


def test_a_scanned_calibration_is_kept_beside_the_one_it_came_from(work):
    run("scan-short")
    assert Path("flat.3.fits").exists() and Path("zero.3.fits").exists()
    run("scan-long")
    assert fits.getdata("flat.1d.fits").shape == (1, 52)


def test_a_named_calibration_goes_through_the_same_steps_in_memory(work):
    """As ccdcheck and ccdproc do inside set_flat, without writing over the file."""
    before = Path("flat.fits").read_bytes()
    named = get_task("ccdproc").run(input=["obj.fits"], output="named.fits", overscan=True,
                                    trim=True, biassec="[53:60,1:40]", trimsec="[1:52,1:37]",
                                    flatcor=True, flat="flat.fits", logfile="")
    assert Path("flat.fits").read_bytes() == before
    assert any(step.action.startswith("flat flat.fits") for step in named.steps)
    done = get_task("ccdproc").run(input=["flat.fits"], output="doneflat.fits", overscan=True,
                                   trim=True, biassec="[53:60,1:40]", trimsec="[1:52,1:37]",
                                   logfile="")
    assert done.outputs
    again = get_task("ccdproc").run(input=["obj.fits"], output="again.fits", overscan=True,
                                    trim=True, biassec="[53:60,1:40]", trimsec="[1:52,1:37]",
                                    flatcor=True, flat="doneflat.fits", logfile="")
    assert again.outputs
    assert np.array_equal(fits.getdata("named.fits"), fits.getdata("again.fits"))


@pytest.mark.parametrize("given, match", [
    (dict(overscan=True, function="median", biassec="[53:60,*]", readaxis="column"),
     "not allowed with readaxis column"),
    (dict(overscan=True, biassec="[1:60,1:40]"), "given as full image"),
    (dict(overscan=True, biassec="[53:90,*]"), "error in the bias section"),
    (dict(trim=True, trimsec="[1:900,*]"), "error in the trim section"),
    (dict(zerocor=True, zero="sub2.fits", trim=True, trimsec="[1:52,1:37]"),
     "CCD section error"),
])
def test_what_it_cannot_do_it_says(work, given, match):
    with pytest.raises(Exception, match=match):
        get_task("ccdproc").run(input=["obj.fits"], output="result.fits", logfile="", **given)
