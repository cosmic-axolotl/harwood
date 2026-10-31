"""listpixels and minmax against IRAF 2.18, line by line.

tests/data/iraf_lists holds what IRAF printed (run.py, cases.py) for the images
of list_images.py: every pixel type, sections that flip, step and drop axes,
physical and world coordinates through LTV, CD and TAN, and fmtio's formats. For
minmax also the limits it leaves in the header and what the next run makes of them.

Where harwood differs from IRAF on purpose, a test below says so.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import iraffmt
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_lists"
LISTS = json.loads((HERE / "listpixels.json").read_text())
MINMAX = json.loads((HERE / "minmax.json").read_text())
# Told apart below: DIFFERENT by a test, UNTOUCHED because harwood leaves the card alone.
DIFFERENT = {"section_valid"}
UNTOUCHED = {"lfine_update", "lmin_update", "lmax_update", "lzero_update"}
YES = {"yes": True, "no": False}
# %f prints 16 decimals, more than a double has at 20 degrees: one bit of the
# declination, 4e-15, differs here and in no line of the other cases.
UNIT_IN_THE_LAST_PLACE = {"format9", "format18"}

_spec = importlib.util.spec_from_file_location("list_images", HERE / "list_images.py")
list_images = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(list_images)


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    where = tmp_path_factory.mktemp("lists") / "fresh"
    list_images.write(where)
    return where


@pytest.fixture
def work(fresh, tmp_path, monkeypatch):
    shutil.copytree(fresh, tmp_path / "w")
    monkeypatch.chdir(tmp_path / "w")
    return tmp_path / "w"


def _params(params):
    out = {k: YES.get(v, v) for k, v in params.items()}
    out["images"] = [out["images"]]
    return out


@pytest.mark.parametrize("case", list(LISTS))
def test_listpixels_prints_what_iraf_prints(work, case):
    lines = get_task("listpixels").run(**_params(LISTS[case]["params"])).stdout
    expected = LISTS[case]["stdout"]
    if case in UNIT_IN_THE_LAST_PLACE:
        assert [float(line.split()[-2]) for line in lines] == pytest.approx(
            [float(line.split()[-2]) for line in expected], abs=1e-14)
        lines, expected = [[line.rsplit(" ", 3)[0] for line in side]
                           for side in (lines, expected)]
    assert lines == expected


@pytest.mark.parametrize("case", [c for c in MINMAX if c not in DIFFERENT])
def test_minmax_prints_what_iraf_prints(work, case):
    for k, step in enumerate(MINMAX[case]):
        result = get_task("minmax").run(**_params(step["params"]))
        expected = [line for line in step["stdout"] if not line.startswith("Warning")]
        if "[ERR]" in step["params"]["images"]:
            expected = [line.replace("[2]", "[ERR]") for line in expected]
        if case in UNTOUCHED and k == 1:
            continue
        assert result.stdout == (expected if step["params"].get("verbose") != "no" else [])
        name = step["params"]["images"]
        if case in UNTOUCHED or case.startswith("empty") or set(name) & set("[,?"):
            continue
        header = fits.getheader(name, 0)
        for key, value in step["header"].items():
            assert header.get(key) == pytest.approx(value, rel=1e-9), key


def test_minmax_gives_the_last_image_as_the_task_s_values(work):
    data = get_task("minmax").run(images=["f.fits,fs.fits"], update=False).data
    assert (data["minval"], data["maxval"]) == (-4.0, 1234.0)
    assert (data["minpix"], data["maxpix"]) == ("[3,4]", "[2,1]")
    assert [row["image"] for row in data["rows"]] == ["f.fits", "fs.fits"]


def test_a_section_is_always_measured(work):
    """IRAF, with limits in the header, prints the whole image's for a section
    -- 1234.5 for a section that does not reach it, without saying so -- if the
    file is at least a second older than the limits. Here a section is looked at."""
    get_task("minmax").run(images=["f.fits"])
    lines = get_task("minmax").run(images=["f.fits[2:4,2:4]"], update=False).stdout
    assert MINMAX["section_valid"][1]["stdout"] == ["    f.fits[2:4,2:4]  -4.5  1234.5"]
    assert lines == ["    f.fits[2:4,2:4] [2,3] -4.5 [3,2] 113.75"]


def test_a_section_never_touches_the_header(work):
    """IRAF's minmax on a section with update and force loses DATAMIN and DATAMAX."""
    get_task("minmax").run(images=["f.fits"])
    before = (work / "f.fits").read_bytes()
    get_task("minmax").run(images=["f.fits[2:4,2:4]"], force=True)
    assert (work / "f.fits").read_bytes() == before


def test_limits_already_there_are_not_rewritten(work):
    """IRAF rewrites the file, to seven figures, whenever update is on."""
    before = (work / "lfine.fits").read_bytes()
    lines = get_task("minmax").run(images=["lfine.fits"]).stdout
    assert lines == MINMAX["lfine"][0]["stdout"]
    assert (work / "lfine.fits").read_bytes() == before


def test_an_empty_image_gets_no_limits(work):
    """IRAF writes DATAMIN = INDEF, which is not FITS."""
    get_task("minmax").run(images=["empty.fits"])
    assert "DATAMIN" not in fits.getheader("empty.fits")


def test_updating_leaves_the_pixels_as_they_are_on_disk(work):
    """Scaled integers stay scaled integers; only the two cards change."""
    for name in ("fscale.fits", "fu.fits"):
        raw = fits.open(name, do_not_scale_image_data=True)
        pixels, bitpix = raw[0].data.copy(), raw[0].header["BITPIX"]
        raw.close()
        get_task("minmax").run(images=[name])
        with fits.open(name, do_not_scale_image_data=True) as after:
            assert after[0].header["BITPIX"] == bitpix
            assert np.array_equal(after[0].data, pixels)
            assert "DATAMIN" in after[0].header
    assert fits.getheader("fscale.fits")["BSCALE"] == 0.5


def test_nan_and_inf_are_left_out_and_said_so(work):
    """IRAF reads NaN and Inf as 0, and then minmax says the minimum of a frame
    of a thousand counts is 0."""
    a = list_images.small() + 1000
    a[1, 1], a[0, 2] = np.nan, np.inf
    fits.writeto("n.fits", a)
    result = get_task("minmax").run(images=["n.fits"])
    assert result.data["minval"] > 1000
    assert "2 pixels" in str(result.warnings[0])
    result = get_task("listpixels").run(images=["n.fits[1:3,1:2]"])
    assert result.stdout[2].endswith("  INDEF") and result.stdout[4].endswith("  INDEF")
    assert "2 pixels" in str(result.warnings[0])


def test_no_pixel_with_a_value(work):
    fits.writeto("n.fits", np.full((2, 2), np.nan, np.float32))
    result = get_task("minmax").run(images=["n.fits"])
    assert result.stdout == ["    n.fits  INDEF  INDEF"]
    assert "DATAMIN" not in fits.getheader("n.fits")


def test_listpixels_refuses_a_missing_image(work):
    with pytest.raises(ValueError, match="Cannot open image"):
        get_task("listpixels").run(images=["nothere.fits"])


def test_a_format_it_cannot_read_is_refused():
    with pytest.raises(ValueError, match="%8.2f"):
        iraffmt.Format("%E")


def test_multispec_world_is_refused_not_guessed(work):
    fits.writeto("m.fits", list_images.small(), fits.Header({"WAT0_001": "system=multispec"}))
    with pytest.raises(ValueError, match="multispec"):
        get_task("listpixels").run(images=["m.fits"], wcs="world")


@pytest.mark.parametrize("task, params", [
    ("imcopy", {"input": ["f.fits"], "output": "out.fits"}),
    ("imarith", {"operand1": "f.fits", "op": "*", "operand2": "2", "result": "out.fits"}),
    ("imshift", {"input": ["f.fits"], "output": "out.fits", "xshift": 1.0, "yshift": 0.0}),
    ("imcombine", {"input": ["f.fits", "f.fits"], "output": "out.fits"}),
])
def test_a_new_image_does_not_inherit_the_limits(work, task, params):
    """A header copied with its DATAMAX would have minmax report the old pixels'."""
    get_task("minmax").run(images=["f.fits"])
    assert "DATAMAX" in fits.getheader("f.fits")
    get_task(task).run(**params)
    assert "DATAMAX" not in fits.getheader("out.fits")
