"""fixpix against IRAF, and the two ways into it.

tests/data/iraf_fixpix_*.txt hold every pixel IRAF 2.18's proto.fixpix
changed in iraf_field.fixpix_case() -- a bad column, a bad row, a block, a
hot pixel, stubs at two edges, an L, and two blocks whose mask values force
the direction. harwood has to change exactly those pixels, to exactly those
values.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import fixpix_case

import harwood
from harwood.core import fixpix
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data"


def _iraf(label):
    rows = np.loadtxt(DATA / f"iraf_fixpix_{label}.txt")
    return rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 2]


@pytest.mark.parametrize("label, codes", [("default", {}),
                                          ("linterp5_cinterp7",
                                           {"linterp": 5, "cinterp": 7})])
def test_fixpix_changes_what_iraf_changes_to_what_iraf_writes(label, codes):
    image, mask = fixpix_case()
    fixed, _ = fixpix.fix(image, mask, **codes)
    x, y, value = _iraf(label)

    changed = np.argwhere(fixed != image)
    assert sorted(map(tuple, changed)) == sorted(zip(y - 1, x - 1, strict=True))
    assert fixed[y - 1, x - 1] == pytest.approx(value, abs=1e-4)


def test_a_bad_row_is_fixed_along_the_columns():
    """The case the old ccdproc step got wrong: it always went along x."""
    yy = np.arange(30.0)[:, None] * np.ones((1, 30))
    mask = np.zeros((30, 30), int)
    mask[12, 5:25] = 1
    fixed, summary = fixpix.fix(yy, mask)
    assert summary["along columns"] == 20
    assert fixed[12, 10] == pytest.approx(12.0), "the gradient runs along y"


def test_integer_images_are_rounded_as_iraf_writes_them():
    image = np.array([[10, 0, 13]], dtype=np.int16)
    fixed, _ = fixpix.fix(image, np.array([[0, 1, 0]]))
    assert fixed.dtype == np.int16 and fixed[0, 1] == 12          # 11.5 -> 12


def test_the_same_code_must_not_mean_both_directions():
    with pytest.raises(ValueError) as raised:
        fixpix.directions(np.ones((3, 3), int), linterp=2, cinterp=2)
    assert raised.value.fix


def test_the_task_takes_a_text_mask_or_the_bpm_keyword(tmp_path):
    from astropy.io import fits

    image, mask = fixpix_case()
    region = np.zeros_like(mask)
    region[5:35, 10] = 1
    fits.writeto(tmp_path / "bpm.fits", region)
    header = fits.Header({"BPM": "bpm.fits"})
    fits.writeto(tmp_path / "frame.fits", image, header)
    (tmp_path / "bad.txt").write_text("11 11 6 35   # the bad column\n", encoding="utf-8")

    by_keyword = get_task("fixpix").run(images=[str(tmp_path / "frame.fits")],
                                        masks=["BPM"], output=str(tmp_path / "k_"))
    by_text = get_task("fixpix").run(images=[str(tmp_path / "frame.fits")],
                                     masks=[str(tmp_path / "bad.txt")],
                                     output=str(tmp_path / "t_"))
    one = fits.getdata(by_keyword.outputs[0])
    two = fits.getdata(by_text.outputs[0])
    assert np.array_equal(one, two)
    assert by_keyword.data["rows"][0]["pixels"] == 30
    assert "fixpix" in str(fits.getheader(by_keyword.outputs[0])["HISTORY"])
    untouched = fits.getdata(tmp_path / "frame.fits")
    assert np.array_equal(untouched, image), "the input is not modified"


# ------------------------------------------- verbose, pixels and every mask


LISTING = json.loads((Path(__file__).parent / "data" / "iraf_fixpix_listing.json").read_text(
        encoding="utf-8"))


@pytest.mark.parametrize("tag", list(LISTING["cases"]))
def test_fixpix_lists_and_fixes_as_iraf(tag, tmp_path, monkeypatch):
    """The same frame and mask as FITS, as IRAF's own .pl and as a region
    file; what IRAF printed with verbose+ pixels+, and the pixels it left."""
    import shutil

    from astropy.io import fits

    case = LISTING["cases"][tag]
    monkeypatch.chdir(tmp_path)
    dtype = np.float32 if case["image"] == "real" else np.int16
    fits.PrimaryHDU(np.array(LISTING[case["image"]], dtype)).writeto(f"{tag}.fits")
    fits.PrimaryHDU(np.array(LISTING["mask"], np.int16)).writeto("mask.fits")
    shutil.copy(Path(__file__).parent / "data" / "fixpix_mask.pl", "mask.pl")
    Path("regions.txt").write_text(LISTING["regions"], encoding="utf-8")
    given = {key: value == "yes" if value in ("yes", "no") else value
             for key, value in case["params"].items()}
    params = {"verbose": True, "pixels": True, **given}
    result = harwood.run("fixpix", images=[f"{tag}.fits"], masks=[case["mask"]],
                         output=str(tmp_path / "out"), **params)
    if params["verbose"]:
        assert result.data["text"] == case["stdout"]
    else:
        assert "text" not in result.data
    fixed = fits.getdata(result.outputs[0])
    assert np.array_equal(fixed, np.array(case["fixed"], dtype))
    assert "Bad pixel file is " + case["mask"] in fits.getheader(result.outputs[0])["FIXPIX"]


def test_fixpix_takes_the_dq_and_keeps_the_reason(tmp_path, monkeypatch):
    """--masks DQ fixes what the image's own DQ flags; a fixed pixel keeps its
    reason, one an outside mask named becomes a bad pixel, and the variance
    is interpolated as the value is."""
    from harwood.core import dq, mef
    from harwood.core.registry import get_task

    monkeypatch.chdir(tmp_path)
    sci = np.tile(np.arange(10.0) * 10.0, (6, 1))
    var = np.tile(np.arange(10.0) + 1.0, (6, 1))
    quality = dq.empty(sci.shape)
    quality[2, 4] = dq.COSMIC_RAY
    sci[2, 4] = 9999.0
    mef.write(tmp_path / "f.fits", mef.Image(fits.Header({"OBJECT": "f"}),
                                             [mef.Plane(sci, var, quality)]))
    get_task("fixpix").run(images=["f.fits"], masks=["DQ"], output="d_")
    fixed = mef.read(tmp_path / "d_f.fits").plane
    assert fixed.sci[2, 4] == pytest.approx(40.0)            # along its line
    assert fixed.var[2, 4] == pytest.approx(5.0)
    assert fixed.dq[2, 4] == dq.COSMIC_RAY, "the reason stays"

    get_task("fixpix").run(images=["f.fits"], masks=["DQ"], output="named.fits")
    np.testing.assert_array_equal(mef.read(tmp_path / "named.fits").plane.sci, fixed.sci)
    (tmp_path / "outs").write_text("listed.fits\n", encoding="utf-8")
    get_task("fixpix").run(images=["f.fits"], masks=["DQ"], output="@outs")
    assert (tmp_path / "listed.fits").exists()

    get_task("fixpix").run(images=["d_f.fits"], masks=["[7,*]"], output="c_")
    again = mef.read(tmp_path / "c_d_f.fits").plane
    assert np.all(again.dq[:, 6] == dq.BAD_PIXEL)
    assert again.dq[2, 4] == dq.COSMIC_RAY
