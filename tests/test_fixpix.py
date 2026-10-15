"""fixpix against IRAF, and the two ways into it.

tests/data/iraf_fixpix_*.txt hold every pixel IRAF 2.18's proto.fixpix
changed in iraf_field.fixpix_case() -- a bad column, a bad row, a block, a
hot pixel, stubs at two edges, an L, and two blocks whose mask values force
the direction. harwood has to change exactly those pixels, to exactly those
values.
"""

from pathlib import Path

import numpy as np
import pytest
from iraf_field import fixpix_case

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
    (tmp_path / "bad.txt").write_text("11 11 6 35   # the bad column\n")

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
