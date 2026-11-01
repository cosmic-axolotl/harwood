"""IRAF's .pl masks, and text2mask, badpiximage and ccdmask, against IRAF 2.18.

tests/data/iraf_masks holds what IRAF wrote (run.py, cases.py): the masks as its own
.pl files, which harwood must read and, for what is written line by line, write to
the same bytes but for the times in the title. ccdmask is the same to the byte on
frames small enough for one block of columns. Where it differs on purpose, a test
below says so.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import plio
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_masks"
GOLD = json.loads((HERE / "masks.json").read_text())

_load = importlib.util.spec_from_file_location("mask_images", HERE / "mask_images.py")
mask_images = importlib.util.module_from_spec(_load)
_load.loader.exec_module(mask_images)
_load = importlib.util.spec_from_file_location("mask_cases", HERE / "cases.py")
cases = importlib.util.module_from_spec(_load)
_load.loader.exec_module(cases)


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    where = tmp_path_factory.mktemp("masks") / "fresh"
    mask_images.write(where)
    return where


@pytest.fixture
def work(fresh, tmp_path, monkeypatch):
    shutil.copytree(fresh, tmp_path / "w")
    monkeypatch.chdir(tmp_path / "w")
    return tmp_path / "w"


def encoded(path) -> bytes:
    """A .pl without its title, which has the time."""
    raw = Path(path).read_bytes()
    _, length, _ = np.frombuffer(raw, ">i4", 3)
    return raw[12 + (length + 1) // 2 * 2:]


@pytest.mark.parametrize("case", [c for c in cases.TEXT2MASK if c != "regions"])
def test_text2mask_writes_the_bytes_iraf_writes(work, case):
    params = dict(cases.TEXT2MASK[case])
    get_task("text2mask").run(mask=f"t_{case}.pl", **params)
    assert np.array_equal(plio.read(f"t_{case}.pl")[0], plio.read(HERE / f"t_{case}.pl")[0])
    # IRAF's empty one has 504 words of what was in memory, and over 4094 columns its
    # way of writing a section cuts the zeros at 4095 where its other way cuts at 4094.
    if case not in ("empty", "wide"):
        assert encoded(f"t_{case}.pl") == encoded(HERE / f"t_{case}.pl")


def test_a_rectangle_of_several_lines_is_whole(work):
    """IRAF's text2mask keeps of it the first line and a little of the second: the
    short pixels go wrong on their way to a mask of ints. The region it asked for is
    what is here, and what it does keep is in it."""
    get_task("text2mask").run(mask="r.pl", **cases.TEXT2MASK["regions"])
    mine = plio.read("r.pl")[0]
    iraf = plio.read(HERE / "t_regions.pl")[0]
    assert (mine[6:9, 1:4] == 3).all()
    assert iraf[6:9, 1:4].tolist() == [[3, 3, 3], [0, 3, 0], [0, 0, 0]]
    assert (mine[iraf != 0] != 0).all()


def test_a_comma_ends_the_numbers(work):
    Path("c.txt").write_text("8, 9\n")
    get_task("text2mask").run(text="c.txt", mask="c.pl", ncols=12, nlines=12)
    assert not plio.read("c.pl")[0].any()


@pytest.mark.parametrize("case", [c for c in cases.BADPIX if not GOLD["badpix"][c]["error"]])
def test_badpiximage_writes_what_iraf_writes(work, case):
    expected = GOLD["badpix"][case]
    get_task("badpiximage").run(image=f"b_{case}.fits", **cases.BADPIX[case])
    header = fits.getheader(f"b_{case}.fits")
    assert fits.getdata(f"b_{case}.fits").ravel().tolist() == expected["pixels"]
    assert header["BITPIX"] == expected["bitpix"]
    assert header["OBJECT"] == expected["object"].strip("' ")
    assert header["EXPTIME"] == 30.0
    assert "DATAMAX" not in header


def test_untrimmed_coordinates_are_moved_by_the_trim(work):
    """IRAF means to, and dies: it reassigns the index of the [ and reads the section
    from somewhere else in the string. These are its formula's numbers."""
    assert "section" in GOLD["badpix"]["untrimmed"]["error"]
    get_task("badpiximage").run(image="u.fits", **cases.BADPIX["untrimmed"])
    bad = np.argwhere(fits.getdata("u.fits") == 0).tolist()
    # 12 12 in a trim starting at 11, 5: column 2, line 8; and the rest likewise.
    assert [7, 1] in bad and [4, 2] in bad and [4, 4] in bad
    assert not any(y < 0 or x < 0 for y, x in bad)
    get_task("badpiximage").run(image="u2.fits", **cases.BADPIX["untrimmed2"])
    assert np.argwhere(fits.getdata("u2.fits") == 0).tolist() == [[2, 1], [2, 2]]


@pytest.mark.parametrize("case", [c for c in cases.CCDMASK if c not in ("wide", "wider")])
def test_ccdmask_writes_the_bytes_iraf_writes(work, case):
    get_task("ccdmask").run(mask=f"c_{case}.pl", **cases.CCDMASK[case])
    assert encoded(f"c_{case}.pl") == encoded(HERE / f"c_{case}.pl")


@pytest.mark.parametrize("case, where", [("wide", (251, 204)), ("wider", (0, 367))])
def test_ccdmask_in_blocks_but_for_a_pixel_of_memory(work, case, where):
    """With blocks of columns, IRAF's second pass leaves the pixel after a run as the
    buffer had it: 0 nearly always, and here one 3 and one 17532. It is 0 here."""
    get_task("ccdmask").run(mask=f"c_{case}.pl", **cases.CCDMASK[case])
    mine, iraf = plio.read(f"c_{case}.pl")[0], plio.read(HERE / f"c_{case}.pl")[0]
    assert np.argwhere(mine != iraf).tolist() == [list(where)]
    assert mine[where] == 0


@pytest.mark.parametrize("case", list(cases.PLIO))
def test_a_mask_named_pl_is_written_as_iraf_writes_it(work, case):
    task, params = cases.PLIO[case]
    get_task(task).run(**({"input": [params["input"]], "output": params["output"]}
                          if task == "imcopy" else params))
    target = params["output"]
    assert encoded(target) == encoded(HERE / target)


def test_iraf_s_title_and_header_go_in_the_file(work):
    get_task("imcopy").run(input=["ramp.fits"], output="m.pl")
    _, header = plio.read("m.pl")
    assert header["TITLE"] == "ramp"
    assert header["EXPTIME"] == 12.0


def test_a_pl_is_an_image_to_every_task(work):
    shutil.copy(HERE / "t_lines.pl", "m.pl")
    mask = plio.read("m.pl")[0]
    assert get_task("minmax").run(images=["m.pl"], update=False).data["maxval"] == mask.max()
    get_task("imexpr").run(expr="a * 2", output="o.fits", a="m.pl")
    assert np.array_equal(fits.getdata("o.fits"), mask * 2)
    assert fits.getdata("o.fits").dtype.kind == "i"
    get_task("imcopy").run(input=["m.pl"], output="o2.fits")
    assert np.array_equal(fits.getdata("o2.fits"), mask)


def test_a_full_mask_reads_back(work):
    a = np.zeros((4, 5000), np.int64)
    a[1, 4990:] = 70000
    a[2, 1:4000] = 2
    a[3, ::2] = 5
    assert np.array_equal(plio.read(plio.write("x.pl", a))[0], a)


# IRAF stops at 17 arguments to a function, 8 vertices less a star's worth; not here.
MSK_BEYOND = {"r_polygon_star"}
MSK_REFUSED = {"real", "real_scalar", "unknown", "wrong_args", "polygon_few", "no_image"}
MSK_MEMORY = {"r_pie_none", "zero", "scalar"}     # no line was written: 504 words of memory


def _mskexpr(case, target):
    params = dict(cases.MSKEXPR[case])
    return get_task("mskexpr").run(masks=target, **params)


@pytest.fixture
def maskwork(work):
    for name in ("t_lines.pl", "t_small.pl"):
        shutil.copy(HERE / name, name)
    return work


@pytest.mark.parametrize("case", [c for c in cases.MSKEXPR
                                  if c not in MSK_REFUSED and c not in MSK_BEYOND])
def test_mskexpr_writes_the_bytes_iraf_writes(maskwork, case):
    target = f"x_{case}.pl"
    result = _mskexpr(case, target)
    assert np.array_equal(plio.read(target)[0], plio.read(HERE / target)[0])
    assert result.stdout == GOLD["mskexpr"][case]["stdout"]
    if case not in MSK_MEMORY:
        assert encoded(target) == encoded(HERE / target)


@pytest.mark.parametrize("case", sorted(MSK_REFUSED))
def test_mskexpr_refuses_what_iraf_refuses(maskwork, case):
    assert GOLD["mskexpr"][case]["error"]
    with pytest.raises(ValueError):
        _mskexpr(case, "x.pl")


def test_a_polygon_can_have_more_than_eight_vertices(maskwork):
    assert "too many function" in GOLD["mskexpr"]["r_polygon_star"]["error"]
    _mskexpr("r_polygon_star", "x.pl")
    mask = plio.read("x.pl")[0]
    assert mask[8, 11] == 1 and mask[0, 0] == 0


def test_a_mask_that_exists_is_left_as_it_is(maskwork):
    _mskexpr("r_circle", "x.pl")
    before = Path("x.pl").read_bytes()
    result = get_task("mskexpr").run(expr="1 > 0", masks="x", dims="4,4")
    assert result.stdout[-1] == "Mask x.pl already exists"
    assert Path("x.pl").read_bytes() == before
