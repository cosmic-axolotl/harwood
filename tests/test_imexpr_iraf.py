"""imexpr against IRAF 2.18, pixel by pixel.

tests/data/iraf_imexpr holds what IRAF wrote (run.py, cases.py) for some six
hundred expressions over the images of expr_images.py: every operator and
function on each pixel type, the types of constants, conditionals of scalars and
vectors, lengths that do not match, coordinates, output types, sections in and
out, and macros. The bits of each pixel must be IRAF's, with its BITPIX and shape,
and so must the keywords of the header it inherits. An expression IRAF refuses, or
that makes it die, is below.
"""

import importlib.util
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_imexpr"
CASES = json.loads((HERE / "imexpr.json").read_text(encoding="utf-8"))

_spec = importlib.util.spec_from_file_location("expr_images", HERE / "expr_images.py")
expr_images = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(expr_images)

# IRAF dies where there is a perfectly good answer, and here it is given.
IRAF_DIES = {
    # 64-bit integer images: "integer divide by zero" on opening them.
    "add_la_ib", "add_la_rb", "sub_la_ib", "sub_la_rb", "mul_la_ib", "mul_la_rb", "div_la_ib",
    "div_la_rb", "copy_long",
    # short() of a vector writes through the wrong pointer.
    "f_short_real", "f_short_double", "f_short_short", "f_short_int",
    # A segmentation violation on @"name".
    "direct_at",
    # NAXIS1 is not among the keywords imgftype knows; here a keyword is a keyword.
    "direct_naxis",
    # It takes a one-letter string for an operand, and goes looking for the image.
    "str_concat", "str_bool_b",
    # PyRAF, not IRAF: it took the quotes off.
    "f_quoted",
}
# IRAF's answer is what was in memory: the median of a scalar, the end of a line
# the expression did not reach.
NOT_A_NUMBER_OF_IRAFS = {"f_median_scalar", "into_section_shorter"}
FAILED = [c for c in CASES if CASES[c]["error"] and c not in IRAF_DIES]
SAME = [c for c in CASES if not CASES[c]["error"] and c not in NOT_A_NUMBER_OF_IRAFS]


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    where = tmp_path_factory.mktemp("imexpr") / "fresh"
    expr_images.write(where)
    shutil.copy(HERE / "db.dat", where)
    shutil.copy(HERE / "expr.txt", where)
    return where


@pytest.fixture
def work(fresh, tmp_path, monkeypatch):
    shutil.copytree(fresh, tmp_path / "w")
    monkeypatch.chdir(tmp_path / "w")
    return tmp_path / "w"


def _run(case):
    params = dict(CASES[case]["params"])
    params.pop("bwidth", None)
    params.pop("btype", None)
    if "rangecheck" in params:
        params["rangecheck"] = params["rangecheck"] == "yes"
    get_task("imexpr").run(**params)
    return params["output"].split("[")[0]


#: What Windows' libm gives a bit away from glibc's, as its CI found
OTHER_LIBM = {"f_cosh_double", "f_sinh_double", "f_tanh_double"}


@pytest.mark.parametrize("case", SAME)
def test_imexpr_writes_the_pixels_iraf_writes(work, case):
    expected = CASES[case]
    name = _run(case)
    with fits.open(name, do_not_scale_image_data=True) as hdul:
        assert hdul[0].header["BITPIX"] == expected["bitpix"]
        assert list(hdul[0].data.shape)[::-1] == expected["shape"]
    data = fits.getdata(name).astype(np.float64).ravel()
    pixels = [struct.unpack(">d", bytes.fromhex(x))[0] for x in expected["bits"]]
    if sys.platform == "win32" and case in OTHER_LIBM:
        # IRAF's bits are glibc's: Windows' libm is another, and a hyperbolic function can
        # come out a bit away there, which is as near as another libm can be held
        np.testing.assert_array_max_ulp(data, np.array(pixels), maxulp=1)
    else:
        assert np.array_equal(data, np.array(pixels), equal_nan=True)
    header = fits.getheader(name)
    for key, value in expected["cards"].items():
        have = header.get(key, 1.0 if key.startswith("LTM") else "")
        if isinstance(have, str):
            assert have.strip() == value
        else:
            assert have == pytest.approx(float(value), abs=1e-6), key


@pytest.mark.parametrize("case", FAILED)
def test_what_iraf_refuses_is_refused(work, case):
    with pytest.raises(ValueError):
        _run(case)


@pytest.mark.parametrize("case", sorted(IRAF_DIES))
def test_where_iraf_dies_there_is_an_answer(work, case):
    _run(case)
    assert fits.getdata(CASES[case]["params"]["output"]).size > 0


def test_the_answers_where_iraf_dies(work):
    get_task("imexpr").run(expr="short(a)", output="s.fits", a="re.fits")
    data = fits.getdata("s.fits")
    assert data.dtype.kind == "i" and data.dtype.itemsize == 2
    assert np.array_equal(data, fits.getdata("re.fits").astype(np.int16))
    get_task("imexpr").run(expr="a + 1", output="l.fits", a="la.fits")
    assert np.array_equal(fits.getdata("l.fits"), fits.getdata("la.fits") + 1)


def test_an_operand_can_be_named_like_itself(work):
    """IRAF takes a=a.fits for keyword fits of operand a."""
    shutil.copy("ra.fits", "a.fits")
    get_task("imexpr").run(expr="a + 1", output="o.fits", a="a.fits")
    assert np.array_equal(fits.getdata("o.fits"), fits.getdata("ra.fits") + 1)


def test_an_operand_needs_a_value(work):
    with pytest.raises(ValueError, match="operand b"):
        get_task("imexpr").run(expr="a + b", output="o.fits", a="ra.fits")


def test_input_and_output_cannot_be_the_same(work):
    with pytest.raises(ValueError, match="cannot be the same"):
        get_task("imexpr").run(expr="a + 1", output="ra.fits", a="ra.fits")


def test_nan_stays_nan(work):
    """IRAF reads a NaN as 0 and goes on, and a + 1 is 1 there."""
    a = expr_images.base()
    a[1, 1] = np.nan
    fits.writeto("n.fits", a)
    get_task("imexpr").run(expr="a + 1", output="o.fits", a="n.fits")
    assert np.isnan(fits.getdata("o.fits")[1, 1])


def test_the_output_has_no_limits_of_the_input(work):
    get_task("minmax").run(images=["ra.fits"])
    get_task("imexpr").run(expr="a * 2", output="o.fits", a="ra.fits")
    assert "DATAMAX" not in fits.getheader("o.fits")


def test_the_trail_is_in_the_header(work):
    get_task("imexpr").run(expr="a * 2", output="o.fits", a="ra.fits")
    assert "imexpr: a * 2" in str(fits.getheader("o.fits")["HISTORY"])


@pytest.mark.parametrize("expr, expected", [
    ("atan2(a, 2.0)", lambda a: np.arctan2(a.astype(float), 2.0).astype(np.float32)),
    ("mod(7, b)", lambda b: np.full_like(b, 7) % b),
])
def test_a_scalar_where_a_vector_could_go(work, expr, expected):
    """IRAF reads the scalar's memory as a vector, and corrupts its own or dies."""
    get_task("imexpr").run(expr=expr, output="o.fits", a="ra.fits", b="onedi.fits")
    source = fits.getdata("ra.fits" if "a," in expr else "onedi.fits")
    assert np.allclose(fits.getdata("o.fits"), expected(source))
