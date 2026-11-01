"""precess and artdata against IRAF 2.18, bit by bit.

tests/data/iraf_artdata holds what IRAF wrote (run.py, cases.py) for the files of
art_images.py. mknoise and mkobjects are random, and for a seed they must be the same
numbers: that takes IRAF's own generator and templates, in its rounding.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import artdata, iraffmt
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_artdata"
GOLD = json.loads((HERE / "artdata.json").read_text())

_load = importlib.util.spec_from_file_location("art_images", HERE / "art_images.py")
art_images = importlib.util.module_from_spec(_load)
_load.loader.exec_module(art_images)
_load = importlib.util.spec_from_file_location("art_cases", HERE / "cases.py")
cases = importlib.util.module_from_spec(_load)
_load.loader.exec_module(cases)


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    where = tmp_path_factory.mktemp("art") / "fresh"
    art_images.write(where)
    return where


@pytest.fixture
def work(fresh, tmp_path, monkeypatch):
    shutil.copytree(fresh, tmp_path / "w")
    monkeypatch.chdir(tmp_path / "w")
    return tmp_path / "w"


def _same(name, expected):
    with fits.open(name, do_not_scale_image_data=True) as hdul:
        assert hdul[0].header["BITPIX"] == expected["bitpix"]
    data = fits.getdata(name).astype(float)
    assert list(data.shape)[::-1] == expected["shape"]
    pixels = np.array(expected["pixels"]).reshape(data.shape)
    assert np.array_equal(data, pixels, equal_nan=True)
    header = fits.getheader(name)
    for key, value in expected["keys"].items():
        if key == "INSTRUME":  # run.py cuts it at the /
            assert header[key] == "IRAF/ARTDATA"
        elif isinstance(header[key], str):
            assert header[key].strip() == value
        else:
            assert header[key] == pytest.approx(float(value), abs=1e-4)


def _run(task, table, case):
    params = dict(table[case]["params"])
    source = params.pop("input", "new.fits")
    target = params.pop("output", None)
    for key in ("poisson", "comments"):
        if key in params:
            params[key] = params[key] == "yes"
    for key in ("cosrays", "objects"):
        if key in params:
            params[key] = [params[key]]
    result = get_task(task).run(input=[source], output=[target] if target else [], **params)
    return target or source, result


@pytest.mark.parametrize("case", list(cases.PRECESS))
def test_precess_prints_what_iraf_prints(work, case):
    params = dict(cases.PRECESS[case])
    result = get_task("precess").run(input=[params.pop("input")], **params)
    printed = GOLD["precess"][case]["stdout"]
    assert result.stdout == [line for line in printed if not line.startswith("Bad")]
    assert len(result.warnings) == sum(line.startswith("Bad") for line in printed)


@pytest.mark.parametrize("case", list(GOLD["mkpattern"]))
def test_mkpattern_writes_the_pixels_iraf_writes(work, case):
    name, _ = _run("mkpattern", GOLD["mkpattern"], case)
    _same(name.split("[")[0], GOLD["mkpattern"][case])


@pytest.mark.parametrize("case", list(GOLD["mknoise"]))
def test_mknoise_writes_the_numbers_iraf_writes(work, case):
    name, _ = _run("mknoise", GOLD["mknoise"], case)
    _same(name, GOLD["mknoise"][case])
    written = GOLD["mknoise"][case].get("written")
    if written is not None:
        assert Path("new.txt").read_text() == written


@pytest.mark.parametrize("case", list(GOLD["mkobjects"]))
def test_mkobjects_writes_the_numbers_iraf_writes(work, case):
    name, _ = _run("mkobjects", GOLD["mkobjects"], case)
    _same(name, GOLD["mkobjects"][case])


@pytest.mark.parametrize(
    "task, case", [("mknoise", "rays_random"), ("mkobjects", "std_header")]
)
def test_the_comments_are_iraf_s(work, task, case):
    name, _ = _run(task, GOLD[task], case)
    mine = [str(c) for c in fits.getheader(name)["COMMENT"]]
    theirs = GOLD[task][case]["comments"]
    assert [c.strip() for c in mine[1:]] == [c.strip() for c in theirs[1:]]


@pytest.mark.parametrize("case", list(GOLD["starlist"]))
def test_starlist_writes_the_lines_iraf_writes(work, case):
    expected = GOLD["starlist"][case]
    get_task("starlist").run(starlist="list.txt", **expected["params"])
    mine = Path("list.txt").read_text().splitlines()
    theirs = expected["text"].splitlines()
    assert mine[1:] == theirs[1:]  # the first has the time


def test_the_generator_is_iraf_s():
    """iraf.urand (nlines=3, ncols=2, ndigits=7, seed=1) printed these."""
    state, _ = artdata.new_state(1)
    values = [iraffmt.printf(float(artdata.urand(state)), "g", 7) for _ in range(6)]
    assert values == [
        "0.7087448",
        "0.4249998",
        "0.9906267",
        "0.5149138",
        "0.4218696",
        "0.648596",
    ]


def test_an_object_of_an_unknown_kind_is_said_and_passed_over(work):
    Path("o.txt").write_text("10 10 7 nosuch 2 0.5 0\n")
    result = get_task("mkobjects").run(
        input=["n.fits"], objects=["o.txt"], ncols=20, nlines=20, background=0.0, header=""
    )
    assert not fits.getdata("n.fits").any()
    assert "nosuch" in str(result.warnings[0])
