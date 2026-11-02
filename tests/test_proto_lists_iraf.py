"""imextensions and mimstatistics against IRAF 2.18, line by line.

tests/data/iraf_proto_lists holds what IRAF printed (run.py, cases.py) for the
files of proto_images.py: extensions by index, name and version, and statistics
through masks, with limits and clipping, and the masks it writes. Where harwood
differs on purpose, a test below says so.
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

HERE = Path(__file__).parent / "data" / "iraf_proto_lists"
GOLD = json.loads((HERE / "proto.json").read_text(encoding="utf-8"))

_spec = importlib.util.spec_from_file_location("proto_images", HERE / "proto_images.py")
proto_images = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proto_images)

YES = {"yes": True, "no": False}
# IRAF, on these, is not what a list of image extensions should be; below.
IMEXT_DIFFERENT = {"tables", "tables_name", "ext_given_name", "ext_given_nosuch", "glob",
                   "noext"}
IMEXT = [c for c in GOLD["imext"]
         if c not in IMEXT_DIFFERENT and not GOLD["imext"][c]["error"]]
IMEXT_ERRORS = [c for c in GOLD["imext"] if GOLD["imext"][c]["error"]
                and c not in IMEXT_DIFFERENT]
MIMSTAT_REFUSED = {"lists_differ", "nofields", "missing_mask"}
MIMSTAT = [c for c in GOLD["mimstat"] if c not in MIMSTAT_REFUSED]


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    where = tmp_path_factory.mktemp("proto") / "fresh"
    proto_images.write(where)
    return where


@pytest.fixture
def work(fresh, tmp_path, monkeypatch):
    shutil.copytree(fresh, tmp_path / "w")
    monkeypatch.chdir(tmp_path / "w")
    return tmp_path / "w"


def _imext(params):
    params = {k: YES.get(v, v) for k, v in params.items()}
    params["input"] = [params["input"]]
    return get_task("imextensions").run(**params)


def _mimstat(params):
    out = {k: YES.get(v, v) for k, v in params.items()}
    out["images"] = [out["images"]]
    for key in ("lsigma", "usigma"):
        if out.get(key) == "INDEF":
            out[key] = None
    return get_task("mimstatistics").run(**out)


@pytest.mark.parametrize("case", IMEXT)
def test_imextensions_lists_what_iraf_lists(work, case):
    expected = GOLD["imext"][case]
    result = _imext(expected["params"])
    wanted = expected["stdout"]
    if expected["params"].get("output") == "list" and not wanted:
        wanted = [""]
    assert result.stdout == wanted
    assert result.data["nimages"] == expected["nimages"]


@pytest.mark.parametrize("case", IMEXT_ERRORS)
def test_imextensions_refuses_what_iraf_refuses(work, case):
    with pytest.raises(ValueError):
        _imext(GOLD["imext"][case]["params"])


def test_a_table_is_not_an_image_extension(work):
    """IRAF skips a binary table but lists an ASCII one."""
    assert GOLD["imext"]["tables"]["stdout"][2] == "tables.fits[4]"
    lines = _imext(GOLD["imext"]["tables"]["params"]).stdout
    assert lines == ["tables.fits[1]", "tables.fits[3]", "tables.fits[5]"]


def test_an_extension_that_is_not_there_is_not_listed(work):
    """IRAF lists every extension of the file for mef.fits[NOPE]."""
    assert len(GOLD["imext"]["ext_given_nosuch"]["stdout"]) == 8
    assert _imext(GOLD["imext"]["ext_given_nosuch"]["params"]).stdout == []


def test_a_template_expands_and_the_real_name_is_given(work):
    """IRAF takes [mw]* for a section of a name, and finds nothing; a name without
    .fits it gives back without it. Either way the list can go to any task."""
    assert GOLD["imext"]["glob"]["stdout"] == []
    lines = _imext(GOLD["imext"]["glob"]["params"]).stdout
    assert lines == ["mef.fits[1]", "withdata.fits[1]"]
    assert GOLD["imext"]["noext"]["stdout"] == ["mef[1]"]
    assert _imext(GOLD["imext"]["noext"]["params"]).stdout == ["mef.fits[1]"]


def test_a_name_and_version_are_read(work):
    """IRAF dies on mef.fits[SCI,2], integer divide by zero."""
    assert _imext(GOLD["imext"]["ext_given_name"]["params"]).stdout == ["mef.fits[2][SCI]"]


@pytest.mark.parametrize("case", MIMSTAT)
def test_mimstatistics_prints_what_iraf_prints(work, case):
    expected = GOLD["mimstat"][case]
    result = _mimstat(expected["params"])
    assert result.stdout == expected["stdout"]
    if "omask" in expected:
        name = Path(expected["params"]["omasks"]).with_suffix(".pl")
        assert plio.read(name)[0].ravel().tolist() == expected["omask"]


@pytest.mark.parametrize("case", sorted(MIMSTAT_REFUSED))
def test_mimstatistics_says_what_is_wrong(work, case):
    """IRAF prints a message and stops, or nothing, or dies; here all three say why."""
    with pytest.raises((ValueError, FileNotFoundError)):
        _mimstat(GOLD["mimstat"][case]["params"])


def test_a_nan_is_left_out(work):
    a = proto_images.frame()
    a[0, 0] = np.nan
    fits.writeto("n.fits", a)
    stats = _mimstat({"images": "n.fits"}).data["rows"][0]
    assert stats["npix"] == a.size - 1
