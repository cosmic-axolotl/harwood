"""ccdgroups against IRAF's own.

tests/data/iraf_ccdgroups.json holds the group files IRAF 2.18.1's ccdgroups
wrote for the seven headers below, grouped every way it can: by pointing
(one pair across RA 0h), by title, by date (with and without a time), by
type, by subset, and by date among the objects only. harwood must write the
same files with the same names in the same order.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core.registry import get_task

REFERENCE = json.loads((Path(__file__).parent / "data" / "iraf_ccdgroups.json")
                       .read_text(encoding="utf-8"))["files"]
HEADERS = [
    ("a1", "object", "V", "M92", "1992-03-05T03:10:00", "17:17:07.0", "+43:08:11"),
    ("a2", "object", "B", "M92", "1992-03-05T03:20:00", "17:17:07.5", "+43:08:20"),
    ("a3", "object", "V", "NGC 4147", "1992-03-05T05:00:00", "12:10:06.0", "+18:32:31"),
    ("a4", "flat", "V", "dome flat", "1992-03-06", "17:17:07.0", "+43:08:11"),
    ("a5", "zero", "", "bias", "1992-03-06T01:00:00", "00:00:00", "+00:00:00"),
    ("a6", "object", "V", "M92", "1992-03-06T03:00:00", "17:17:09.0", "+43:09:30"),
    ("a7", "dark", "", "dark 300", "1992-03-06T02:00:00", "23:59:59.0", "+00:00:10"),
]


@pytest.fixture
def night(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, kind, subset, title, date, ra, dec in HEADERS:
        header = fits.Header({"IMAGETYP": kind, "FILTERS": subset, "OBJECT": title,
                              "DATE-OBS": date, "RA": ra, "DEC": dec, "EXPTIME": 1.0})
        fits.writeto(f"{name}.fits", np.zeros((4, 4), np.float32), header)
    Path("inst.dat").write_text("subset\tFILTERS\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("group", ["position", "title", "date", "ccdtype", "subset"])
def test_ccdgroups_writes_iraf_s_files(night, group):
    get_task("ccdgroups").run(images=["a*.fits"], output=f"{group}_", group=group,
                              instrument="inst.dat")
    expected = {name: names for name, names in REFERENCE.items()
                if name.startswith(f"{group}_")}
    written = {path.name: path.read_text(encoding="utf-8").split()
               for path in sorted(night.glob(f"{group}_*"))}
    assert written == expected


def test_ccdtype_selects_before_grouping(night):
    get_task("ccdgroups").run(images=["a*.fits"], output="sel_", group="date",
                              ccdtype="object", instrument="inst.dat")
    assert {p.name: p.read_text(encoding="utf-8").split() for p in night.glob("sel_*")} == {
        name: names for name, names in REFERENCE.items() if name.startswith("sel_")}


def test_a_second_run_appends_and_says_so(night):
    run = dict(images=["a*.fits"], output="t_", group="ccdtype", instrument="inst.dat")
    get_task("ccdgroups").run(**run)
    result = get_task("ccdgroups").run(**run)
    assert Path("t_object").read_text(
            encoding="utf-8").split() == ["a1.fits", "a2.fits", "a3.fits",
                                                    "a6.fits"] * 2
    assert any("already exists" in w.message for w in result.warnings)
