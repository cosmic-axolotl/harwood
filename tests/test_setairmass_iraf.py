"""setairmass against IRAF's own output.

IRAF 2.18.1's setairmass was run on the headers of the 1992 course frames
(tests/data/iraf_setairmass.json): what it printed -- effective, beginning,
middle and end airmass -- and what it wrote, AIRMASS in single precision and
UTMIDDLE as a date and time. The headers are in the file, so this needs
neither IRAF nor the course data.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core.registry import get_task

DATA = json.loads((Path(__file__).parent / "data" / "iraf_setairmass.json").read_text())


@pytest.fixture
def frames(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, case in DATA["cases"].items():
        header = fits.Header()
        for key, value in case["header"].items():
            header[key] = value
        fits.PrimaryHDU(np.zeros((4, 4), np.float32), header).writeto(f"{name}.fits")
    return tmp_path


def test_setairmass_prints_irafs_airmasses(frames):
    result = get_task("setairmass").run(images=["sp00*.fits"], observatory="kpno",
                                        update=False)
    for row in result.data["rows"]:
        effective, begin, middle, end = DATA["cases"][row["image"][:6]]["printed"]
        assert (row["airmass"], row["begin"], row["middle"], row["end"]) == \
            pytest.approx((effective, begin, middle, end), abs=5e-5), row["image"]


def test_setairmass_writes_what_iraf_writes(frames):
    get_task("setairmass").run(images=["sp00*.fits"], observatory="kpno")
    for name, case in DATA["cases"].items():
        header = fits.getheader(f"{name}.fits")
        assert header["AIRMASS"] == pytest.approx(case["AIRMASS"], abs=1e-6), name
        assert header["UTMIDDLE"] == case["UTMIDDLE"], name


def test_the_time_stamp_and_the_airmass_written_follow_intype_and_outtype(frames):
    expected = DATA["intype_end_outtype_beginning"]
    get_task("setairmass").run(images=["sp0019.fits", "sp0024.fits"],
                               observatory="kpno", intype="end", outtype="beginning")
    for name, case in expected.items():
        header = fits.getheader(f"{name}.fits")
        assert header["AIRMASS"] == pytest.approx(case["AIRMASS"], abs=1e-6), name
        assert header["UTMIDDLE"] == case["UTMIDDLE"], name


def test_without_st_the_sidereal_time_is_worked_out_and_said(frames):
    """IRAF refuses a header with no ST; harwood works it out from the UT and
    the longitude (IRAF's ast_mst), and says so. On these frames the two
    sidereal times agree to the second the header was written to."""
    for name in DATA["cases"]:
        with fits.open(f"{name}.fits", mode="update") as hdus:
            del hdus[0].header["ST"]
    result = get_task("setairmass").run(images=["sp00*.fits"], observatory="kpno",
                                        update=False)
    assert any("no ST keyword" in str(w) for w in result.warnings)
    for row in result.data["rows"]:
        assert row["airmass"] == pytest.approx(
            DATA["cases"][row["image"][:6]]["printed"][0], abs=2e-3)
