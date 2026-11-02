"""setjd against IRAF's own output (tests/data/iraf_setjd.json): the Julian
date of mid-exposure, the heliocentric one by IRAF's approximate Sun, and the
local Julian day, for the headers of the 1992 course frames."""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core.registry import get_task

DATA = json.loads((Path(__file__).parent / "data" / "iraf_setjd.json").read_text(
        encoding="utf-8"))


@pytest.fixture
def frames(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, case in DATA["cases"].items():
        header = fits.Header()
        for key, value in case["header"].items():
            header[key] = value
        fits.PrimaryHDU(np.zeros((4, 4), np.float32), header).writeto(f"{name}.fits")
    return tmp_path


def test_setjd_writes_irafs_dates(frames):
    get_task("setjd").run(images=["sp00*.fits"], observatory="kpno")
    for name, case in DATA["cases"].items():
        header = fits.getheader(f"{name}.fits")
        assert header["JD"] == pytest.approx(case["JD"], abs=1e-8), name
        assert header["HJD"] == pytest.approx(case["HJD"], abs=1e-8), name
        assert header["LJD"] == case["LJD"], name


def test_a_night_keeps_one_local_julian_day_across_midnight(tmp_path, monkeypatch):
    """LJD is int(JD - timezone/24): it changes at local noon, so an evening
    and the morning after are one night. harwood used to add half a day,
    which split every night at local midnight."""
    monkeypatch.chdir(tmp_path)
    for name, date, ut in (("evening", "1992-01-25", "02:00:00"),    # 19:00 local
                           ("morning", "1992-01-25", "12:00:00")):   # 05:00 local
        fits.PrimaryHDU(np.zeros((2, 2), np.float32),
                        fits.Header({"DATE-OBS": date, "UT": ut,
                                     "EXPTIME": 0.0})).writeto(f"{name}.fits")
    rows = get_task("setjd").run(images=["evening.fits", "morning.fits"],
                                 observatory="kpno", hjd="", listonly=True).data["rows"]
    assert rows[0]["ljd"] == rows[1]["ljd"]


def test_empty_keywords_are_not_written(frames):
    get_task("setjd").run(images=["sp0019.fits"], observatory="kpno", hjd="", ljd="")
    header = fits.getheader("sp0019.fits")
    assert "JD" in header and "HJD" not in header and "LJD" not in header
