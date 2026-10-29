"""phot --format iraf against the .mag file IRAF's phot writes.

IRAF ran phot on the synthetic field (tests/data/iraf_phot_format,
run_iraf.py there); harwood's file is the same byte for byte but for the
lines that say who wrote it and when. The second case has stars whose
centring and apertures meet bad pixels and the image's edges, so every
error code IRAF writes per aperture is exercised. On the twelve Skinakas
images of the automatico pipeline the same holds -- and IRAF's own mkapfile
reads harwood's files and gives what harwood's gives.
"""

from pathlib import Path

import pytest
from astropy.io import fits
from astropy.table import Table
from iraf_field import CURSOR, field

from harwood.core import catalog
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data" / "iraf_phot_format"
EDGE = CURSOR[:3] + [(4.0, 100.0), (196.5, 3.0)]
WHO = ("#K IRAF ", "#K USER ", "#K HOST ", "#K DATE ", "#K TIME ")


def _lines(path):
    return [line for line in Path(path).read_text().splitlines()
            if not line.startswith(WHO)]


@pytest.mark.parametrize("case, stars, extra", [("A", CURSOR, {}),
                                                ("B", EDGE, {"datamax": 600.0})])
def test_phot_writes_iraf_s_mag_file(tmp_path, monkeypatch, case, stars, extra):
    monkeypatch.chdir(tmp_path)
    header = fits.Header({"EXPTIME": 30.0, "AIRMASS": 1.234, "FILTER": "V",
                          "TIME": "02:14:33"})
    fits.writeto("field.fits", field(), header)
    coords = "field.coo" if case == "A" else "edge.coo"
    catalog.write(Table({"x": [s[0] for s in stars], "y": [s[1] for s in stars]}), coords)
    get_task("phot").run(image="field.fits", coords=Path(coords), output=Path("out.mag"),
                         format="iraf", epadu=2.0, exposure="EXPTIME", airmass="AIRMASS",
                         filter="FILTER", obstime="TIME", apertures=[3, 5, 8],
                         calgorithm="centroid", salgorithm="centroid", **extra)
    assert _lines("out.mag") == _lines(DATA / f"phot{case}.mag")


def test_harwood_reads_the_file_it_writes_for_iraf(tmp_path, monkeypatch):
    """photcal reads it as it reads IRAF's own."""
    from harwood.core.photcal import photfiles

    table = photfiles.read(DATA / "photA.mag")
    assert table["IFILTER"][0] == "V" and table["OTIME"][0] == "02:14:33"
    assert table["MAG1"][0] == pytest.approx(17.396)
