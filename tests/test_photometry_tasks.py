"""center, fitsky, xyxymatch and ccdproc's fixpix, run as tasks.

Their arithmetic is checked against IRAF elsewhere; these check the part a
user meets -- files in, files out, the columns named as IRAF names them,
flags turned into warnings that say what to do.
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from iraf_field import CURSOR, field

from harwood.core import catalog
from harwood.core.registry import get_task


@pytest.fixture
def frame(tmp_path):
    fits.writeto(tmp_path / "field.fits", field())
    coords = Table({"x": [c[0] for c in CURSOR], "y": [c[1] for c in CURSOR]})
    catalog.write(coords, tmp_path / "field.coo")
    return tmp_path


def test_center_writes_the_new_centres_as_x_and_y(frame):
    result = get_task("center").run(image=str(frame / "field.fits"),
                                     coords=frame / "field.coo",
                                     output=frame / "field.ctr")
    table = catalog.read(frame / "field.ctr")
    assert {"xinit", "yinit", "x", "y", "xshift", "xerr", "cier"} <= set(table.colnames)
    assert table["x"][0] == pytest.approx(40.313, abs=1e-3)       # IRAF's XCENTER
    assert np.all(table["xinit"] == [c[0] for c in CURSOR])
    assert result.data["flagged"] == 0
    # The output feeds phot with calgorithm=none, as IRAF's .ctr feeds phot.
    x, y = catalog.coordinates(table, frame / "field.ctr")
    assert x[0] == pytest.approx(40.313, abs=1e-3)


def test_center_turns_a_flag_into_a_warning_with_a_fix(frame):
    result = get_task("center").run(image=str(frame / "field.fits"),
                                    coords=frame / "field.coo",
                                    output=frame / "tight.ctr", maxshift=0.5)
    assert result.warnings
    warning = result.warnings[0]
    assert "107" in warning.message and "--maxshift" in warning.fix


def test_fitsky_reports_iraf_s_sky_without_recentring(frame):
    get_task("fitsky").run(image=str(frame / "field.fits"),
                           coords=frame / "field.coo", output=frame / "field.sky",
                           salgorithm="mode")
    table = catalog.read(frame / "field.sky")
    assert {"msky", "stdev", "sskew", "nsky", "nsrej", "sier"} <= set(table.colnames)
    assert table["x"][0] == CURSOR[0][0], "fitsky measures where it is told"
    assert np.all(table["sier"] == 0)


def test_phot_columns_and_magnitudes_per_second_when_asked(frame):
    with fits.open(frame / "field.fits", mode="update") as hdul:
        hdul[0].header["EXPTIME"] = 10.0
    get_task("phot").run(image=str(frame / "field.fits"), coords=frame / "field.coo",
                         output=frame / "a.mag", epadu=2.0, apertures=[3, 5, 8])
    get_task("phot").run(image=str(frame / "field.fits"), coords=frame / "field.coo",
                         output=frame / "b.mag", epadu=2.0, apertures=[3, 5, 8],
                         exposure="EXPTIME")
    a, b = catalog.read(frame / "a.mag"), catalog.read(frame / "b.mag")
    for name in ("sum_3", "area_5", "flux_8", "mag_3", "merr_5", "pier_8", "msky",
                 "nsrej", "cier", "itime"):
        assert name in a.colnames, name
    assert a["mag_3"][0] == pytest.approx(13.703, abs=1e-3)       # IRAF, itime 1
    assert b["mag_3"][0] - a["mag_3"][0] == pytest.approx(2.5, abs=1e-6)


def test_xyxymatch_recovers_a_known_shift(tmp_path):
    rng = np.random.default_rng(4)
    x, y = rng.uniform(20, 480, 40), rng.uniform(20, 480, 40)
    catalog.write(Table({"x": x, "y": y}), tmp_path / "ref.coo")
    catalog.write(Table({"x": x + 3.25, "y": y - 1.5}), tmp_path / "in.coo")
    result = get_task("xyxymatch").run(input=tmp_path / "in.coo",
                                       reference=tmp_path / "ref.coo",
                                       output=tmp_path / "pairs.txt",
                                       matching="triangles")
    assert result.data["pairs"] == 40
    assert abs(result.data["xshift"]) == pytest.approx(3.25, abs=1e-3)
    assert abs(result.data["yshift"]) == pytest.approx(1.5, abs=1e-3)
    assert result.data["rms"] < 1e-3
    pairs = catalog.read(tmp_path / "pairs.txt")
    assert len(pairs) == 40
    assert np.allclose(pairs["xin"] - pairs["xref"], 3.25)


def test_ccdproc_fixpix_is_iraf_s_fixpix(tmp_path):
    """ccdproc's step and the fixpix task run the same code: a bad row is
    fixed along the columns by both."""
    from harwood.core import fixpix

    yy = np.tile(np.arange(60.0)[:, None] * 3 + 100, (1, 50)).astype(np.float32)
    mask = np.zeros((60, 50), np.int16)
    mask[30, 5:45] = 1
    fits.writeto(tmp_path / "raw.fits", yy)
    fits.writeto(tmp_path / "bpm.fits", mask)
    result = get_task("ccdproc").run(input=[str(tmp_path / "raw.fits")],
                                     output=tmp_path / "fixed.fits", fixpix=True,
                                     bpm=tmp_path / "bpm.fits")
    fixed = fits.getdata(tmp_path / "fixed.fits")
    assert fixed[30, 20] == pytest.approx(100 + 3 * 30), "along the column"
    assert np.array_equal(fixed, fixpix.fix(yy, mask)[0])
    assert "FIXPIX" in fits.getheader(tmp_path / "fixed.fits")
    step, = [s for s in result.steps if s.action == "fixpix"]
    assert step.measured["along columns"] == 40
