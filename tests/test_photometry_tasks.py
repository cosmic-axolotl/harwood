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


def test_phot_takes_every_iraf_algorithm_by_name(tmp_path):
    """The options reach the arithmetic: the dirty field with gauss centring
    and a grown, clipped mode sky gives IRAF's own numbers (s_mode_clip in
    iraf_apphot_algorithms.json is centroid centring; the sky is the sky)."""
    import json
    from pathlib import Path

    from iraf_field import dirty_field

    fits.writeto(tmp_path / "dirty.fits", dirty_field())
    catalog.write(Table({"x": [c[0] for c in CURSOR], "y": [c[1] for c in CURSOR]}),
                  tmp_path / "dirty.coo")
    params = dict(image=str(tmp_path / "dirty.fits"), coords=tmp_path / "dirty.coo",
                  epadu=2.0, apertures=[3, 5, 8], salgorithm="mode", sloclip=5.0,
                  shiclip=10.0, rgrow=1.5)
    task = get_task("phot")
    plan = task.plan(task.Params(**params))
    assert "clipping 5%/10%, growing rejections by 1.5" in plan[2].detail
    task.run(output=tmp_path / "dirty.mag", **params)
    table = catalog.read(tmp_path / "dirty.mag")
    reference = json.loads((Path(__file__).parent / "data"
                            / "iraf_apphot_algorithms.json").read_text())
    rows = reference["cases"]["s_mode_clip"]["rows"]
    assert list(table["msky"]) == pytest.approx([r[7] for r in rows], rel=6e-7)
    assert list(table["nsrej"]) == [r[11] for r in rows]


def test_xyxymatch_recovers_a_known_shift(tmp_path):
    rng = np.random.default_rng(4)
    x, y = rng.uniform(20, 480, 40), rng.uniform(20, 480, 40)
    catalog.write(Table({"x": x, "y": y}), tmp_path / "ref.coo")
    catalog.write(Table({"x": x + 3.25, "y": y - 1.5}), tmp_path / "in.coo")
    result = get_task("xyxymatch").run(input=tmp_path / "in.coo",
                                       reference=tmp_path / "ref.coo",
                                       output=tmp_path / "pairs.txt",
                                       matching="triangles", separation=0.0)
    # separation=0: IRAF's default of 9 px would merge two close pairs here.
    assert result.data["pairs"] == 40
    # IRAF's transformation runs from the input to the reference.
    assert result.data["xshift"] == pytest.approx(-3.25, abs=1e-3)
    assert result.data["yshift"] == pytest.approx(1.5, abs=1e-3)
    rows = [line.split() for line in (tmp_path / "pairs.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    assert len(rows) == 40
    assert np.allclose([float(r[2]) - float(r[0]) for r in rows], 3.25, atol=1e-3)


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


def test_histplot_and_radplot_read_the_graphics_cursor(frame, monkeypatch):
    """IRAF's histplot and radplot: the sky marked by hand -- here from a
    gcommands file, each star reading until q, the end of the file leaving
    the first guess (the median for histplot, the mean for radplot). Without
    a cursor every star is flagged 208, as IRAF flags it in batch."""
    monkeypatch.setenv("HARWOOD_DISPLAY", "headless")
    run = get_task("fitsky").run
    common = dict(image=str(frame / "field.fits"), coords=frame / "field.coo")
    none = catalog.read(run(salgorithm="histplot", output=frame / "a.sky",
                            **common).outputs[0])
    assert set(none["sier"]) == {208}
    (frame / "hcur").write_text("0 0 1 m\n105.25 5 1 m\n0 0 1 q\n0 0 1 q\n")
    marked = catalog.read(run(salgorithm="histplot", gcommands=str(frame / "hcur"),
                              output=frame / "h.sky", **common).outputs[0])
    assert set(marked["sier"]) == {0}
    histogram_edge = marked["msky"][0]
    assert histogram_edge != 0, "a mark below the histogram is clamped to it"
    from harwood.core import apphot, fitsky

    a, _, _ = fitsky.sky_buffer(fits.getdata(frame / "field.fits"), CURSOR[1][0],
                                CURSOR[1][1], 10.0, 10.0, apphot.DataPars())
    middle = fitsky._medr(a, np.arange(a.n + 1), a.n)        # apmedr, the first guess
    assert marked["msky"][1] == pytest.approx(float(middle)), "q at once keeps it"
    (frame / "rcur").write_text("3 101.5 1 m\n0 0 1 q\n")
    radial = catalog.read(run(salgorithm="radplot", gcommands=str(frame / "rcur"),
                              output=frame / "r.sky", **common).outputs[0])
    assert radial["msky"][0] == pytest.approx(101.5)


def test_phot_reads_the_sky_from_a_file(frame):
    """salgorithm=file: phot's skyfile, one 'x y msky stdev sskew nsky nsrej'
    line per star; fitsky has no sky file and flags 209, as in IRAF."""
    lines = [f"{c[0]} {c[1]} 100.0 2.5 0.1 300 4" for c in CURSOR]
    (frame / "skies").write_text("\n".join(lines[:-1]) + "\n")
    result = get_task("phot").run(image=str(frame / "field.fits"),
                                  coords=frame / "field.coo", salgorithm="file",
                                  skyfile=str(frame / "skies"), output=frame / "f.mag")
    table = catalog.read(result.outputs[0])
    assert table["msky"][0] == pytest.approx(100.0) and table["nsky"][0] == 300
    assert table["sier"][-1] == 210, "no line left for the last star"
    fitsky = catalog.read(get_task("fitsky").run(
        image=str(frame / "field.fits"), coords=frame / "field.coo", salgorithm="file",
        output=frame / "f.sky").outputs[0])
    assert set(fitsky["sier"]) == {209}


def test_absorption_features_centre_as_iraf_centres_them():
    """emission=no: the box is turned over about its top before centring,
    as apfitcen does; every algorithm, with and without a cthreshold, lands
    where IRAF's center does."""
    import json
    from pathlib import Path

    from harwood.core import apphot

    ref = json.loads((Path(__file__).parent / "data" / "iraf_center_absorption.json")
                     .read_text())
    yy, xx = np.indices((80, 90))
    data = 1000 + ((xx * 7 + yy * 13) % 11).astype(float)
    for cx, cy, depth in ((20.3, 30.6, 300), (55.8, 40.2, 150), (70.1, 65.4, 400)):
        data -= depth * np.exp(-((xx + 1 - cx) ** 2 + (yy + 1 - cy) ** 2) / (2 * 1.6 ** 2))
    data = data.astype(np.float32)
    for key, centres in ref["centres"].items():
        algorithm, threshold = key.split("/")
        for (x0, y0), (ix, iy) in zip(ref["starts"], centres, strict=True):
            c = apphot.center(data, x0, y0,
                              apphot.CenterPars(calgorithm=algorithm, cbox=7,
                                                cthreshold=float(threshold)),
                              apphot.DataPars(fwhmpsf=3.0, sigma=3.0, emission=False))
            assert (round(c.x, 3), round(c.y, 3)) == (ix, iy), key
