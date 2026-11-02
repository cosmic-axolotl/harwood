"""radprof against IRAF's own output.

IRAF 2.18 ran each case on the synthetic field (tests/data/iraf_radprof, run_iraf.py
there). Its .prf text holds the profile and its integral to seven figures and the FWHM
to three decimals, and harwood's agree to the last of them, with the error codes.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from iraf_field import CURSOR, field

from harwood.core import catalog, curfit, radprof
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data" / "iraf_radprof"
KEYS = json.loads((Path(__file__).parent / "data" / "iraf_radprof_keys" / "radprof_keys.json")
                  .read_text(encoding="utf-8"))
PARAMS = dict(epadu=2.0, apertures=[3.0, 5.0], cbox=5.0, annulus=10.0, dannulus=10.0,
              calgorithm="centroid", salgorithm="centroid")
EDGE = [*CURSOR[:2], (4.0, 100.0), (196.5, 3.0), (-50.0, 10.0), (100.0, 100.0)]
CASES = {"rA": (CURSOR, 12.0, 0.5, dict(order=5, kreject=3.0, nreject=1)),
         "rB": (CURSOR, 8.0, 0.25, dict(order=3, kreject=2.0, nreject=4)),
         "rC": (CURSOR, 15.0, 1.0, dict(order=8, kreject=0.0, nreject=0)),
         "rD": (EDGE, 12.0, 0.5, dict(order=5, kreject=3.0, nreject=1)),
         "rE": (CURSOR, 6.0, 0.5, dict(order=1, kreject=3.0, nreject=1))}


def _number(text):
    return float("nan") if text == "INDEF" else float(text)


def _iraf(path, napertures=2):
    """The stars of a .prf: its apertures, its FWHM line and its profile."""
    lines = [line.rstrip().rstrip("\\").rstrip("*").rstrip()
             for line in path.read_text(
                     encoding="utf-8").splitlines() if not line.startswith("#")]
    stars, i = [], 0
    while i < len(lines):
        i += 4
        apertures = [lines[i + k].split() for k in range(napertures)]
        i += napertures
        summary = lines[i].split()
        i += 1
        rows = []
        while i < len(lines) and not lines[i].startswith("field.fits"):
            rows.append([_number(v) for v in lines[i].split()])
            i += 1
        stars.append(dict(mag=[_number(a[4]) for a in apertures],
                          merr=[_number(a[5]) for a in apertures],
                          fwhm=_number(summary[0]), inorm=_number(summary[1]),
                          tinorm=_number(summary[2]), rier=int(summary[3]),
                          rows=np.array(rows)))
    return stars


@pytest.mark.parametrize("case", CASES)
def test_radprof_writes_what_iraf_writes(tmp_path, monkeypatch, case):
    stars, radius, step, settings = CASES[case]
    monkeypatch.chdir(tmp_path)
    fits.writeto("field.fits", field())
    catalog.write(Table({"x": [s[0] for s in stars], "y": [s[1] for s in stars]}), "c.coo")
    get_task("radprof").run(image="field.fits", radius=radius, step=step, coords=Path("c.coo"),
                            output=Path("out.prf"), **PARAMS, **settings)
    ours = catalog.read("out.prf")
    assert (DATA / f"{case}.err").read_text(encoding="utf-8") == ""
    theirs = _iraf(DATA / f"{case}.prf")
    assert len(ours) == len(theirs)
    for row, star in zip(ours, theirs, strict=True):
        assert row["rier"] == star["rier"]
        for k, suffix in enumerate(("_3", "_5")):
            assert np.isclose(row[f"mag{suffix}"], star["mag"][k], atol=6e-4, equal_nan=True)
            assert np.isclose(row[f"merr{suffix}"], star["merr"][k], atol=6e-4, equal_nan=True)
        if np.isnan(star["fwhm"]):
            assert np.isnan(row["pfwhm"])
            continue
        assert row["pfwhm"] == pytest.approx(star["fwhm"], abs=6e-4)
        assert row["inorm"] == pytest.approx(star["inorm"], rel=1e-6)
        assert row["tinorm"] == pytest.approx(star["tinorm"], rel=1e-6)
        assert np.allclose(row["intensity"], star["rows"][:, 1], atol=1e-6, rtol=1e-6)
        assert np.allclose(row["tintensity"], star["rows"][:, 2], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("case", CASES)
def test_format_iraf_is_iraf_s_file(tmp_path, monkeypatch, case):
    """Byte by byte, but for who, where and when."""
    stars, radius, step, settings = CASES[case]
    monkeypatch.chdir(tmp_path)
    fits.writeto("field.fits", np.asarray(field(), "f4"))
    coords = "edge.coo" if stars is EDGE else "field.coo"
    catalog.write(Table({"x": [s[0] for s in stars], "y": [s[1] for s in stars]}), coords)
    get_task("radprof").run(image="field.fits", radius=radius, step=step, coords=Path(coords),
                            output=Path("out.prf"), format="iraf", **PARAMS, **settings)
    ours = Path("out.prf").read_text(encoding="utf-8").splitlines()
    theirs = (DATA / f"{case}.prf").read_text(encoding="utf-8").splitlines()
    assert ours[5:] == theirs[5:]


@pytest.mark.parametrize("case", list(KEYS))
def test_by_the_cursor_it_writes_iraf_s_file(tmp_path, monkeypatch, case):
    """What each key writes and with which numbers: tests/data/iraf_radprof_keys."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARWOOD_DISPLAY", "headless")
    fits.writeto("field.fits", np.asarray(field(), "f4"))
    catalog.write(Table({"x": [s[0] for s in CURSOR], "y": [s[1] for s in CURSOR]}),
                  "field.coo")
    Path("keys.cur").write_text(KEYS[case]["keys"], encoding="utf-8")
    get_task("radprof").run(image="field.fits", radius=12.0, step=0.5,
                            coords=Path("field.coo"), output=Path("out.prf"), format="iraf",
                            icommands="keys.cur", order=5, kreject=3.0, nreject=1, **PARAMS)
    ours = Path("out.prf").read_text(encoding="utf-8").splitlines()
    theirs = KEYS[case]["prf"].splitlines()
    assert ours[5:] == theirs[5:]


def test_a_failed_fit_leaves_its_code_for_the_stars_after_it():
    """IRAF never clears it: a star fine in itself says it failed."""
    image = field().astype(np.float32)
    from harwood.core import apphot

    fitter = radprof.Fitter(12.0, 0.5, order=5)
    sky = apphot.fitsky(image, 40.3, 45.7, apphot.SkyPars("centroid", 10.0, 10.0))
    assert fitter.fit(image, 40.3, 45.7, sky, [3.0], 25.0, apphot.DataPars()).ier == radprof.OK
    fitter.ier = radprof.RP_SINGULAR
    assert fitter.fit(image, 40.3, 45.7, sky, [3.0], 25.0,
                      apphot.DataPars()).ier == radprof.RP_SINGULAR


def test_a_point_rejected_from_a_curve_leaves_it_as_if_never_there():
    x, y = np.arange(20, dtype=np.float32), np.arange(20, dtype=np.float32) ** 2
    with_all, without = curfit.Curve("spline3", 2, 0, 19), curfit.Curve("spline3", 2, 0, 19)
    one = np.ones(1, np.float32)
    for k in range(20):
        with_all.accumulate(x[k:k + 1], y[k:k + 1], one)
        if k != 7:
            without.accumulate(x[k:k + 1], y[k:k + 1], one)
    with_all.reject(x[7:8], y[7:8], one)
    assert with_all.npts == without.npts == 19
    assert with_all.solve() == without.solve() == curfit.OK
    assert np.allclose(with_all.coeff, without.coeff, atol=1e-2)


def test_an_aperture_past_the_radius_is_warned_of(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fits.writeto("field.fits", field())
    catalog.write(Table({"x": [CURSOR[0][0]], "y": [CURSOR[0][1]]}), "c.coo")
    result = get_task("radprof").run(image="field.fits", radius=4.0, step=0.5,
                                     coords=Path("c.coo"), output=Path("o.prf"),
                                     apertures=[6.0], annulus=10.0, dannulus=10.0)
    assert any("short of light" in str(warning) for warning in result.warnings)
    assert result.data["stars"] == 1


def test_radprof_needs_coordinates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fits.writeto("field.fits", field())
    with pytest.raises(ValueError):
        get_task("radprof").run(image="field.fits", radius=4.0, step=0.5)
