"""phot's errors from the image's variance, against the scatter they claim.

A hundred identical stars, each put down at the same sub-pixel position so
their fluxes are draws from one distribution; half of them sit in a
vignetted corner where the flat's response is 0.6. After the flat, a count
there stands for fewer photons, so the noise is larger than the counts
suggest -- which the variance ccdproc carries knows, and IRAF's error,
built from epadu and the sky alone, cannot. The flux scatter in each half
must be what ferr says; IRAF's merr_model falls short in the corner.
"""

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import dq, mef
from harwood.core.registry import get_task

GAIN, READNOISE = 2.0, 5.0
GRID = 10


def _field(rng):
    size = 40 * GRID
    y, x = np.mgrid[1:size + 1, 1:size + 1].astype(float)
    response = np.where(x > size / 2, 0.6, 1.0)
    electrons = np.full((size, size), 100.0)                  # sky
    stars = []
    for j in range(GRID):
        for i in range(GRID):
            cx, cy = 20.3 + 40 * i, 20.6 + 40 * j
            electrons += 300000.0 / (2 * np.pi * 2.0 ** 2) * np.exp(
                -((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2.0 ** 2))
            stars.append((cx, cy))
    detected = rng.poisson(electrons * response) + rng.normal(0, READNOISE, electrons.shape)
    raw = detected / GAIN                                      # ADU, bias off
    var_raw = np.maximum(raw, 0) / GAIN + (READNOISE / GAIN) ** 2
    sci = raw / response                                       # flat-fielded
    var = var_raw / response ** 2
    return sci, var, np.array(stars), x[0] > size / 2


def test_the_errors_from_the_variance_describe_the_scatter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sci, var, stars, _ = _field(np.random.default_rng(4))
    quality = dq.empty(sci.shape)
    mef.write(tmp_path / "field.fits", mef.Image(
        fits.Header({"OBJECT": "grid", "EXPTIME": 1.0}),
        [mef.Plane(sci, var, quality)]))
    np.savetxt(tmp_path / "stars.coo", stars, fmt="%.2f", header="x y", comments="")
    get_task("phot").run(image="field.fits", coords="stars.coo", output="field.mag",
                         apertures=[6.0], annulus=10.0, dannulus=6.0, epadu=GAIN,
                         calgorithm="none")
    from harwood.core import catalog

    table = catalog.read(tmp_path / "field.mag")
    flux, ferr = np.asarray(table["flux"]), np.asarray(table["ferr"])
    model = np.asarray(table["merr_model"]) * flux / 1.0857
    corner = stars[:, 0] > sci.shape[1] / 2
    for where in (~corner, corner):
        assert np.std(flux[where], ddof=1) / np.median(ferr[where]) == \
            pytest.approx(1.0, abs=0.25)                       # 50 stars: 10% one sigma
    # In the corner IRAF's model takes the flat-fielded counts for photons:
    # with the star's photons dominating, it is short by sqrt(0.6), 0.77.
    # (Its sky term is measured in the annulus, so it is right either way.)
    assert np.median(model[corner]) / np.median(ferr[corner]) == pytest.approx(0.77, abs=0.05)
    assert np.median(model[~corner]) / np.median(ferr[~corner]) == pytest.approx(1.0, abs=0.05)
    assert "pdq" not in table.colnames, "nothing flagged, nothing said"
    assert table.meta["harwood_merr"] == "the image's VAR"


def test_a_flagged_pixel_in_the_aperture_is_said(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sci, var, stars, _ = _field(np.random.default_rng(5))
    quality = dq.empty(sci.shape)
    cx, cy = stars[0]
    quality[int(cy) - 1, int(cx) - 1] = dq.SATURATED
    mef.write(tmp_path / "field.fits", mef.Image(fits.Header({"OBJECT": "grid"}),
                                                 [mef.Plane(sci, var, quality)]))
    np.savetxt(tmp_path / "stars.coo", stars[:3], fmt="%.2f", header="x y", comments="")
    result = get_task("phot").run(image="field.fits", coords="stars.coo",
                                  output="field.mag", apertures=[6.0], annulus=10.0,
                                  dannulus=6.0, epadu=GAIN, calgorithm="none")
    from harwood.core import catalog

    table = catalog.read(tmp_path / "field.mag")
    assert list(table["pdq"]) == [dq.SATURATED, 0, 0]
    assert any("saturated" in str(w) for w in result.warnings)
