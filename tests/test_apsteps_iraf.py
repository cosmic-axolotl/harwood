"""apfind, apresize, aprecenter, aptrace and apsum, one after another, against IRAF.

They are apall in IRAF too: t_apall.x with the switches each one's parameter
file sets, and the apertures handed from step to step in database/ap<frame>.
tests/data/iraf_apsteps.json and .npz hold what IRAF 2.18.1 wrote running
them in sequence on tests/iraf_field's frames -- apfind finding on a single
line, as its nsum=1 has it; apresize sizing on that line; aptrace following
the result; apsum extracting it, traced again or as stored, with variance
weights, through another frame's apertures; aprecenter moving a reference's
apertures onto a frame 1.3 px along the slit. The apertures after the last
step, the RECENTER line and every band extracted must come out the same.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import arc_frame, spectra_frame

from harwood.core import apextract, database, spectrum
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA / "iraf_apsteps.json").read_text(encoding="utf-8"))
SPECTRA = np.load(DATA / "iraf_apsteps.npz")
BANDS = {"spectrum": "flux", "raw": "raw", "background": "sky", "sigma": "error"}


@pytest.mark.parametrize("case", list(REFERENCE["cases"]))
def test_the_steps_of_apall_one_by_one_are_irafs(tmp_path, monkeypatch, case):
    entry = REFERENCE["cases"][case]
    header = fits.Header({"DISPAXIS": 2})
    fits.writeto(tmp_path / "A.fits", spectra_frame(), header)
    fits.writeto(tmp_path / "B.fits", spectra_frame(1.3), header)
    fits.writeto(tmp_path / "arc.fits", arc_frame(), header)
    monkeypatch.chdir(tmp_path)

    recentred = None
    for task, image, params in entry["steps"]:
        given = {key: value for key, value in params.items() if key != "extras"}
        if task in ("apall", "apfind"):
            given["nfind"] = 2
        if task in ("apall", "apsum"):
            given["output"] = f"{image}.ms.fits"
        result = get_task(task).run(input=image, **given)
        recentred = result.data.get("recenter", recentred)
    assert recentred == entry["log"]

    image = entry["steps"][-1][1]
    record = database.read(image, Path("."), prefix="ap")
    stored = [apextract.from_record(r, 300) for r in record.extra["apertures"]]
    for aperture, (centre, low, high) in zip(stored, entry["apertures"], strict=True):
        assert aperture.centre == pytest.approx(centre, abs=1e-4)
        assert aperture.low == pytest.approx(low, abs=1e-4)
        assert aperture.high == pytest.approx(high, abs=1e-4)

    keys = [k for k in SPECTRA.files if k.startswith(case + "__")]
    assert bool(keys) == (entry["steps"][-1][0] in ("apsum", "apall"))
    if keys:
        ours = spectrum.read(f"{image}.ms.fits")
        for key in keys:
            theirs = SPECTRA[key].astype(float)
            mine = np.array([getattr(a, BANDS[key.split("__")[1]]) for a in ours],
                            dtype=float)
            assert np.max(np.abs(mine - theirs) / np.maximum(np.abs(theirs), 1.0)) < 5e-6


def test_a_step_that_does_not_extract_writes_only_the_apertures(tmp_path, monkeypatch):
    fits.writeto(tmp_path / "A.fits", spectra_frame(), fits.Header({"DISPAXIS": 2}))
    monkeypatch.chdir(tmp_path)
    result = get_task("aptrace").run(input="A", nfind=2)
    assert [Path(p).name for p in result.outputs] == ["apA"]
    assert [step.action for step in get_task("aptrace").plan(
        get_task("aptrace").Params(**{**get_task("aptrace").defaults, "input": "A"}))] == \
        ["read", "profile", "reference", "trace"]
