"""dopcor against IRAF 2.18's: what it says, the DOPCORnn cards it writes, and each
aperture's wavelengths and fluxes.

tests/data/iraf_dopcor holds the cases (dop_cases.py), the spectrum they start from
(dop_inputs.py: three apertures with linear wavelengths, as IRAF's multispec and as
harwood's tables), IRAF's runs of them (run.py) and what they left (dop.json). IRAF
keeps the correction as a z in each aperture's world coordinates; harwood's
wavelengths are in the table, so they are held to IRAF's (w1 + dw (p-1)) / (1+z).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import spectrum
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_dopcor"
GOLD = json.loads((HERE / "dop.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from dop_cases import CASES  # noqa: E402
from dop_inputs import HEADER, N, spectra  # noqa: E402


@pytest.fixture
def start(tmp_path, monkeypatch):
    header = fits.Header()
    for key, value in HEADER.items():
        header[key] = value
    pixel = np.arange(1.0, N + 1.0)
    spectrum.write(tmp_path / "in.fits",
                   [spectrum.Aperture(number=ap, centre=10.0 * ap, low=-3.0, high=3.0,
                                      pixel=pixel, flux=flux.astype(float),
                                      wavelength=w1 + dw * (pixel - 1))
                    for ap, w1, dw, flux in spectra()], header=header)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(case):
    results, source = [], "in"
    for number, step in enumerate(CASES[case], 1):
        result = get_task("dopcor").run(input=source, output=f"o{number}", **step)
        results.append(result)
        source = f"o{number}"
    return results


@pytest.mark.parametrize("case", list(CASES))
def test_dopcor_corrects_as_iraf_does(start, case):
    results = _run(case)
    for number, (result, theirs) in enumerate(zip(results, GOLD[case], strict=True), 1):
        where = f"{case}, run {number}"
        assert result.data["text"] == theirs["said"], where
        written = Path(f"o{number}.fits")
        assert written.exists() == theirs["written"], where
        if not theirs["written"]:
            continue
        header = fits.getheader(written)
        assert {k: str(header[k]) for k in header if k.startswith("DOPCOR")} == \
            theirs["cards"], where
        for aperture in spectrum.read(written):
            iraf_s = theirs["apertures"][str(aperture.number)]
            pixel = np.arange(N, dtype=float)
            wave = (iraf_s["w1"] + iraf_s["dw"] * pixel) / (1 + iraf_s["z"])
            # IRAF writes z in the WCS with %g's figures.
            assert aperture.wavelength == pytest.approx(wave, rel=1e-9), where
            assert aperture.redshift == pytest.approx(iraf_s["z"], rel=1e-6, abs=1e-12)
            assert np.array_equal(aperture.flux.astype(np.float32),
                                  np.asarray(iraf_s["flux"], np.float32)), where


def test_a_spectrum_in_pixels_has_no_dispersion_function(start):
    pixel = np.arange(1.0, N + 1.0)
    spectrum.write(start / "pix.fits", [spectrum.Aperture(
        number=1, centre=1.0, low=-1.0, high=1.0, pixel=pixel, flux=pixel)])
    result = get_task("dopcor").run(input="pix", output="o", redshift="0.01")
    assert result.data["text"] == ["Warning: [pix] has no dispersion function"]
    assert not Path("o.fits").exists()


def test_no_correction_at_all_is_an_error(start):
    with pytest.raises(ValueError, match="No correction specified"):
        get_task("dopcor").run(input="in", output="o", redshift="0.01",
                               dispersion=False, flux=False)
