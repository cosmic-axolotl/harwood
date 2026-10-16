"""dispcor against IRAF's own output.

tests/data/iraf_dispcor holds what IRAF 2.18.1's dispcor wrote for
tests/iraf_field.star_spectrum() under twelve sets of parameters -- every
interpolant, flux conserved or averaged, given grids, a log grid, a grid
running past the arc, a decreasing solution, two arcs combined by their
weights -- with the dispersion solutions IRAF's autoidentify wrote, which
harwood reads from IRAF's database format and evaluates coefficient for
coefficient.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import star_spectrum

from harwood.core import spectrum
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data" / "iraf_dispcor"
REFERENCE = json.loads((DATA / "cases.json").read_text())
SPECTRA = np.load(DATA / "spectra.npz")
HEADERS = {"star": {"REFSPEC1": "default"}, "starflip": {"REFSPEC1": "flip"},
           "star2": {"REFSPEC1": "default 0.7", "REFSPEC2": "cheb 0.3"}}


@pytest.mark.parametrize("case", list(REFERENCE["cases"]))
def test_dispcor_resamples_as_iraf_does(tmp_path, case):
    params = dict(REFERENCE["cases"][case])
    source = params.pop("input")
    flux = star_spectrum(flip=source == "starflip")
    aperture = spectrum.Aperture(number=1, centre=1.0, low=-1.0, high=1.0,
                                 pixel=np.arange(1.0, flux.size + 1.0),
                                 flux=flux.astype(float), title="star")
    spectrum.write(tmp_path / "star.ms.fits", [aperture],
                   header=fits.Header(HEADERS[source]), source="star")
    get_task("dispcor").run(input=str(tmp_path / "star.ms.fits"),
                            output=tmp_path / "out.fits", database=DATA,
                            check=False, **params)
    ours = spectrum.read(tmp_path / "out.fits")[0]

    crval, cdelt, naxis, log = REFERENCE["grids"][case]
    assert ours.flux.size == naxis
    wave = np.log10(ours.wavelength) if log else ours.wavelength
    assert wave[0] == pytest.approx(crval, abs=1e-9 if log else 1e-7)
    assert np.diff(wave).mean() == pytest.approx(cdelt, rel=1e-9)
    theirs = SPECTRA[case].astype(float)
    # Single precision, and IRAF's inverse is rounded to 0.0001 px: where a
    # pixel edge lands a step the other way the integral moves by 1e-4 of a
    # pixel's worth.
    assert np.max(np.abs(ours.flux - theirs) / np.maximum(np.abs(theirs), 1.0)) < 1e-4


def test_without_linearizing_the_pixels_are_untouched(tmp_path):
    flux = star_spectrum()
    aperture = spectrum.Aperture(number=1, centre=1.0, low=-1.0, high=1.0,
                                 pixel=np.arange(1.0, flux.size + 1.0),
                                 flux=flux.astype(float), title="star")
    spectrum.write(tmp_path / "star.ms.fits", [aperture],
                   header=fits.Header(HEADERS["star"]), source="star")
    get_task("dispcor").run(input=str(tmp_path / "star.ms.fits"),
                            output=tmp_path / "out.fits", database=DATA,
                            linearize=False, check=False)
    ours = spectrum.read(tmp_path / "out.fits")[0]
    np.testing.assert_array_equal(ours.flux, flux.astype(float))
    # The wavelengths are IRAF's grid's ends: the solution at pixels 1 and n.
    crval, cdelt, naxis, _ = REFERENCE["grids"]["lin"]
    assert ours.wavelength[0] == pytest.approx(crval, abs=1e-6)
    assert ours.wavelength[-1] == pytest.approx(crval + cdelt * (naxis - 1), abs=1e-6)
