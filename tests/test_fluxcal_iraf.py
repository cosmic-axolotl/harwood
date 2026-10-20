"""standard, sensfunc and calibrate against IRAF's own output.

tests/data/iraf_fluxcal holds what IRAF 2.18.1 did with the spectra of
tests/iraf_field.standard_spectra(): standard on two exposures of a made-up
standard at airmasses 1.15 and 1.85 (its std file, as written), sensfunc on
that file with four fitting functions (the sensitivity spectra), and
calibrate on an object with each of them.

Each step is checked on IRAF's own input, so a difference belongs to the
step: harwood's standard against IRAF's std file, harwood's sensfunc on
IRAF's std file against IRAF's sensitivity spectra, harwood's calibrate
with IRAF's sensitivity spectra against IRAF's calibrated spectra. Then the
whole chain on harwood's own steps.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import standard_spectra

from harwood.core import flux, spectrum, stdfile
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data"
REF = DATA / "iraf_fluxcal"
CASES = json.loads((REF / "cases.json").read_text())
OUT = np.load(REF / "outputs.npz")
W1, DW, N = 4014.29852458081, 0.28769335609626, 1024


def write(path, name):
    counts, airmass, exptime = standard_spectra()[name]
    aperture = spectrum.Aperture(number=1, centre=1.0, low=-1.0, high=1.0,
                                 pixel=np.arange(1.0, N + 1.0),
                                 flux=counts.astype(float),
                                 wavelength=W1 + DW * np.arange(N), title=name)
    header = fits.Header({"AIRMASS": airmass, "EXPTIME": exptime,
                          "OBJECT": "fakestar" if name != "obj" else "object"})
    spectrum.write(path, [aperture], header=header, source=name)
    return path


def test_standard_measures_what_iraf_measures(tmp_path):
    for name in ("std1", "std2"):
        get_task("standard").run(input=str(write(tmp_path / f"{name}.fits", name)),
                                 output=tmp_path / "std", caldir=str(REF / "mycal"),
                                 extinction=str(DATA / "kpnoextinct.dat"))
    ours = stdfile.read(tmp_path / "std").measurements
    theirs = stdfile.read(REF / "std").measurements
    assert len(ours) == len(theirs) == 26
    for a, b in zip(ours, theirs, strict=True):
        assert a.wavelength == b.wavelength and a.width == b.width
        # IRAF prints the counts and flambda to five figures.
        assert a.counts == pytest.approx(b.counts, rel=5e-5)
        assert a.flux == pytest.approx(b.flux, rel=5e-5)


@pytest.mark.parametrize("case", list(CASES["sensfunc"]))
def test_sensfunc_on_iraf_s_standards_is_iraf_s(tmp_path, case):
    function, order = CASES["sensfunc"][case]
    get_task("sensfunc").run(standards=REF / "std", sensitivity="sens",
                             extinction=str(DATA / "kpnoextinct.dat"),
                             function=function, order=order, database=tmp_path)
    ours = flux.SensSpectrum.read(tmp_path / "sens.0001.fits")
    w1, dw, nw = CASES["grids"]["sens_" + case]
    assert ours.values.size == nw
    assert np.float32(ours.w1) == np.float32(w1) and np.float32(ours.dw) == np.float32(dw)
    # IRAF's curfit, in its single precision: the same bits.
    assert np.array_equal(ours.values, OUT["sens_" + case])


@pytest.mark.parametrize("case", ["s3", "ch", "le", "s1", "noext"])
def test_calibrate_with_iraf_s_sensitivity_is_iraf_s(tmp_path, case):
    # IRAF's sensitivity spectrum, rebuilt from the frozen values and grid.
    key = "sens_" + ("s3" if case == "noext" else case)
    w1, dw, _ = CASES["grids"][key]
    flux.SensSpectrum(w1, dw, OUT[key]).write(sens := tmp_path / "sens.fits")
    get_task("calibrate").run(input=str(write(tmp_path / "obj.fits", "obj")),
                              output=tmp_path / "cal.fits", sensitivity=str(sens),
                              extinction=str(DATA / "kpnoextinct.dat"),
                              extinct=case != "noext")
    ours = spectrum.read(tmp_path / "cal.fits")[0].flux
    theirs = OUT["cal_" + case].astype(float)
    assert np.max(np.abs(ours - theirs) / np.abs(theirs)) < 1e-5


def test_the_whole_chain_lands_on_iraf_s_flux(tmp_path):
    """harwood's own standard, sensfunc and calibrate: the difference left is
    IRAF's std file holding five figures and its curfit single precision."""
    for name in ("std1", "std2"):
        get_task("standard").run(input=str(write(tmp_path / f"{name}.fits", name)),
                                 output=tmp_path / "std", caldir=str(REF / "mycal"),
                                 extinction=str(DATA / "kpnoextinct.dat"))
    get_task("sensfunc").run(standards=tmp_path / "std", sensitivity="sens",
                             database=tmp_path)
    get_task("calibrate").run(input=str(write(tmp_path / "obj.fits", "obj")),
                              output=tmp_path / "cal.fits", sensitivity="sens",
                              database=tmp_path,
                              extinction=str(DATA / "kpnoextinct.dat"))
    ours = spectrum.read(tmp_path / "cal.fits")[0].flux
    theirs = OUT["cal_s3"].astype(float)
    ratio = ours / theirs
    assert np.max(np.abs(ratio - 1)) < 2e-3
    assert abs(np.median(ratio) - 1) < 1e-4


def test_fnu_is_iraf_s(tmp_path):
    """fnu: each sample of the sensitivity times lambda^2/c before it is
    interpolated, as cal_getflux does -- IRAF's erg/cm2/s/Hz spectrum."""
    w1, dw, _ = CASES["grids"]["sens_s3"]
    flux.SensSpectrum(w1, dw, OUT["sens_s3"]).write(sens := tmp_path / "sens.fits")
    result = get_task("calibrate").run(input=str(write(tmp_path / "obj.fits", "obj")),
                                       output=tmp_path / "cal.fits", sensitivity=str(sens),
                                       extinction=str(DATA / "kpnoextinct.dat"), fnu=True)
    ours = spectrum.read(tmp_path / "cal.fits")[0].flux
    theirs = np.load(REF / "cal_fnu_s3.npy").astype(float)
    assert np.max(np.abs(ours - theirs) / np.abs(theirs)) < 1e-5
    assert result.data["unit"] == "erg/cm2/s/Hz"


def test_each_aperture_takes_its_own_sensitivity(tmp_path):
    """ignoreaps=no: sens.0002 for aperture 2 when it is there."""
    w1, dw, _ = CASES["grids"]["sens_s3"]
    flux.SensSpectrum(w1, dw, OUT["sens_s3"]).write(tmp_path / "sens.fits")
    flux.SensSpectrum(w1, dw, OUT["sens_s3"] + 2.5).write(tmp_path / "sens.0001.fits")
    obj = write(tmp_path / "obj.fits", "obj")
    common = dict(input=str(obj), sensitivity=str(tmp_path / "sens"),
                  extinction=str(DATA / "kpnoextinct.dat"))
    get_task("calibrate").run(output=tmp_path / "own.fits", **common)
    get_task("calibrate").run(output=tmp_path / "one.fits", ignoreaps=True, **common)
    own = spectrum.read(tmp_path / "own.fits")[0].flux
    one = spectrum.read(tmp_path / "one.fits")[0].flux
    assert np.allclose(own, one * 10 ** (-0.4 * 2.5), rtol=1e-5)
