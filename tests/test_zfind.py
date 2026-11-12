"""zfind against what is known: spectra made at a redshift, recovered within their
error; the error itself honest (the pull of many noise draws has a spread of 1);
a nucleus whose [N II] 6583 outshines Halpha, where RVSAO's emsao search takes the
one for the other and zfind does not; noise alone flagged as doubtful; and a
template spectrum, for absorption lines."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import emsao, zfind

LINES = {6548.06: 300, 6562.82: 1500, 6583.57: 800, 6716.44: 350, 6730.82: 260,
         6300.3: 120, 3727.3: 900, 4861.33: 500, 4958.91: 600, 5006.84: 1700}


def _spectrum(w, z, lines, rng, noise=8.0, sigma_kms=100.0, cont=400.0):
    f = cont + 0.05 * (w - w[0])
    for rest, amp in lines.items():
        centre = rest * (1 + z)
        f = f + amp * np.exp(-0.5 * ((w - centre) / (centre * sigma_kms / zfind.C)) ** 2)
    return f + rng.normal(0, noise, w.size)


@pytest.mark.parametrize("z, w0, dw, n", [(0.0026, 6300.0, 1.1, 600),
                                          (0.05, 3600.0, 2.0, 1000),
                                          (0.3, 4000.0, 3.0, 1300)])
def test_the_redshift_made_is_the_redshift_found(z, w0, dw, n):
    rng = np.random.default_rng(int(z * 1e4))
    w = w0 + dw * np.arange(n)
    r = zfind.zscan(w, _spectrum(w, z, LINES, rng), zmax=0.5)
    assert abs(r.z - z) < 4 * r.z_err
    assert r.z_err < 3e-5
    assert not r.doubtful


def test_the_error_is_what_the_noise_makes_it():
    rng = np.random.default_rng(7)
    w = 6300.0 + 1.1 * np.arange(600)
    weak = {k: v * 0.08 for k, v in LINES.items()}
    pulls = []
    for _ in range(40):
        r = zfind.zscan(w, _spectrum(w, 0.0026, weak, rng), zmin=-0.002, zmax=0.008)
        pulls.append((r.z - 0.0026) / r.z_err)
    assert 0.7 < np.std(pulls) < 1.35
    assert abs(np.mean(pulls)) < 0.5


def test_a_nucleus_whose_nii_outshines_halpha(tmp_path, monkeypatch):
    """emsao's search takes the tallest line near Halpha for Halpha; here it is
    [N II] 6583, and the redshift comes out 950 km/s too high. zfind fits them all."""
    rng = np.random.default_rng(3)
    w = 6300.0 + 1.1 * np.arange(600)
    liner = {6548.06: 450, 6562.82: 1000, 6583.57: 1350, 6716.44: 450, 6730.82: 380}
    f = _spectrum(w, 0.0029, liner, rng, sigma_kms=95.0).astype(np.float32)
    h = fits.Header({"CRVAL1": 6300.0, "CRPIX1": 1.0, "CDELT1": 1.1, "CD1_1": 1.1,
                     "CTYPE1": "LINEAR", "WCSDIM": 1, "DC-FLAG": 0,
                     "WAT0_001": "system=equispec",
                     "WAT1_001": "wtype=linear label=Wavelength units=angstroms",
                     "APNUM1": "1 1 100 110", "OBJECT": "liner"})
    fits.writeto(tmp_path / "liner.fits", f, h)
    monkeypatch.chdir(tmp_path)
    search = emsao.Emsao({"vel_corr": "none", "logfiles": ""})
    search.run(["liner.fits"])
    assert abs(search.results[0].vel - 0.0029 * emsao.C0) > 500     # the wrong line
    r = zfind.zscan(w, f.astype(float), zmax=0.5)
    assert abs(r.cz - 0.0029 * zfind.C) < 4 * r.cz_err
    assert not r.doubtful


def test_noise_alone_is_doubtful():
    rng = np.random.default_rng(11)
    w = 6300.0 + 1.1 * np.arange(600)
    r = zfind.zscan(w, 400 + rng.normal(0, 8, w.size), zmax=0.5)
    assert r.doubtful


def test_a_template_for_absorption_lines():
    rng = np.random.default_rng(5)
    rest = np.linspace(3800, 6000, 4000)
    template = np.ones_like(rest)
    for line, depth in ((3933.7, 0.6), (3968.5, 0.5), (4304.4, 0.25), (5175.4, 0.3),
                        (5893.0, 0.2), (4861.3, 0.15)):
        template -= depth * np.exp(-0.5 * ((rest - line) / 3.0) ** 2)
    w = 4000.0 + 1.5 * np.arange(1200)
    z = 0.035
    f = 1000 * np.interp(w, rest * (1 + z), template) + rng.normal(0, 10, w.size)
    r = zfind.zscan(w, f, rest_lines=[], zmin=0.0, zmax=0.1, templates=[(rest, template)])
    assert abs(r.z - z) < 4 * r.z_err
    assert not r.doubtful
