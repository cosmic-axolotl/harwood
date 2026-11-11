"""The spectra of IRAF's splot gabarito: python sp_inputs.py <folder> (harwood's venv).

one: a 1-D spectrum with a linear world system (an IRAF onedspec), a sloping
continuum, three emission lines -- two blended -- an absorption line and noise.
slit: a long slit, 41 columns by 400 lines, dispersion along the columns
(DISPAXIS=2), the same spectrum in a galaxy-like profile across the slit.
ms: a multispec image of three apertures: linear, log-linear, and a chebyshev
dispersion function (dtype 2)."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

N = 400
W0, DW = 6400.0, 1.25


def spectrum(wave, rng, scale=1.0):
    cont = 1000 + 0.4 * (wave - 6400)
    lines = (2500 * np.exp(-0.5 * ((wave - 6563.0) / 2.2) ** 2)
             + 900 * np.exp(-0.5 * ((wave - 6583.5) / 2.0) ** 2)
             + 700 * np.exp(-0.5 * ((wave - 6548.0) / 2.0) ** 2)
             - 350 * np.exp(-0.5 * ((wave - 6700.0) / 3.0) ** 2))
    return scale * (cont + lines) + rng.normal(0, 12, wave.size)


def write(where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(73)
    wave = W0 + DW * np.arange(N)
    one = fits.PrimaryHDU(spectrum(wave, rng).astype(np.float32))
    h = one.header
    h["OBJECT"] = "NGC 660 nucleus"
    h["EXPTIME"] = 900.0
    h.update({"CRVAL1": W0, "CRPIX1": 1.0, "CDELT1": DW, "CD1_1": DW, "CTYPE1": "LINEAR",
              "WCSDIM": 1, "DC-FLAG": 0, "WAT0_001": "system=equispec",
              "WAT1_001": "wtype=linear label=Wavelength units=angstroms",
              "APNUM1": "2 1 124.00 134.00", "BANDID1": "spectrum"})
    one.writeto(where / "one.fits", overwrite=True)

    prof = np.exp(-0.5 * ((np.arange(1, 42) - 21.0) / 5.0) ** 2)
    slit = np.array([spectrum(wave, rng, p) for p in prof]).T + 50.0
    s = fits.PrimaryHDU(slit.astype(np.float32))
    s.header.update({"OBJECT": "slit", "DISPAXIS": 2, "CRVAL2": W0, "CRPIX2": 1.0,
                     "CD2_2": DW, "CDELT2": DW, "CTYPE2": "LINEAR", "CRVAL1": 1.0,
                     "CRPIX1": 1.0, "CD1_1": 1.0, "CTYPE1": "LINEAR", "WCSDIM": 2,
                     "WAT0_001": "system=world", "WAT1_001": "wtype=linear",
                     "WAT2_001": "wtype=linear label=Wavelength units=angstroms"})
    s.writeto(where / "slit.fits", overwrite=True)

    lw0, ldw = np.log10(6380.0), 8.3e-5
    cheb = "1. 0. 1 3 1. 400. 6650.0 249.5 0.8"
    wave_log = 10 ** (lw0 + ldw * np.arange(N))
    xn = (np.arange(1, N + 1) - 200.5) / 199.5
    wave_cheb = 6650.0 + 249.5 * xn + 0.8 * (2 * xn * xn - 1)
    data = np.array([spectrum(wave, rng), spectrum(wave_log, rng, 0.8),
                     spectrum(wave_cheb, rng, 1.2)], np.float32)
    m = fits.PrimaryHDU(data)
    m.header.update({"OBJECT": "three apertures", "WCSDIM": 2, "CTYPE1": "MULTISPE",
                     "CTYPE2": "MULTISPE", "CDELT1": 1.0, "CDELT2": 1.0, "CD1_1": 1.0,
                     "CD2_2": 1.0, "LTM1_1": 1.0, "LTM2_2": 1.0,
                     "WAT0_001": "system=multispec",
                     "WAT1_001": "wtype=multispec label=Wavelength units=angstroms",
                     "APID1": "first", "APID3": "third"})
    specs = (f'spec1 = "1 1 0 {W0} {DW} {N} 0. 10.00 20.00" '
             f'spec2 = "2 1 1 {float(lw0)!r} {ldw} {N} 0. 30.00 40.00" '
             f'spec3 = "3 1 2 6400. 1.25 {N} 0. 50.00 60.00 {cheb}"')
    text = "wtype=multispec " + specs
    for k in range(0, len(text), 68):
        m.header[f"WAT2_{k // 68 + 1:03d}"] = text[k:k + 68]
    m.writeto(where / "ms.fits", overwrite=True)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
