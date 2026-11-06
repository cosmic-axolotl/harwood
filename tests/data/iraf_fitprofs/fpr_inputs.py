"""The spectra the fitprofs cases fit, as IRAF's images (python fpr_inputs.py <folder>,
harwood's venv) and as harwood's tables (spectra()): `a`, two apertures of a sloping
continuum with a gaussian line at 4100, a blend of two at 4180 and 4192, a lorentzian
at 4260 and a voigt at 4330, and noise; `one`, a single aperture with one absorption
line at 4150; `c`, two apertures in the four bands apall writes; and the positions
files: `p1` one line, `p2` the blend with peaks, `p3` each line with its own profile and
widths, `p4` the voigt, `p5` the absorption line, `empty` nothing in it."""

import sys
from pathlib import Path

import numpy as np

N = 200
#: name: ([(aperture, beam, w1, dw, title)], bands)
SPECTRA = {
    "a": ([(1, 1, 4000.0, 2.0, "a one"), (2, 2, 4004.0, 2.02, "a two")], 1),
    "one": ([(1, 1, 4002.0, 1.5, "one")], 1),
    "c": ([(1, 1, 4000.0, 2.0, "c one"), (2, 2, 4006.0, 2.0, "c two")], 4),
}
POSITIONS = {
    "p1": "4100\n",
    "p2": "4180 300\n4192 150\n",
    "p3": "4100 INDEF gaussian 7\n4180 300 gaussian 6\n4192 150 lorentzian 5\n"
          "4260 INDEF lorentzian 8\n",
    "p4": "4330 INDEF voigt 6 5\n",
    "p5": "4150 -250 gaussian 6\n",
    "empty": "\n",
}
BANDS = ("spectrum", "raw", "background", "sigma")


def _voigt_shape(x, g, lw):
    from scipy.special import wofz

    sigma = g / 2.3548200450309493
    gamma = lw / 2.0
    z = (x + 1j * gamma) / (sigma * np.sqrt(2.0))
    return wofz(z).real / wofz(1j * gamma / (sigma * np.sqrt(2.0))).real


def data(name):
    """[band][line] of pixels, in real."""
    rows, nbands = SPECTRA[name]
    rng = np.random.default_rng(sum(map(ord, name)) + 7)
    out = np.zeros((nbands, len(rows), N), dtype=np.float32)
    x = np.arange(N, dtype=float)
    for line, (ap, _, w1, dw, _) in enumerate(rows):
        wave = w1 + dw * x
        flux = 900.0 + 0.4 * (wave - 4000.0) + 15.0 * ap
        if name == "one":
            flux = flux - 250.0 * np.exp(-0.5 * ((wave - 4150.3) / 2.6) ** 2)
        else:
            flux = flux + 500.0 * np.exp(-0.5 * ((wave - 4101.2) / 3.0) ** 2) \
                + 320.0 * np.exp(-0.5 * ((wave - 4179.4) / 2.5) ** 2) \
                + 170.0 * np.exp(-0.5 * ((wave - 4193.1) / 2.8) ** 2) \
                + 260.0 / (1.0 + ((wave - 4259.0) / 3.5) ** 2) \
                + 280.0 * _voigt_shape(wave - 4331.0, 5.0, 4.0)
        flux = flux + rng.normal(0.0, 6.0, N)
        out[0, line] = flux
        if nbands > 1:
            out[1, line] = flux * 1.1 + 30.0
            out[2, line] = 50.0 + 0.05 * x + rng.normal(0.0, 3.0, N)
            out[3, line] = np.sqrt(np.abs(flux)) + 2.0
    return out


def _header(name):
    from astropy.io import fits

    rows, nbands = SPECTRA[name]
    header = fits.Header()
    header["OBJECT"] = f"star {name}"
    if len(rows) == 1 and nbands == 1:
        ap, beam, w1, dw, title = rows[0]
        header["WCSDIM"] = 1
        header["CTYPE1"] = "LINEAR"
        header["CRVAL1"] = w1
        header["CDELT1"] = dw
        header["CD1_1"] = dw
        header["CRPIX1"] = 1.0
        header["LTM1_1"] = 1.0
        header["WAT0_001"] = "system=equispec"
        header["WAT1_001"] = "wtype=linear label=Wavelength units=angstroms"
        header["APNUM1"] = f"{ap} {beam} {ap * 10 - 3}. {ap * 10 + 3}."
        header["OBJECT"] = title
        return header
    header["WCSDIM"] = 3 if nbands > 1 else 2
    for axis in range(1, header["WCSDIM"] + 1):
        header[f"CTYPE{axis}"] = "MULTISPE" if axis < 3 else "LINEAR"
        header[f"CD{axis}_{axis}"] = 1.0
        header[f"LTM{axis}_{axis}"] = 1.0
    header["WAT0_001"] = "system=multispec"
    header["WAT1_001"] = "wtype=multispec label=Wavelength units=angstroms"
    text = "wtype=multispec " + " ".join(
        f'spec{i} = "{ap} {beam} 0 {w1!r} {dw!r} {N} 0. {ap * 10 - 3}. {ap * 10 + 3}."'
        for i, (ap, beam, w1, dw, _) in enumerate(rows, 1))
    for i in range(0, len(text), 68):
        header[f"WAT2_{i // 68 + 1:03d}"] = text[i:i + 68]
    if nbands > 1:
        header["WAT3_001"] = "wtype=linear"
        for band, what in enumerate(BANDS, 1):
            header[f"BANDID{band}"] = what
    for i, (ap, beam, *_rest, title) in enumerate(rows, 1):
        header[f"APNUM{i}"] = f"{ap} {beam} {ap * 10 - 3}. {ap * 10 + 3}."
        header[f"APID{i}"] = title
    return header


def _positions(folder: Path) -> None:
    for name, text in POSITIONS.items():
        (folder / name).write_text(text, encoding="utf-8")


def write(folder: Path) -> None:
    from astropy.io import fits

    for name, (rows, nbands) in SPECTRA.items():
        pixels = data(name)
        if nbands == 1:
            pixels = pixels[0]
            if len(rows) == 1:
                pixels = pixels[0]
        fits.writeto(folder / f"{name}.fits", pixels, _header(name), overwrite=True)
    _positions(folder)


def spectra(folder: Path) -> None:
    """The same spectra as harwood's tables."""
    from astropy.io import fits

    from harwood.core import spectrum

    for name, (rows, nbands) in SPECTRA.items():
        pixels = data(name)
        pixel = np.arange(1.0, N + 1.0)
        header = fits.Header()
        header["OBJECT"] = f"star {name}"
        apertures = []
        for line, (ap, beam, w1, dw, title) in enumerate(rows):
            bands = {"flux": pixels[0, line].astype(float)}
            if nbands > 1:
                bands.update(raw=pixels[1, line].astype(float),
                             sky=pixels[2, line].astype(float),
                             error=pixels[3, line].astype(float))
            apertures.append(spectrum.Aperture(
                number=ap, centre=10.0 * ap, low=-3.0, high=3.0, pixel=pixel,
                wavelength=w1 + dw * (pixel - 1), title=title, beam=beam, **bands))
        spectrum.write(folder / f"{name}.fits", apertures, header=header)
    _positions(folder)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
