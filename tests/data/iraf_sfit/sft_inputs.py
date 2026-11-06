"""The spectra the sfit and continuum cases fit, as IRAF's images (python sft_inputs.py
<folder>, harwood's venv) and as harwood's tables (spectra()): `a`, three apertures of
a continuum with absorption lines, noise and cosmic rays; `c`, two apertures in the four
bands apall writes (spectrum, raw, background, sigma -- harwood's flux, raw, sky and
error); `one`, a single aperture, one-dimensional; `z`, with zeros and negative values,
for a ratio's division by zero and a log of what has none; `done`, a copy of `a` whose
SFIT and SFITB say its lines 1 and 3 of band 1 are fitted already; and `pre`, an
existing output of `a`'s size, with SFIT saying line 2 is."""

import sys
from pathlib import Path

import numpy as np

N = 160
#: name: ([(aperture, beam, w1, dw, title)], bands, cards)
SPECTRA = {
    "a": ([(1, 1, 4000.0, 2.0, "a one"), (2, 2, 4010.0, 2.05, "a two"),
           (3, 3, 4005.0, 1.95, "a three")], 1, {}),
    "c": ([(1, 1, 4000.0, 2.0, "c one"), (2, 2, 4012.0, 2.1, "c two")], 4, {}),
    "one": ([(1, 1, 4002.0, 2.02, "one")], 1, {}),
    "z": ([(1, 1, 4000.0, 2.0, "z one"), (2, 2, 4000.0, 2.0, "z two")], 1, {}),
    "done": ([(1, 1, 4000.0, 2.0, "a one"), (2, 2, 4010.0, 2.05, "a two"),
              (3, 3, 4005.0, 1.95, "a three")], 1, {"SFIT": "1,3", "SFITB": "1"}),
    "pre": ([(1, 1, 4000.0, 2.0, "a one"), (2, 2, 4010.0, 2.05, "a two"),
             (3, 3, 4005.0, 1.95, "a three")], 1, {"SFIT": "2", "SFITB": "1"}),
}
SOURCE = {"done": "a", "pre": "a"}
BANDS = ("spectrum", "raw", "background", "sigma")


def data(name):
    """[band][line] of pixels, in real."""
    source = SOURCE.get(name, name)
    rows, nbands, _ = SPECTRA[name]
    rng = np.random.default_rng(sum(map(ord, source)))
    out = np.zeros((nbands, len(rows), N), dtype=np.float32)
    x = np.arange(N, dtype=float)
    for line, (ap, _, w1, dw, _) in enumerate(rows):
        wave = w1 + dw * x
        cont = 1000.0 + 0.8 * (wave - 4150.0) - 0.002 * (wave - 4150.0) ** 2 \
            + 40.0 * np.sin(wave / 60.0) + 20.0 * ap
        lines = sum(depth * np.exp(-0.5 * ((wave - centre) / width) ** 2)
                    for centre, depth, width in ((4080.0, 300.0, 4.0), (4190.0, 450.0, 3.0),
                                                 (4260.0, 200.0, 6.0)))
        flux = cont - lines + rng.normal(0.0, 12.0, N)
        for at in rng.choice(N, 3, replace=False):
            flux[at] += rng.uniform(300.0, 900.0)
        if name == "pre":
            flux = flux * 0.5
        if name == "z":
            flux = flux - (900.0 if ap == 1 else 0.0)
            flux[[10, 11, 50, 120]] = 0.0
        out[0, line] = flux
        if nbands > 1:
            out[1, line] = flux * 1.1 + 30.0
            out[2, line] = 50.0 + 0.05 * x + rng.normal(0.0, 3.0, N)
            out[3, line] = np.sqrt(np.abs(flux)) + 2.0
    return out


def _header(name):
    from astropy.io import fits

    rows, nbands, cards = SPECTRA[name]
    header = fits.Header()
    header["OBJECT"] = f"star {SOURCE.get(name, name)}"
    ndim = 1 if len(rows) == 1 and nbands == 1 else (2 if nbands == 1 else 3)
    header["WCSDIM"] = max(ndim, 2) if ndim > 1 else 1
    if ndim == 1:
        ap, beam, w1, dw, title = rows[0]
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
    else:
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
    for key, value in cards.items():
        header[key] = value
    return header


def write(folder: Path) -> None:
    from astropy.io import fits

    for name, (rows, nbands, _) in SPECTRA.items():
        pixels = data(name)
        if nbands == 1:
            pixels = pixels[0]
            if len(rows) == 1:
                pixels = pixels[0]
        fits.writeto(folder / f"{name}.fits", pixels, _header(name), overwrite=True)


def spectra(folder: Path) -> None:
    """The same spectra as harwood's tables."""
    from astropy.io import fits

    from harwood.core import spectrum

    for name, (rows, nbands, cards) in SPECTRA.items():
        pixels = data(name)
        pixel = np.arange(1.0, N + 1.0)
        header = fits.Header()
        header["OBJECT"] = f"star {SOURCE.get(name, name)}"
        for key, value in cards.items():
            header[key] = value
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


if __name__ == "__main__":
    write(Path(sys.argv[1]))
