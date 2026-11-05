"""The spectra the sarith cases work on, as IRAF's multispec (python sa_inputs.py
<folder>, harwood's venv) and as harwood's tables (spectra(name)): `in`, three
apertures linear from 4010 A; `other`, the same apertures on another dispersion,
for a second operand that has to be resampled; `others`, apertures 4 to 6, none of
in's; `four`, two apertures with apall's four bands -- spectrum, raw, background,
sigma."""

import sys
from pathlib import Path

import numpy as np

N = 200
BANDIDS = ["spectrum - background fit, weights variance, clean no",
           "raw - background fit, weights none, clean no",
           "background - background fit", "sigma - background fit, weights variance"]
#: name: [(aperture, beam, w1, dw, title)], bands
SPECTRA = {
    "in": ([(1, 1, 4010.0, 1.6, "first"), (2, 2, 4020.0, 1.7, "second"),
            (3, 5, 4030.0, 1.8, "third")], 1),
    "other": ([(1, 1, 4012.3, 1.55, "o1"), (2, 2, 4015.0, 1.75, "o2"),
               (3, 5, 4033.3, 1.7, "o3")], 1),
    "others": ([(4, 4, 4010.0, 1.6, "x4"), (5, 5, 4020.0, 1.7, "x5"),
                (6, 6, 4030.0, 1.8, "x6")], 1),
    "four": ([(1, 1, 4010.0, 1.6, "a"), (2, 2, 4020.0, 1.7, "b")], 4),
}


def data(name):
    """[aperture line][band] of pixels."""
    pix = np.arange(N, dtype=float)
    rows, bands = SPECTRA[name]
    seed = sum(map(ord, name))
    out = []
    for i, (ap, *_) in enumerate(rows):
        base = 500.0 + 80.0 * np.sin(pix / (6.0 + i) + seed) + 7.0 * ap
        out.append([(base * (1.0 + 0.3 * b) + (b == 2) * 40.0 - (b == 3) * 450.0
                     ).astype(np.float32) if b < 3 else
                    np.sqrt(np.abs(base)).astype(np.float32) for b in range(bands)])
    return out


def write(folder: Path) -> None:
    from astropy.io import fits

    for name, (rows, bands) in SPECTRA.items():
        header = fits.Header()
        header["OBJECT"] = f"star {name}"
        header["WCSDIM"] = 3 if bands > 1 else 2
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
        if bands > 1:
            header["WAT3_001"] = "wtype=linear"
            for b in range(bands):
                header[f"BANDID{b + 1}"] = BANDIDS[b]
        for i, (ap, beam, _, _, title) in enumerate(rows, 1):
            header[f"APNUM{i}"] = f"{ap} {beam} {ap * 10 - 3}. {ap * 10 + 3}."
            header[f"APID{i}"] = title
        cube = np.array([[band for band in line] for line in data(name)], np.float32)
        array = cube[:, 0, :] if bands == 1 else np.transpose(cube, (1, 0, 2))
        fits.writeto(folder / f"{name}.fits", array, header, overwrite=True)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
