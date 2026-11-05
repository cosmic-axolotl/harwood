"""The spectra the deredden cases start from, as IRAF's multispec (python
der_inputs.py <folder>, harwood's venv: in.fits and far.fits) and as harwood's tables
(spectra(name), for the test). in's three apertures reach every piece of Cardelli,
Clayton & Mathis's law: the far ultraviolet, the optical, the infrared; far's is
shorter than the law goes."""

import sys
from pathlib import Path

import numpy as np

N = 200
HEADER = {"OBJECT": "reddened star", "EXPTIME": 900.0}
RANGES = {"in": [(1, 1100.0, 2900.0), (2, 3000.0, 9000.0), (3, 9000.0, 30000.0)],
          "far": [(1, 500.0, 2000.0)]}


def spectra(name="in"):
    """(aperture, w1, dw, flux) for each aperture of `name`."""
    pix = np.arange(N, dtype=float)
    out = []
    for ap, start, end in RANGES[name]:
        flux = (500.0 + 50.0 * np.cos(pix / 9.0 + ap) + 5.0 * ap).astype(np.float32)
        out.append((ap, start, (end - start) / (N - 1), flux))
    return out


def write(folder: Path) -> None:
    from astropy.io import fits

    for name in RANGES:
        rows = spectra(name)
        header = fits.Header()
        for key, value in HEADER.items():
            header[key] = value
        header["WCSDIM"] = 2
        header["CTYPE1"], header["CTYPE2"] = "MULTISPE", "MULTISPE"
        header["CD1_1"], header["CD2_2"] = 1.0, 1.0
        header["LTM1_1"], header["LTM2_2"] = 1.0, 1.0
        header["DC-FLAG"] = 0
        header["WAT0_001"] = "system=multispec"
        header["WAT1_001"] = "wtype=multispec label=Wavelength units=angstroms"
        text = "wtype=multispec " + " ".join(
            f'spec{ap} = "{ap} {ap} 0 {w1!r} {dw!r} {N} 0. {ap * 10 - 3}. {ap * 10 + 3}."'
            for ap, w1, dw, _ in rows)
        for i in range(0, len(text), 68):
            header[f"WAT2_{i // 68 + 1:03d}"] = text[i:i + 68]
        for ap, *_ in rows:
            header[f"APNUM{ap}"] = f"{ap} {ap} {ap * 10 - 3}. {ap * 10 + 3}."
        data = np.vstack([flux for *_, flux in rows]).astype(np.float32)
        fits.writeto(folder / f"{name}.fits", data, header, overwrite=True)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
