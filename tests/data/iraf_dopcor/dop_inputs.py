"""The spectrum the dopcor cases start from, three apertures with linear wavelengths,
as IRAF's multispec (python dop_inputs.py <folder>, harwood's venv: in.fits) and as
harwood's tables (spectra(), for the test)."""

import sys
from pathlib import Path

import numpy as np

N = 200
HEADER = {"OBJECT": "galaxy", "Z": 0.0123, "VHELIO": -23.45, "EXPTIME": 600.0}


def spectra():
    """(aperture, w1, dw, flux) for each aperture."""
    pix = np.arange(N, dtype=float)
    out = []
    for ap in (1, 2, 3):
        flux = (1000.0 + 100.0 * np.sin(pix / 7.0 + ap) + 10.0 * ap).astype(np.float32)
        out.append((ap, 4000.0 + 10.0 * ap, 1.5 + 0.1 * ap, flux))
    return out


def write(folder: Path) -> None:
    from astropy.io import fits

    rows = spectra()
    header = fits.Header()
    for key, value in HEADER.items():
        header[key] = value
    header["WCSDIM"] = 2
    header["CTYPE1"], header["CTYPE2"] = "MULTISPE", "MULTISPE"
    header["CD1_1"], header["CD2_2"], header["LTM1_1"], header["LTM2_2"] = 1.0, 1.0, 1.0, 1.0
    header["DC-FLAG"] = 0
    header["WAT0_001"] = "system=multispec"
    header["WAT1_001"] = "wtype=multispec label=Wavelength units=angstroms"
    specs = " ".join(f'spec{ap} = "{ap} {ap} 0 {w1} {dw} {N} 0. {ap * 10 - 3}. {ap * 10 + 3}."'
                     for ap, w1, dw, _ in rows)
    text = "wtype=multispec " + specs
    for i in range(0, len(text), 68):
        header[f"WAT2_{i // 68 + 1:03d}"] = text[i:i + 68]
    for ap, *_ in rows:
        header[f"APNUM{ap}"] = f"{ap} {ap} {ap * 10 - 3}. {ap * 10 + 3}."
    data = np.vstack([flux for *_, flux in rows]).astype(np.float32)
    fits.writeto(folder / "in.fits", data, header, overwrite=True)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
