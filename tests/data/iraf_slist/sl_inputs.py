"""The spectra the slist cases list, as IRAF's multispec (python sl_inputs.py
<folder>, harwood's venv) and as harwood's tables (spectra(name)): `in`, three
apertures, linear, logarithmic and linear, with the header's times and place; `pix`,
one aperture still in pixels."""

import sys
from pathlib import Path

import numpy as np

N = 120
HEADER = {"OBJECT": "M 31", "EXPTIME": 1200.0, "UT": "03:12:45.5", "ST": "01:40:12.3",
          "RA": "00:42:44.3", "DEC": "41:16:09", "HA": "-00:58:10.2", "AIRMASS": 1.084}
#: (aperture, beam, dc, w1, dw, title): dw and w1 of a log aperture in log10.
APERTURES = {"in": [(1, 1, 0, 4010.0, 1.6, "first"), (2, 2, 1, 3.6021, 0.0002, "second"),
                    (3, 5, 0, 4030.0, 1.8, "")],
             "pix": [(4, 4, -1, 1.0, 1.0, "arc")]}


def wavelengths(dc, w1, dw):
    pix = np.arange(N, dtype=float)
    if dc == 1:
        return 10 ** (w1 + dw * pix)
    if dc == 0:
        return w1 + dw * pix
    return None


def spectra(name):
    """(aperture, beam, dc, w1, dw, title, flux, wavelength) of each aperture."""
    pix = np.arange(N, dtype=float)
    return [(ap, beam, dc, w1, dw, title,
             (200.0 + 20.0 * np.sin(pix / 5.0 + ap)).astype(np.float32),
             wavelengths(dc, w1, dw))
            for ap, beam, dc, w1, dw, title in APERTURES[name]]


def write(folder: Path) -> None:
    from astropy.io import fits

    for name, rows in APERTURES.items():
        header = fits.Header()
        for key, value in HEADER.items():
            header[key] = value
        header["WCSDIM"] = 2
        header["CTYPE1"], header["CTYPE2"] = "MULTISPE", "MULTISPE"
        header["CD1_1"], header["CD2_2"] = 1.0, 1.0
        header["LTM1_1"], header["LTM2_2"] = 1.0, 1.0
        header["WAT0_001"] = "system=multispec"
        header["WAT1_001"] = "wtype=multispec label=Wavelength units=angstroms"
        text = "wtype=multispec " + " ".join(
            f'spec{i} = "{ap} {beam} {dc} {w1!r} {dw!r} {N} 0. {ap * 10 - 3}. {ap * 10 + 3}."'
            for i, (ap, beam, dc, w1, dw, _) in enumerate(rows, 1))
        for i in range(0, len(text), 68):
            header[f"WAT2_{i // 68 + 1:03d}"] = text[i:i + 68]
        for i, (ap, beam, _, _, _, title) in enumerate(rows, 1):
            header[f"APNUM{i}"] = f"{ap} {beam} {ap * 10 - 3}. {ap * 10 + 3}."
            if title:
                header[f"APID{i}"] = title
        data = np.vstack([s[6] for s in spectra(name)]).astype(np.float32)
        fits.writeto(folder / f"{name}.fits", data, header, overwrite=True)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
