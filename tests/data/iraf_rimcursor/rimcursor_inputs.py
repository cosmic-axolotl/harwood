"""The images of rimcursor's gabarito, in harwood's python."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


def write(folder):
    folder = Path(folder)
    data = np.zeros((30, 60), "f4")
    tan = fits.Header()
    for key, value in (
        ("WCSDIM", 2), ("CTYPE1", "RA---TAN"), ("CTYPE2", "DEC--TAN"), ("CRVAL1", 150.25),
        ("CRVAL2", -33.5), ("CRPIX1", 30.5), ("CRPIX2", 15.5), ("CD1_1", -1.2e-4),
        ("CD1_2", 2.0e-5), ("CD2_1", 2.0e-5), ("CD2_2", 1.2e-4),
    ):  # fmt: skip
        tan[key] = value
    fits.writeto(folder / "tan.fits", data, tan, overwrite=True)
    lin = fits.Header()
    for key, value in (
        ("WCSDIM", 2), ("CTYPE1", "LINEAR"), ("CTYPE2", "LINEAR"), ("CRVAL1", 4000.0),
        ("CRVAL2", 7.0), ("CRPIX1", 1.0), ("CRPIX2", 1.0), ("CD1_1", 2.5), ("CD2_2", 0.5),
        ("LTM1_1", 0.5), ("LTM2_2", 0.5), ("LTV1", 3.0), ("LTV2", -2.0),
    ):  # fmt: skip
        lin[key] = value
    fits.writeto(folder / "lin.fits", data, lin, overwrite=True)
    attrs = tan.copy()
    attrs["WAT0_001"] = "system=world"
    attrs["WAT1_001"] = 'wtype=tan axtype=ra format="%10.3H"'
    attrs["WAT2_001"] = 'wtype=tan axtype=dec format="%9.2h"'
    fits.writeto(folder / "attrs.fits", data, attrs, overwrite=True)


if __name__ == "__main__":
    write(sys.argv[1])
