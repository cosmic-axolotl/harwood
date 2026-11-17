"""The frames of pkg_cases.py: ccdp_inputs' and the object in four pixel types, a taken
backup name, a backup directory and a text bad pixel file."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from ccdp_inputs import write as base

TYPES = {"sobj": np.int16, "uobj": np.uint16, "iobj": np.int32, "dobj": np.float64}


def write(folder) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base(folder)
    data, header = fits.getdata(folder / "obj.fits", header=True)
    for name, dtype in TYPES.items():
        pixels = np.round(data).astype(dtype)
        fits.writeto(folder / f"{name}.fits", pixels, header, overwrite=True)
    fits.writeto(folder / "Bobj.fits", np.zeros((2, 2), np.float32), overwrite=True)
    (folder / "bak").mkdir(exist_ok=True)
    (folder / "badpix.dat").write_text("10 10 1 37\n20 22 5 8\n30 30 30 30\n",
                                       encoding="utf-8")


if __name__ == "__main__":
    write(sys.argv[1])
