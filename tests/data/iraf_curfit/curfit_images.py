"""The images of curfit's gabarito, in harwood's python."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


def write(folder):
    folder = Path(folder)
    rng = np.random.default_rng(5)
    two = 100 + 3.3 * np.arange(50)[None, :] + rng.normal(0, 4, (30, 50))
    two[3, :] = -200
    three = 7 + np.arange(24 * 3).reshape(3, 4, 6).astype("f4") * 0.37
    fits.writeto(folder / "fits.fits", two.astype("f4"), overwrite=True)
    fits.writeto(folder / "cube.fits", three, overwrite=True)


if __name__ == "__main__":
    write(sys.argv[1])
