"""The frames of IRAF's cosmicrays gabarito, in harwood's python."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

COSMIC = [(12, 15, 900.0), (50, 20, 400.0), (30, 40, 250.0), (70, 8, 600.0), (20, 45, 120.0),
          (65, 50, 2000.0), (40, 25, 80.0), (3, 3, 500.0), (79, 59, 700.0), (45, 2, 300.0)]
PAIRS = [(55, 33, 800.0), (56, 33, 700.0), (10, 30, 500.0), (10, 31, 450.0)]
STARS = [(25.3, 12.7, 900.0, 2.0), (60.8, 38.2, 2500.0, 1.6), (15.1, 52.4, 400.0, 1.2),
         (42.6, 48.9, 6000.0, 2.2), (72.2, 25.5, 300.0, 3.0)]


def write(folder):
    folder = Path(folder)
    rng = np.random.default_rng(31)
    ny, nx = 60, 80
    y, x = np.mgrid[1:ny + 1, 1:nx + 1].astype(float)
    data = 100.0 + 0.3 * x - 0.2 * y + rng.normal(0, 4.0, (ny, nx))
    for a, b, flux, sigma in STARS:
        data += flux * np.exp(-((x - a) ** 2 + (y - b) ** 2) / (2 * sigma ** 2)) / (2 * np.pi
                                                                                  * sigma ** 2)
    for column, line, value in COSMIC + PAIRS:
        data[line - 1, column - 1] += value
    fits.writeto(folder / "f.fits", np.round(data, 2).astype("f4"), overwrite=True)
    fits.writeto(folder / "s.fits", np.round(data).astype("i2"), overwrite=True)
    (folder / "train.cur").write_text(_cosmics(5) + _stars(4) + "1 1 1 q\n")
    (folder / "traing.cur").write_text(_cosmics(3) + "1 1 1 g\n" + _stars(3) + "1 1 1 g\n"
                                       + _cosmics(5) + "1 1 1 q\n")
    (folder / "trainm.cur").write_text(
        "30.5 3.25 1 s\n" + _cosmics(2) + "44 3 1 c\n1 1 1 x\n"
        "70 58 1 s\n1 1 1 g\n60.8 38.2 1 s\n43 49 1 s\n1 1 1 q\n")


def _cosmics(count):
    return "".join(f"{column} {line} 1 c\n" for column, line, _ in COSMIC[:count])


def _stars(count):
    return "".join(f"{a:.0f} {b:.0f} 1 s\n" for a, b, _, _ in STARS[:count])


if __name__ == "__main__":
    write(sys.argv[1])
