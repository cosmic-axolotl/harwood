"""The images of imcombine's offsets, masks and project gabarito, in harwood's python."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from harwood.core import plio

DITHER = [(0, 0), (3, -2), (-4, 5), (7, 2), (-1, -6)]
SKY = [100.0, 130.0, 90.0, 210.0, 115.0]
EXPTIME = [10.0, 12.5, 8.0, 30.0, 11.0]


def frame(k, ny=30, nx=40):
    rng = np.random.default_rng(100 + k)
    y, x = np.mgrid[1:ny + 1, 1:nx + 1]
    dx, dy = DITHER[k]
    star = 900.0 * np.exp(-((x - 15 + dx) ** 2 + (y - 12 + dy) ** 2) / 8.0)
    star += 400.0 * np.exp(-((x - 30 + dx) ** 2 + (y - 22 + dy) ** 2) / 6.0)
    data = SKY[k] + star * EXPTIME[k] / 10.0 + rng.normal(0, 6.0, (ny, nx))
    data[(3 + k) * 3 % ny, (5 + 2 * k) * 4 % nx] += 4000.0
    data[(8 + k) % ny, (20 + k) % nx] = -50.0
    return np.round(data, 2).astype("f4")


def mask(k, ny=30, nx=40):
    out = np.zeros((ny, nx), "i4")
    out[2 + k:5 + k, 4:9] = 1
    out[15:17, 10 + k:14 + k] = 2
    out[20, 25:30] = 3
    out[4:7, 30 + k] = 5
    return out


def write(folder):
    folder = Path(folder)
    names = []
    for k in range(5):
        h = fits.Header()
        h["OBJECT"] = f"frame {k}"
        h["EXPTIME"] = EXPTIME[k]
        h["RDN"] = 5.0 + k
        h["GAINK"] = 1.5 + 0.25 * k
        h["SNK"] = 0.01 * k
        h["SCL"] = 1.0 + 0.1 * k
        h["ZER"] = 5.0 * k
        h["WGT"] = 1.0 + k
        dx, dy = DITHER[k]
        h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
        h["CRVAL1"], h["CRVAL2"] = 150.25, -33.5
        h["CRPIX1"], h["CRPIX2"] = 20.25 + dx, 15.5 + dy
        h["CD1_1"], h["CD1_2"] = -1.2e-4, 0.0
        h["CD2_1"], h["CD2_2"] = 0.0, 1.2e-4
        h["LTV1"], h["LTV2"] = float(dx), float(dy)
        if k != 3:
            h["BPM"] = f"m{k}.pl"
        h["MSK"] = f"a{k}.pl"
        fits.writeto(folder / f"f{k}.fits", frame(k), h, overwrite=True)
        plio.write(folder / f"m{k}.pl", mask(k))
        plio.write(folder / f"a{k}.pl", np.zeros((30, 40), "i4") if k == 4 else mask(4 - k))
        names.append(f"f{k}.fits")
    (folder / "all.lis").write_text("\n".join(names) + "\n")
    (folder / "offs.txt").write_text("".join(f"{dx} {dy}\n" for dx, dy in DITHER))
    (folder / "half.txt").write_text("0.4 0.5\n3.5 -2.5\n-4.5 5.4\n6.6 2.2\n-1 -6\n")
    (folder / "abs.txt").write_text("# absolute\n2 3\n5 1\n1 8\n9 5\n2 2\n")
    (folder / "scale.txt").write_text("1.0\n1.25\n# a comment\n0.8\n2.5\n1.1\n")
    (folder / "zero.txt").write_text("0\n10\n-5\n40\n7\n")
    (folder / "wgt.txt").write_text("1\n2\n3\n4\n5\n")
    # Of different sizes, no offsets
    for k, shape in enumerate(((30, 40), (28, 45), (33, 35))):
        fits.writeto(folder / f"g{k}.fits", frame(k, *shape), overwrite=True)
    (folder / "sizes.lis").write_text("g0.fits\ng1.fits\ng2.fits\n")
    # One dimension
    for k in range(5):
        fits.writeto(folder / f"o{k}.fits", frame(k)[12], overwrite=True)
    (folder / "line.lis").write_text("".join(f"o{k}.fits\n" for k in range(5)))
    # A cube, and its mask
    h = fits.Header()
    h["EXPTIME"] = 20.0
    h["BPM"] = "cubem.pl"
    h["CRPIX1"], h["CRPIX2"], h["CRPIX3"] = 5.0, 6.0, 1.0
    h["CRVAL1"], h["CRVAL2"], h["CRVAL3"] = 10.0, 20.0, 30.0
    h["CD1_1"], h["CD2_2"], h["CD3_3"] = 0.5, 0.25, 2.0
    h["CTYPE1"], h["CTYPE2"], h["CTYPE3"] = "LINEAR", "LINEAR", "LINEAR"
    cube = np.stack([frame(k) for k in range(5)])
    fits.writeto(folder / "cube.fits", cube, h, overwrite=True)
    plio.write(folder / "cubem.pl", np.stack([mask(k) for k in range(5)]))


if __name__ == "__main__":
    write(sys.argv[1])
