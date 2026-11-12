"""The inputs of IRAF's graph gabarito: python gr_inputs.py <folder> (harwood's venv).

Text lists -- x y pairs with every kind of line gg_rdlist2 meets (comments, blank
lines, an indented comment, one value where two are expected, INDEF, h:m:s, a d
exponent, words after the numbers, a comma), a column of y alone, x y and a mark
size, values at and under zero for log axes, a constant, three thousand points
(past the 2048 of its first buffer) -- and images: a 1-D spectrum with a labelled
linear world system, a 2-D frame with a rotated CD and an LTV (a physical system
that is not the logical), a cube, a frame with no world system, and an axis of
right ascension with an %H format."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

TWO = """\
# x and y
1 2
2 4.5

3 3
   # an indented comment is not one
4
5 1:30
6 1.5e1
7 1d1
8 INDEF
9 6 words after
10,7
11 2
"""

ONE = "\n".join(f"{v:g}" for v in (3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5)) + "\n"

THREE = """\
1 1 0.02
2 3 0.01
3 2
4 5 0.03
5 4 0.015
"""

NEG = "1 -5\n2 0\n3 0.5\n4 20\n5 300\n6 -40\n"
POS = "1 0.2\n2 3\n3 45\n4 700\n5 9000\n"
FLAT = "1 5\n2 5\n3 5\n"
POINT = "7 3\n"


def write(where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    for name, text in (("two.txt", TWO), ("one.txt", ONE), ("three.txt", THREE),
                       ("neg.txt", NEG), ("pos.txt", POS), ("flat.txt", FLAT),
                       ("point.txt", POINT)):
        (where / name).write_text(text, encoding="utf-8")
    x = np.arange(3000)
    (where / "big.txt").write_text(
        "".join(f"{i} {np.sin(i / 150.0) * 100 + i * 0.01:.5f}\n" for i in x),
        encoding="utf-8")

    rng = np.random.default_rng(11)
    n = 120
    w = 4000.0 + 2.5 * np.arange(n)
    spec = (500 + 0.2 * (w - 4000) + 300 * np.exp(-0.5 * ((w - 4150) / 4) ** 2)
            + rng.normal(0, 5, n)).astype(np.float32)
    one = fits.PrimaryHDU(spec)
    one.header.update({"OBJECT": "a spectrum", "CRVAL1": 4000.0, "CRPIX1": 1.0,
                       "CDELT1": 2.5, "CD1_1": 2.5, "CTYPE1": "LINEAR", "WCSDIM": 1,
                       "WAT0_001": "system=world",
                       "WAT1_001": "wtype=linear label=Wavelength units=Angstroms"})
    one.writeto(where / "spec.fits", overwrite=True)

    yy, xx = np.mgrid[1:21, 1:31]
    frame = (100 + 3 * xx + 0.5 * yy * yy
             + 800 * np.exp(-0.5 * (((xx - 12) / 2.5) ** 2 + ((yy - 9) / 3.0) ** 2))
             + rng.normal(0, 2, xx.shape)).astype(np.float32)
    f = fits.PrimaryHDU(frame)
    f.header.update({"OBJECT": "a frame", "WCSDIM": 2, "CTYPE1": "LINEAR",
                     "CTYPE2": "LINEAR", "CRVAL1": 10.0, "CRVAL2": -3.0, "CRPIX1": 4.0,
                     "CRPIX2": 2.5, "CD1_1": 0.75, "CD1_2": 0.05, "CD2_1": -0.02,
                     "CD2_2": 1.5, "LTV1": -2.0, "LTV2": 5.0, "LTM1_1": 1.0,
                     "LTM2_2": 1.0, "WAT0_001": "system=world",
                     "WAT1_001": "wtype=linear label=Position units=mm",
                     "WAT2_001": "wtype=linear label=Height"})
    f.writeto(where / "frame.fits", overwrite=True)

    cube = (np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6) * 1.5
            + rng.normal(0, 1, (4, 5, 6))).astype(np.float32)
    c = fits.PrimaryHDU(cube)
    c.header["OBJECT"] = "a cube"
    c.writeto(where / "cube.fits", overwrite=True)

    bare = fits.PrimaryHDU((frame[:8, :10] * 0.1).astype(np.float32))
    bare.writeto(where / "bare.fits", overwrite=True)

    ra = fits.PrimaryHDU((spec[:40] - 400).astype(np.float32))
    ra.header.update({"OBJECT": "right ascension", "CRVAL1": 150.0, "CRPIX1": 1.0,
                      "CDELT1": 0.01, "CD1_1": 0.01, "CTYPE1": "LINEAR", "WCSDIM": 1,
                      "WAT0_001": "system=world",
                      "WAT1_001": "wtype=linear label=RA units=hours format=%H"})
    ra.writeto(where / "ra.fits", overwrite=True)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
