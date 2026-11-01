"""The frames and lists the mask tasks are run on, in IRAF and in the tests."""

from pathlib import Path

import numpy as np
from astropy.io import fits

REGIONS = """# a comment
3 3
5 7 2 2
9 9 4 6
2 4 7 9
-5 2 1 1
10 30 8 8
40 50 1 2
a b
7
1 2 3
12 12 12 12 99
6.7 3.2
8, 9
15 15 3 8
15 15 5 5
"""
LINES = """# rectangles of one line, which is all IRAF's .pl keeps whole
3 3
5 7 2 2
9 9 4 4
2 4 7 7
-5 2 1 1
10 30 8 8
40 50 1 1
a b
7
1 2 3
12 12 12 12 99
6.7 3.2
8, 9
3 7 2 2
4 5 2 2
"""
REGIONS_DB = """# Sample MSKEXPR expression database file.
RADIUS=     4.5
bdata1      (i < -100 || i > 25000)
cmpie(xc,yc,r,t1,t2)    circle (xc, yc, r) && (! pie (xc, yc, t1, t2))
"""
BADPIX = """# the bad pixels
3 3
5 7 2 2
9 9 4 6
-5 2 1 1
40 50 1 2
x y
"""
UNTRIMMED = """# coordinates of the untrimmed frame
12 12
13 15 9 9
1 5 1 1
20 22 30 40
"""


def flat(nl, nc, seed):
    """A flat field with the defects ccdmask is there for."""
    rng = np.random.default_rng(seed)
    a = 1000 + rng.normal(0, 10, (nl, nc)) + 0.2 * np.arange(nc) + 0.1 * np.arange(nl)[:, None]
    a[:, nc // 3] -= 400                      # a dead column
    a[nl // 2:, nc // 2] += 20                # a trap too weak pixel by pixel
    a[nl // 4, :] *= 0.5                      # a bad line
    a[10:13, 20:24] = 0                       # a square
    a[nl // 2, nc // 4] = 9000                # a hot pixel
    a[5:9, nc - 1] = 3000                     # on the last column
    a[nl - 3:, 40] = 50
    return a.astype(np.float32)


def write(where):
    where = Path(where)
    where.mkdir(parents=True, exist_ok=True)
    (where / "regions.txt").write_text(REGIONS)
    (where / "lines.txt").write_text(LINES)
    (where / "badpix.txt").write_text(BADPIX)
    (where / "untrimmed.txt").write_text(UNTRIMMED)
    (where / "empty.txt").write_text("")
    (where / "regions.db").write_text(REGIONS_DB)
    (where / "expr.dat").write_text("cannulus (12., 9., 3., 7.) ? 1 : 0\n")
    h = fits.Header({"OBJECT": "a template", "EXPTIME": 30.0, "DATAMAX": 5.0})
    fits.PrimaryHDU(np.zeros((12, 20), np.float32), h).writeto(where / "tmpl.fits",
                                                               overwrite=True)
    h = fits.Header({"TRIM": "Trim data section is [11:40:1,5:30:1]"})
    fits.PrimaryHDU(np.zeros((26, 30), np.float32), h).writeto(where / "trim.fits",
                                                               overwrite=True)
    h = fits.Header({"TRIM": "Trim data section is [11:40:2,5:30:2]"})
    fits.PrimaryHDU(np.zeros((13, 15), np.float32), h).writeto(where / "trim2.fits",
                                                               overwrite=True)
    fits.PrimaryHDU(flat(100, 120, 1)).writeto(where / "flat.fits", overwrite=True)
    fits.PrimaryHDU(flat(100, 120, 2), fits.Header({"OBJECT": "s"})).writeto(
        where / "flats.fits", overwrite=True)
    fits.PrimaryHDU(flat(1000, 600, 3)).writeto(where / "wide.fits", overwrite=True)
    fits.PrimaryHDU(flat(1000, 1100, 4)).writeto(where / "wider.fits", overwrite=True)
    fits.PrimaryHDU(flat(40, 60, 5)).writeto(where / "small.fits", overwrite=True)
    fits.PrimaryHDU(np.round(flat(100, 120, 6)).astype(np.int16)).writeto(where / "short.fits",
                                                                          overwrite=True)
    fits.PrimaryHDU(np.arange(240, dtype=np.float32).reshape(12, 20) * 1.5,
                    fits.Header({"OBJECT": "ramp", "EXPTIME": 12.0})).writeto(
        where / "ramp.fits", overwrite=True)
