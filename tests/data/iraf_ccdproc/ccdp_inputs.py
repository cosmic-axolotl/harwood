"""The raw frames of ccdproc's gabarito, in harwood's python."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from harwood.core import plio

NY, NX = 40, 60  # with a strip of 8 columns at the end, and 3 lines at the top


def raw(kind, seed, level, exptime=0.0, shape=(NY, NX), column=False, extra=None):
    rng = np.random.default_rng(seed)
    ny, nx = shape
    y, x = np.mgrid[1:ny + 1, 1:nx + 1]
    pattern = 300.0 + 0.1 * y + 0.03 * x + 3.0 * np.sin(x / 7.0)
    light = level * (1.0 + 0.05 * np.cos(x / 11.0) * np.cos(y / 9.0))
    data = pattern + light + rng.normal(0, 2.5, (ny, nx))
    if column:
        data[-5:, :] = 305.0 + 0.2 * x[-5:, :] + rng.normal(0, 2.0, (5, nx))
    else:
        data[:, -8:] = 305.0 + 0.2 * y[:, -8:] + rng.normal(0, 2.0, (ny, 8))
    data[5, 7] += 3000.0
    h = fits.Header()
    h["OBJECT"] = kind
    h["IMAGETYP"] = kind
    h["EXPTIME"] = exptime
    h["DARKTIME"] = exptime + 1.5
    h["FILTER"] = "V"
    h["CTYPE1"], h["CTYPE2"] = "LINEAR", "LINEAR"
    h["CRPIX1"], h["CRPIX2"] = 20.5, 10.5
    h["CRVAL1"], h["CRVAL2"] = 100.0, 200.0
    h["CD1_1"], h["CD2_2"] = 0.5, 0.25
    for key, value in (extra or {}).items():
        h[key] = value
    return np.round(data, 2).astype("f4"), h


def write(folder):
    folder = Path(folder)

    def put(name, *args, **kw):
        data, h = raw(*args, **kw)
        fits.writeto(folder / name, data, h, overwrite=True)

    put("zero.fits", "zero", 1, 0.0)
    put("dark.fits", "dark", 2, 100.0)
    put("flat.fits", "flat", 3, 8000.0, 5.0)
    put("obj.fits", "object", 4, 900.0, 60.0)
    put("comp.fits", "comp", 5, 400.0, 20.0)
    put("other.fits", "other", 6, 500.0, 20.0)
    put("notype.fits", "stuff", 7, 500.0, 20.0)
    put("illumraw.fits", "illum", 8, 1500.0, 10.0)
    put("fringeraw.fits", "fringe", 9, 900.0, 30.0)
    # What mkillumcor and mkfringecor leave
    y, x = np.mgrid[1:38, 1:53]
    h = fits.Header()
    h["IMAGETYP"] = "illum"
    h["MKILLUM"] = "yes"
    h["CCDSEC"] = "[1:52,1:37]"
    fits.writeto(folder / "illum.fits", (1.0 + 0.002 * x - 0.001 * y).astype("f4"), h,
                 overwrite=True)
    h["CCDMEAN"] = 1.5
    fits.writeto(folder / "illummean.fits", (1.0 + 0.002 * x - 0.001 * y).astype("f4"), h,
                 overwrite=True)
    del h["MKILLUM"]
    fits.writeto(folder / "illumnoflag.fits", (1.0 + 0.002 * x).astype("f4"), h,
                 overwrite=True)
    h = fits.Header()
    h["IMAGETYP"] = "fringe"
    h["MKFRINGE"] = "yes"
    h["EXPTIME"] = 30.0
    h["CCDSEC"] = "[1:52,1:37]"
    fits.writeto(folder / "fringe.fits", (4.0 * np.sin(x / 3.0 + y / 5.0)).astype("f4"), h,
                 overwrite=True)
    h["FRINGSCL"] = 0.8
    fits.writeto(folder / "fringescl.fits", (4.0 * np.sin(x / 3.0 + y / 5.0)).astype("f4"), h,
                 overwrite=True)
    del h["MKFRINGE"]
    fits.writeto(folder / "fringenoflag.fits", (4.0 * np.sin(x / 3.0)).astype("f4"), h,
                 overwrite=True)
    # A frame read out along the columns: its strip across the top
    for name, kind, seed, level, exptime in (("czero.fits", "zero", 11, 0.0, 0.0),
                                             ("cflat.fits", "flat", 12, 8000.0, 5.0),
                                             ("cobj.fits", "object", 13, 900.0, 60.0)):
        put(name, kind, seed, level, exptime, column=True)
    # Part of the chip: a data section, and a CCD section in it
    put("sub.fits", "object", 14, 900.0, 60.0,
        extra={"DATASEC": "[3:52,2:37]", "CCDSEC": "[3:52,2:37]"})
    put("sub2.fits", "object", 15, 900.0, 60.0,
        extra={"DATASEC": "[1:30,1:20]", "CCDSEC": "[11:40,6:25]"})
    # From the header
    put("hdr.fits", "object", 16, 900.0, 60.0,
        extra={"BIASSEC": "[53:60,*]", "TRIMSEC": "[1:52,1:37]"})
    # Scanned
    put("scan.fits", "object", 17, 900.0, 60.0, extra={"NSCANROW": 4})
    # Calibrations with their own CCD sections, wider than the frame's
    part = {"CCDSEC": "[6:55,3:37]", "DATASEC": "[2:51,4:38]", "CCDMEAN": 8500.0,
            "OVERSCAN": "done", "TRIM": "done", "ZEROCOR": "done", "CCDMEANT": 2000000000}
    put("zero2.fits", "zero", 18, 0.0, 0.0, extra=part)
    put("flat2.fits", "flat", 19, 8000.0, 5.0, extra=part)
    y, x = np.mgrid[1:41, 1:61]
    h = fits.Header()
    h["MKILLUM"] = "yes"
    h["CCDMEAN"] = 1.25
    h["CCDMEANT"] = 2000000000
    h["CCDSEC"] = part["CCDSEC"]
    h["DATASEC"] = part["DATASEC"]
    fits.writeto(folder / "illum2.fits", (1.0 + 0.002 * x - 0.001 * y).astype("f4"), h,
                 overwrite=True)
    h = fits.Header()
    h["MKFRINGE"] = "yes"
    h["EXPTIME"] = 30.0
    h["CCDSEC"] = part["CCDSEC"]
    h["DATASEC"] = part["DATASEC"]
    fits.writeto(folder / "fringe2.fits", (4.0 * np.sin(x / 3.0 + y / 5.0)).astype("f4"), h,
                 overwrite=True)
    mask = np.zeros((NY, NX), "i4")
    mask[10:14, 20] = 1
    mask[20, 30:36] = 1
    mask[0:3, 5] = 1
    mask[36:40, 40] = 1
    mask[15, 55] = 1
    plio.write(folder / "bad.pl", mask)
    fits.writeto(folder / "bad.fits", mask.astype("i2"), overwrite=True)


if __name__ == "__main__":
    write(sys.argv[1])
