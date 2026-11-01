"""The images and the translation file of ccdinstrument's gabarito, in harwood's python."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

INST = """# test
exptime   itime
subset    FILTERS  "no filter"
imagetyp  IMAGETYP
OBJECT    object
'flat frame' flat
flatcor   FLATCOR  "no"
darkcor   DARKCOR  no
ncombine  NCOMBINE 1
"""


def write(folder):
    folder = Path(folder)
    (folder / "inst.dat").write_text(INST)
    h = fits.Header()
    for key, value in (
        ("OBJECT", "M 92 V"), ("IMAGETYP", "OBJECT"), ("EXPTIME", 120.5), ("DARKTIME", 121),
        ("FILTERS", "V band"), ("CCDSEC", "[1:10,1:10]"), ("FLATCOR", "done"),
        ("BIASSEC", "[1:2,1:10]"), ("TOOBIG", "a" * 60), ("ZEROCOR", "yes"), ("TEL", True),
        ("DATE-OBS", "1992-06-10"), ("HISTORY", "hello"), ("COMMENT", "c   "),
    ):  # fmt: skip
        h[key] = value
    data = np.zeros((10, 10), "f4")
    fits.writeto(folder / "a.fits", data, h, overwrite=True)
    h["IMAGETYP"], h["FILTERS"], h["OBJECT"] = "flat frame", "R band", ""
    fits.writeto(folder / "b.fits", data, h, overwrite=True)
    h["IMAGETYP"] = "weird"
    h["FILTERS"] = "V, x/y"
    del h["OBJECT"]
    fits.writeto(folder / "c.fits", data, h, overwrite=True)
    del h["IMAGETYP"]
    del h["FILTERS"]
    fits.PrimaryHDU((data + 5).astype("i2"), h, scale_back=False).writeto(
        folder / "d.fits", overwrite=True
    )
    scaled = fits.PrimaryHDU((data + 5).astype("u2"))
    scaled.writeto(folder / "e.fits", overwrite=True)
    primary = fits.PrimaryHDU(header=h)
    extension = fits.ImageHDU(data, name="SCI")
    extension.header["INHERIT"] = True
    extension.header["GAIN"] = 2.5
    fits.HDUList([primary, extension]).writeto(folder / "m.fits", overwrite=True)


if __name__ == "__main__":
    write(sys.argv[1])
