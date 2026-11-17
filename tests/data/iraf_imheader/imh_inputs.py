"""The images of IRAF's imheader gabarito (python imh_inputs.py <folder>): one per pixel
type, one-dimensional, untitled, an extension, and a header with what a real one has
-- EXTEND = F, HISTORY, COMMENT, a long string, a blank keyword."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

TYPES = {"b8": np.uint8, "i16": np.int16, "u16": np.uint16, "i32": np.int32,
         "f32": np.float32, "f64": np.float64}


def write(where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    for name, dtype in TYPES.items():
        header = fits.Header()
        header["OBJECT"] = f"test {name}"
        fits.writeto(where / f"{name}.fits", np.zeros((3, 4), dtype), header, overwrite=True)
    fits.writeto(where / "one.fits", np.zeros(5, np.float32), overwrite=True)
    fits.writeto(where / "notitle.fits", np.zeros((2, 2), np.float32), overwrite=True)
    primary = fits.PrimaryHDU()
    primary.header["OBSERVAT"] = "KPNO"
    fits.HDUList([primary, fits.ImageHDU(np.zeros((3, 4), np.float32), name="SCI")]
                 ).writeto(where / "mef.fits", overwrite=True)
    header = fits.Header()
    header["EXTEND"] = (False, "File may contain extensions")
    header["OBJECT"] = ("NGC 660 PA42 #2", "")
    header["EXPTIME"] = (900.0, "Elapsed exposure time in seconds")
    header["DATE-OBS"] = "19/12/85"
    header["LONGSTR"] = "a string long enough to fill most of the card, as some do"
    header["GAIN"] = 1.5
    header.add_history("19/12/85          /Date tape file created")
    header.add_comment("a comment card")
    header.append(("", "a blank keyword"), end=True)
    fits.writeto(where / "real.fits", np.zeros((5, 6), np.int16), header, overwrite=True)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
