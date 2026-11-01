"""The image of imexamine's other keys: tests/iraf_field.field()."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parents[2]))
from iraf_field import field  # noqa: E402

if __name__ == "__main__":
    header = fits.Header()
    header["OBJECT"] = "stars"
    header["CTYPE1"], header["CTYPE2"] = "LINEAR", "LINEAR"
    header["CRPIX1"], header["CRPIX2"], header["CRVAL1"], header["CRVAL2"] = 1.0, 1.0, 1.0, 1.0
    pixels = np.asarray(field(), "f4")
    fits.writeto(Path(sys.argv[1]) / "f.fits", pixels, header, overwrite=True)
