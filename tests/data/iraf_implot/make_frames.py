"""The images of the implot graph cases, here: python make_frames.py [DIR]."""
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

where = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
yy, xx = np.indices((60, 80))
a = (500 + 2 * xx + 3 * yy + 40 * np.sin(xx / 4.0) * np.cos(yy / 5.0)
     + (xx * 7 + yy * 13) % 9).astype(np.float32)
b = (100 + xx * yy / 10.0 + 5 * np.cos(xx / 3.0)).astype(np.float32)
fits.PrimaryHDU(a, fits.Header({"LTV1": -2.0, "LTV2": -1.0, "OBJECT": "synthetic a"})
                ).writeto(where / "sa.fits", overwrite=True)
fits.PrimaryHDU(b, fits.Header({"OBJECT": "synthetic b"})).writeto(where / "sb.fits",
                                                                   overwrite=True)
star = (50 + 1000 * np.exp(-0.5 * (((xx - 39.3) / 2.5) ** 2 + ((yy - 24.6) / 2.5) ** 2))
        + 0.02 * xx).astype(np.float32)
fits.PrimaryHDU(star, fits.Header({"OBJECT": "a star"})).writeto(where / "star.fits",
                                                                 overwrite=True)
fits.PrimaryHDU(np.full((60, 80), 7.0, dtype=np.float32)).writeto(where / "flat.fits",
                                                                  overwrite=True)
wave = (200 + 80 * np.sin(np.arange(300) / 20.0)).astype(np.float32)
fits.PrimaryHDU(wave, fits.Header({
    "OBJECT": "a spectrum", "CRVAL1": 4000.0, "CRPIX1": 1.0, "CD1_1": 2.5, "CDELT1": 2.5,
    "CTYPE1": "LINEAR", "DC-FLAG": 0, "WCSDIM": 1, "LTM1_1": 1.0,
    "WAT0_001": "system=world",
    "WAT1_001": 'wtype=linear label="Rest wavelength" units=Angstroms'})
                ).writeto(where / "spec.fits", overwrite=True)
