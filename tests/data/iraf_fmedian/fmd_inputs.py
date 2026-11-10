"""The images of IRAF's fmedian and boxcar gabaritos: python fmd_inputs.py <folder>
(harwood's venv, for astropy). median's (../iraf_median/med_inputs.py), and two
more: one whose DATAMIN and DATAMAX are not its pixels' limits, which fmedian takes
from the header when they are there, and a cube, which boxcar refuses."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "iraf_median"))
from med_inputs import _field  # noqa: E402
from med_inputs import write as write_median  # noqa: E402


def write(where: Path) -> None:
    write_median(where)
    hdu = fits.PrimaryHDU(_field(9, 11, 9).astype(np.float32))
    hdu.header["DATAMIN"] = -100.0
    hdu.header["DATAMAX"] = 300.0
    hdu.writeto(where / "lim.fits", overwrite=True)
    fits.PrimaryHDU(np.stack([_field(6, 7, k) for k in range(3)]).astype(np.float32)
                    ).writeto(where / "cube.fits", overwrite=True)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
