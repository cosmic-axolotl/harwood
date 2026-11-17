"""The images of IRAF's imcopy gabarito (python icp_inputs.py <folder>): imheader's, one of
each pixel type, and a 16-bit image scaled by BSCALE and BZERO."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "iraf_imheader"))
from imh_inputs import write as base  # noqa: E402


def write(where: Path) -> None:
    base(where)
    rng = np.random.default_rng(3)
    for name, dtype in (("b8", np.uint8), ("i16", np.int16), ("u16", np.uint16),
                        ("i32", np.int32), ("f32", np.float32), ("f64", np.float64)):
        header = fits.getheader(where / f"{name}.fits")
        data = rng.integers(0, 200, size=(3, 4)).astype(dtype)
        if dtype == np.uint16:
            data = data + np.uint16(40000)
        if dtype in (np.float32, np.float64):
            data = data + np.asarray(0.37, dtype)
        fits.writeto(where / f"{name}.fits", data, header, overwrite=True)
    hdu = fits.PrimaryHDU(np.arange(12, dtype=np.int16).reshape(3, 4))
    hdu.header["BSCALE"] = 0.5
    hdu.header["BZERO"] = 10.0
    hdu.writeto(where / "scaled.fits", overwrite=True)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
