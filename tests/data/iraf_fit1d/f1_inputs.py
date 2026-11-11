"""The images of IRAF's fit1d and background gabarito: python f1_inputs.py <folder>
(harwood's venv, for astropy).

sky: a rectified long-slit frame, 61 columns by 120 lines, a galaxy in the middle
columns on a sky that varies along the slit, with noise and a few cosmic rays --
so that some lines reject points and others do not; spec: a 1-D spectrum; bpm: a
mask of a bad column and a few pixels; old: an existing image to be written into."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

NX, NY = 61, 120


def write(where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(660)
    y, x = np.indices((NY, NX)) + 1.0
    sky = 200 + 40 * np.sin(y / 9.0) + (x - 31) * (0.3 + 0.002 * y) + 0.004 * (x - 31) ** 2
    gal = (500 + 3 * y) * np.exp(-0.5 * ((x - 30.0) / 4.0) ** 2)
    data = sky + gal + rng.normal(0, 4.0, (NY, NX))
    for line, col in ((7, 5), (30, 55), (31, 56), (64, 10), (90, 50), (111, 3)):
        data[line - 1, col - 1] += 900
    fits.PrimaryHDU(data.astype(np.float32)).writeto(where / "sky.fits", overwrite=True)
    spec = (1000 + 0.5 * np.arange(300) + 30 * np.sin(np.arange(300) / 17.0)
            + rng.normal(0, 3, 300))
    spec[[40, 41, 200]] += 500
    fits.PrimaryHDU(spec.astype(np.float32)).writeto(where / "spec.fits", overwrite=True)
    bpm = np.zeros((NY, NX), np.int16)
    bpm[:, 8] = 1
    bpm[20:23, 50:53] = 2
    fits.PrimaryHDU(bpm).writeto(where / "bpm.fits", overwrite=True)
    fits.PrimaryHDU(np.full((NY, NX), 7.0, np.float32)).writeto(where / "old.fits",
                                                                 overwrite=True)
    fits.PrimaryHDU(np.round(data).astype(np.int16)).writeto(where / "skyi.fits",
                                                              overwrite=True)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
