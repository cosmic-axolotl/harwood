"""IRAF's results, from run.py's folder, as imcgeo.json: the sha256 of each output's values."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from imcgeo_cases import CASES, OUTPUTS

from harwood.core import plio


def values(path):
    """A real image as IRAF wrote it, and a mask as whole numbers."""
    path = Path(path)
    if path.suffix == ".pl":
        return plio.read(path)[0].astype(">i4")
    return np.asarray(fits.getdata(path)).astype(">f4")


def digest(path):
    array = values(path)
    return {"shape": list(array.shape), "sha": hashlib.sha256(array.tobytes()).hexdigest()}


CARDS = ("CRPIX1", "CRPIX2", "LTV1", "LTV2", "CD1_1", "NCOMBINE")

if __name__ == "__main__":
    results = Path(sys.argv[1])
    gold = {}
    for name in CASES:
        header = fits.getheader(results / name / "out.fits")
        gold[name] = {"files": {file: digest(results / name / file)
                                for file in ("out.fits", *OUTPUTS.values())},
                      "cards": {key: header[key] for key in CARDS if key in header}}
    (Path(__file__).parent / "imcgeo.json").write_text(json.dumps(gold, indent=0))
    print(len(gold))
