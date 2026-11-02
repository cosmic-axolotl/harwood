"""IRAF's results, from run.py's folder, as ccdp.json: the sha256 of the bits and the cards."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from ccdp_cases import CASES

FLAGS = ("OVERSCAN", "TRIM", "ZEROCOR", "DARKCOR", "FLATCOR", "ILLUMCOR", "FRINGCOR",
         "READCOR", "FIXPIX")
TEXT = ("DATASEC", "CCDSEC", "BIASSEC", "TRIMSEC")
NUMBER = ("CCDMEAN", "LTV1", "LTV2", "CRPIX1", "CRPIX2")


def digest(array):
    array = np.asarray(array).astype(">f4")
    return list(array.shape), hashlib.sha256(array.tobytes()).hexdigest()


if __name__ == "__main__":
    results = Path(sys.argv[1])
    gold = {}
    for name, (image, _) in CASES.items():
        path = results / name / image
        header = fits.getheader(path)
        shape, sha = digest(fits.getdata(path))
        gold[name] = {"shape": shape, "sha": sha,
                      "flags": [key for key in FLAGS if key in header],
                      "text": {key: header[key] for key in TEXT if key in header},
                      "number": {key: header[key] for key in NUMBER if key in header},
                      "error": (results / name / "err.txt").read_text(encoding="utf-8")
                      if (results / name / "err.txt").exists() else ""}
    (Path(__file__).parent / "ccdp.json").write_text(json.dumps(gold, indent=0),
                                                     encoding="utf-8")
    print(len(gold))
