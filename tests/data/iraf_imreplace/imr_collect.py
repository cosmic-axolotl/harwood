"""What IRAF's imreplace left in each case's folder, as imreplace.json: the edited
image's stored values (as written, before BSCALE/BZERO), BITPIX, and what it said."""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from imr_cases import CASES

results = Path(sys.argv[1])
out = {}
for name, params in CASES.items():
    work = results / name
    root = params["image"].split("[")[0]
    with fits.open(work / f"{root}.fits", do_not_scale_image_data=True) as hdul:
        header = hdul[0].header
        data = np.asarray(hdul[0].data)
        out[name] = {"bitpix": header["BITPIX"],
                     "bscale": header.get("BSCALE", 1.0), "bzero": header.get("BZERO", 0.0),
                     "data": [None if np.isnan(v) else float(v)
                              for v in data.ravel().astype(np.float64)],
                     "negzero": [bool(v) for v in np.signbit(data.ravel())
                                 & (data.ravel() == 0)] if data.dtype.kind == "f" else None,
                     "said": (work / "said.txt").read_text(encoding="utf-8").splitlines()}
Path(__file__).with_name("imreplace.json").write_text(json.dumps(out), encoding="utf-8")
