"""What IRAF's fit1d and background wrote for each case, as fit1d.json: what was
printed, and the output's pixels as their bytes (base64, big-endian), shape and
BITPIX."""

import base64
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from f1_cases import CASES, INTERACTIVE

results = Path(sys.argv[1])
out = {}
for case, (_task, _image, output, _params) in {**CASES, **INTERACTIVE}.items():
    work = results / case
    entry = {"said": (work / "said.txt").read_text(encoding="utf-8").splitlines()}
    path = work / f"{output}.fits"
    if path.exists():
        with fits.open(path) as hdul:
            data = np.asarray(hdul[0].data)
            big = data.astype(data.dtype.newbyteorder(">"))
            entry.update(bitpix=hdul[0].header["BITPIX"], shape=list(data.shape),
                         dtype=big.dtype.str,
                         data=base64.b64encode(big.tobytes()).decode())
    out[case] = entry
Path(__file__).with_name("fit1d.json").write_text(json.dumps(out, indent=0),
                                                   encoding="utf-8")
