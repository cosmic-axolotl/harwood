"""What IRAF's response and illumination wrote for each case, as longslit.json: the
output's pixels, shape, BITPIX, CCDMEAN and MKILLUM, and what was printed."""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from ls_cases import ILLUMINATION, RESPONSE

results = Path(sys.argv[1])
out = {}
for case, params in {**RESPONSE, **ILLUMINATION}.items():
    work = results / case
    entry = {"said": (work / "said.txt").read_text(encoding="utf-8").splitlines()}
    path = work / (params.get("output", "o") + ".fits")
    if path.exists():
        with fits.open(path) as hdul:
            header, data = hdul[0].header, np.asarray(hdul[0].data, dtype=np.float64)
        entry.update(bitpix=header["BITPIX"], shape=list(data.shape),
                     ccdmean=header.get("CCDMEAN"), mkillum=header.get("MKILLUM"),
                     data=[None if not np.isfinite(v) else float(v) for v in data.ravel()])
    out[case] = entry
Path(__file__).with_name("longslit.json").write_text(json.dumps(out), encoding="utf-8")
