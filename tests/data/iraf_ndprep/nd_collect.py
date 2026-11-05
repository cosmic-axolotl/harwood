"""What IRAF's ndprep made, as nd.json: each image's shape and pixels (their bits) and
the world coordinates along its dispersion."""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from nd_cases import CASES

work = Path(sys.argv[1]).resolve()
out = {}
for name in CASES:
    data = np.asarray(fits.getdata(work / f"{name}.fits"), np.float32)
    rows = [r.split() for r in (work / f"{name}.world").read_text(
        encoding="utf-8").splitlines() if r.strip()]
    out[name] = {"shape": list(data.shape), "bits": data.ravel().view(np.uint32).tolist(),
                 "world": [float(r[0]) for r in rows]}
Path(__file__).with_name("nd.json").write_text(json.dumps(out), encoding="utf-8")
