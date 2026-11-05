"""What IRAF's disptrans left, as dt.json: what it said, each line's world
coordinates, and the pixels' bits."""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

results = Path(sys.argv[1])
out = {}
for work in sorted(results.iterdir()):
    case = {"said": (work / "said.txt").read_text(encoding="utf-8").splitlines()}
    if not (work / "o.fits").exists():
        out[work.name] = case
        continue
    data = np.asarray(fits.getdata(work / "o.fits"), np.float32)
    for line in (1, 2, 3):
        rows = [r.split() for r in (work / f"world{line}.txt").read_text(
            encoding="utf-8").splitlines() if r.strip()]
        case[str(line)] = {"world": [float(r[0]) for r in rows],
                           "bits": data[line - 1].view(np.uint32).tolist()}
    out[work.name] = case
Path(__file__).with_name("dt.json").write_text(json.dumps(out), encoding="utf-8")
