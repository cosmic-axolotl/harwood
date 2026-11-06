"""What IRAF's sfit and continuum left in each case's folder, as sft.json: what they
said, the file `logfile` if they wrote one, and for every image the bits of its pixels
and its SFIT and SFITB."""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

results = Path(sys.argv[1])
out = {}
for work in sorted(results.iterdir()):
    case = {"said": (work / "said.txt").read_text(encoding="utf-8").splitlines(),
            "images": {}}
    if (work / "logfile").exists():
        case["logfile"] = (work / "logfile").read_text(encoding="utf-8").splitlines()
    for image in sorted(work.glob("*.fits")):
        header = fits.getheader(image)
        data = np.asarray(fits.getdata(image), np.float32)
        case["images"][image.stem] = {
            "shape": list(data.shape), "bits": data.view(np.uint32).ravel().tolist(),
            "SFIT": header.get("SFIT"), "SFITB": header.get("SFITB")}
    out[work.name] = case
Path(__file__).with_name("sft.json").write_text(json.dumps(out), encoding="utf-8")
