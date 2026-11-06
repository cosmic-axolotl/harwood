"""What IRAF's telluric and skytweak left in each case's folder, as twk.json: what they
said, the file `logfile` if they wrote one, and for every image the bits of its pixels
and the header cards it sets."""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

results = Path(sys.argv[1])
out = {}
for work in sorted(results.iterdir()):
    case = {"said": (work / "said.txt").read_text(encoding="utf-8").splitlines(),
            "warned": (work / "warned.txt").read_text(encoding="utf-8").splitlines(),
            "images": {}}
    if (work / "logfile").exists():
        case["logfile"] = (work / "logfile").read_text(encoding="utf-8").splitlines()
    for image in sorted(work.glob("*.fits")):
        if image.read_bytes()[:6] != b"SIMPLE":
            # What IRAF began and never wrote, when it failed.
            case["images"][image.stem] = None
            continue
        header = fits.getheader(image)
        data = np.asarray(fits.getdata(image), np.float32)
        case["images"][image.stem] = {
            "shape": list(data.shape), "bits": data.view(np.uint32).ravel().tolist(),
            "cards": {k: str(header[k]) for k in header
                      if k.startswith(("APNUM", "APID", "BANDID", "WAT", "OBJECT", "CRVAL",
                                       "CDELT", "CD1_1", "LTV", "LTM", "NAXIS", "DC-FLAG"))}}
    out[work.name] = case
Path(__file__).with_name("twk.json").write_text(json.dumps(out), encoding="utf-8")
