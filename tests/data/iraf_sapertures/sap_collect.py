"""What IRAF's sapertures left in each case's folder, as sap.json: what it said, and
each line's aperture, beam, dtype, w1, dw, z, limits and title."""

import json
import re
import sys
from pathlib import Path

from astropy.io import fits

results = Path(sys.argv[1])
out = {}
for work in sorted(results.iterdir()):
    header = fits.getheader(work / "in.fits")
    wat = "".join(str(header[k]).ljust(68) for k in sorted(header) if k.startswith("WAT2_"))
    specs = {int(m[0]): m[1].split() for m in re.findall(r'spec(\d+) = "([^"]*)"', wat)}
    lines = {}
    for line, v in specs.items():
        lines[str(line)] = {"ap": int(v[0]), "beam": int(v[1]), "dtype": int(v[2]),
                            "w1": float(v[3]), "dw": float(v[4]), "z": float(v[6]),
                            "aplow": float(v[7]), "aphigh": float(v[8]),
                            "title": str(header.get(f"APID{line}", header.get("OBJECT")))}
    if not specs:                   # equispec, after wcsreset: APNUM, in pixels
        for key in sorted(k for k in header if k.startswith("APNUM")):
            line = key[5:]
            v = str(header[key]).split()
            lines[line] = {"ap": int(v[0]), "beam": int(v[1]), "dtype": -1, "w1": 1.0,
                           "dw": 1.0, "z": 0.0, "aplow": float(v[2]), "aphigh": float(v[3]),
                           "title": str(header.get(f"APID{line}", header.get("OBJECT")))}
    out[work.name] = {"said": (work / "said.txt").read_text(encoding="utf-8").splitlines(),
                      "lines": lines}
Path(__file__).with_name("sap.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
