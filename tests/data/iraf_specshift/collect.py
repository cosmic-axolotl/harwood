"""What IRAF's specshift left in each case's folder, as specshift.json: what it
printed and each aperture's w1, dw and z."""

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
    specs = {m[0]: [float(v) for v in m[1].split()]
             for m in re.findall(r'spec(\d+) = "([^"]*)"', wat)}
    out[work.name] = {"said": (work / "said.txt").read_text(encoding="utf-8").splitlines(),
                      "apertures": {ap: {"w1": v[3], "dw": v[4], "z": v[6]}
                                    for ap, v in specs.items()}}
Path(__file__).with_name("specshift.json").write_text(json.dumps(out, indent=1),
                                                      encoding="utf-8")
