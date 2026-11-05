"""What IRAF left in each dopcor case's folder, as dop.json: for each run, what it
printed, its DOPCORnn cards, and each aperture's w1, dw, z and fluxes."""

import json
import re
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

results = Path(sys.argv[1])
out = {}
for work in sorted(results.iterdir()):
    runs = []
    for said in sorted(work.glob("o*.txt"), key=lambda p: int(p.stem[1:])):
        image = said.with_suffix(".fits")
        run = {"said": said.read_text(encoding="utf-8").splitlines(), "written": False}
        if image.exists():
            header = fits.getheader(image)
            wat = "".join(str(header[k]).ljust(68) if len(str(header[k])) < 68 else
                          str(header[k]) for k in sorted(header) if k.startswith("WAT2_"))
            specs = {int(m[0]): [float(v) for v in m[1].split()]
                     for m in re.findall(r'spec(\d+) = "([^"]*)"', wat)}
            data = fits.getdata(image)
            run.update(written=True,
                       cards={k: str(header[k]) for k in header if k.startswith("DOPCOR")},
                       apertures={str(ap): {"w1": v[3], "dw": v[4], "z": v[6],
                                            "flux": np.asarray(data[ap - 1], float).tolist()}
                                  for ap, v in specs.items()})
        runs.append(run)
    out[work.name] = runs
Path(__file__).with_name("dop.json").write_text(json.dumps(out, indent=0), encoding="utf-8")
