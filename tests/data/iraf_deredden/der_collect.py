"""What IRAF's deredden left in each case's folder, as der.json: for each run, what it
printed, its DEREDDEN card and each aperture's fluxes."""

import json
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
            data = np.atleast_2d(fits.getdata(image))
            run.update(written=True, card=str(header.get("DEREDDEN", "")),
                       flux={str(i + 1): np.asarray(row, float).tolist()
                             for i, row in enumerate(data)})
        runs.append(run)
    out[work.name] = runs
Path(__file__).with_name("der.json").write_text(json.dumps(out), encoding="utf-8")
