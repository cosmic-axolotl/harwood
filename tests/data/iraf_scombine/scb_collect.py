"""What IRAF's scombine left in each case's folder, as scb.json: its log, and for each
output its NCOMBINE and EXPTIME and its lines -- aperture, beam, title, length, the
pixels' bits and the world coordinates."""

import json
import re
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

results = Path(sys.argv[1])
out = {}
for work in sorted(results.iterdir()):
    case = {"said": (work / "said.txt").read_text(encoding="utf-8").splitlines(),
            "outputs": {}}
    for image in sorted(work.glob("*.world")):
        image = image.with_suffix(".fits")
        header = fits.getheader(image)
        data = np.asarray(fits.getdata(image), np.float32)
        if data.ndim == 1:
            data = data[None, :]
        nlines, npix = data.shape
        wat = "".join(str(header[k]).ljust(68) for k in sorted(header)
                      if k.startswith("WAT2_"))
        specs = {int(m[0]): m[1].split() for m in re.findall(r'spec(\d+) = "([^"]*)"', wat)}
        rows = [r.split() for r in (work / f"{image.stem}.world").read_text(
            encoding="utf-8").splitlines() if r.strip()]
        world = np.array([float(r[0]) for r in rows]).reshape(nlines, npix)
        lines = []
        for line in range(1, nlines + 1):
            apnum = str(header[f"APNUM{line}"]).split() if f"APNUM{line}" in header \
                else specs[line][:2]
            nw = int(specs[line][5]) if line in specs else npix
            lines.append({"ap": int(apnum[0]), "beam": int(apnum[1]),
                          "title": str(header.get(f"APID{line}", header.get("OBJECT", ""))),
                          "nw": nw, "dtype": int(specs[line][2]) if line in specs else None,
                          "bits": data[line - 1, :nw].view(np.uint32).tolist(),
                          "world": world[line - 1, :nw].tolist()})
        case["outputs"][image.name] = {"ncombine": header.get("NCOMBINE"),
                                       "exptime": header.get("EXPTIME"),
                                       "object": header.get("OBJECT"),
                                       "lines": lines}
    out[work.name] = case
Path(__file__).with_name("scb.json").write_text(json.dumps(out), encoding="utf-8")
