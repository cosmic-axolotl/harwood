"""What IRAF left in each case's folder, as crf.json: the pixels it changed, the mask, the card
of the header, the training it saved, the positions it printed. """

import json
import re
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

results = Path(sys.argv[1])
out = {}
for work in sorted(results.iterdir()):
    first = fits.getdata(work / ("s.fits" if work.name.startswith("short") else "f.fits"))
    second = fits.getdata(work / "o.fits")
    mask = fits.getdata(work / "m.fits")
    changed = np.nonzero(first != second)
    out[work.name] = {
        "changed": [[int(x) + 1, int(y) + 1, float(second[y, x])]
                    for y, x in zip(*changed, strict=True)],
        "mask": [[int(x) + 1, int(y) + 1] for y, x in zip(*np.nonzero(mask), strict=True)],
        "crcor": fits.getheader(work / "o.fits")["CRCOR"],
        "saved": (work / "saved.txt").read_text() if (work / "saved.txt").exists() else "",
        "printed": re.findall(r"(\d+ \d+)$", (work / "stdout.txt").read_text(), re.M),
        "error": (work / "error.txt").read_text(),
    }
Path(__file__).with_name("crf.json").write_text(json.dumps(out, indent=0))
