"""What IRAF's ccdproc left under the package's parameters, as pkg.json: the images of
each case (their BITPIX, BZERO and the sha256 of their pixels as written), the files
in the folder and in bak/, what it printed (the questions), and the logfile, the dates
cut off."""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from pkg_cases import CASES

DATE = re.compile(r"[A-Z][a-z]{2} [ \d]\d [ \d]\d:\d\d ")


def image(path: Path) -> dict:
    with fits.open(path, do_not_scale_image_data=True) as hdul:
        header = hdul[0].header
        raw = np.ascontiguousarray(hdul[0].data)
    return {"bitpix": header["BITPIX"], "bzero": header.get("BZERO"),
            "shape": list(raw.shape), "sha": hashlib.sha256(raw.tobytes()).hexdigest()}


if __name__ == "__main__":
    results = Path(sys.argv[1])
    fresh = {p.name for p in (results.parent / "in").iterdir()}
    gold = {}
    for name, (images, _, _, _, _) in CASES.items():
        work = results / name
        files = sorted(p.name for p in work.glob("*.fits"))
        gold[name] = {
            "images": {n: image(work / n) for n in images.split(",")},
            "new": sorted(set(files) - fresh) + sorted(f"bak/{p.name}" for p in
                                                       (work / "bak").glob("*")),
            "said": (work / "out.txt").read_text(encoding="utf-8")
            if (work / "out.txt").exists() else "",
            "logfile": DATE.sub("", (work / "logfile").read_text(encoding="utf-8"))
            if (work / "logfile").exists() else "",
            "error": (work / "error.txt").read_text(encoding="utf-8")
            if (work / "error.txt").exists() else ""}
    (Path(__file__).parent / "pkg.json").write_text(json.dumps(gold, indent=0),
                                                    encoding="utf-8")
    print(len(gold))
