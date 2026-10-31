"""IRAF 2.18's ccdred.combine, for tests/test_imcombine_iraf.py.

Frames of tests/iraf_field.combine_stack, written by make.py with harwood's python:
    python make.py <folder>; python3 run.py <folder>       (the one with PyRAF)
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from cases import CASES
from pyraf import iraf

iraf.noao(_doprint=0)
iraf.imred(_doprint=0)
iraf.ccdred(_doprint=0)
fresh = Path(sys.argv[1]).resolve()
here = Path(__file__).parent.resolve()
work = fresh.parent / "work"


def read(path):
    raw = path.read_bytes()
    cards, k, done = {}, 0, False
    while not done:
        for i in range(k, k + 2880, 80):
            card = raw[i:i + 80].decode("latin1")
            if card.startswith("END"):
                done = True
                break
            if card[8:10] == "= ":
                cards[card[:8].strip()] = card[10:].split("/")[0].strip()
        k += 2880
    n = int(cards["NAXIS1"]) * int(cards["NAXIS2"])
    return np.frombuffer(raw, ">f4", n, k).astype(float).tolist(), cards


out = {}
for name, params in CASES.items():
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(fresh, work)
    iraf.chdir(str(work))
    iraf.ccdred.instrument = "ccddb$kpno/direct.dat" if params.get("ccdtype") else ""
    iraf.ccdred.ssfile = "subsets"
    iraf.ccdred.verbose = "no"
    params = dict(params)
    images = params.pop("images", "@all.lis")
    base = dict(plfile="", sigma="", ccdtype="", subsets="no", delete="no", clobber="no",
                combine="average", reject="none", project="no", outtype="real", offsets="none",
                masktype="none", maskvalue=0.0, blank=0.0, scale="none", zero="none",
                weight="none", statsec="", lthreshold="INDEF", hthreshold="INDEF", nlow=1,
                nhigh=1, nkeep=1, mclip="yes", lsigma=3.0, hsigma=3.0, rdnoise="0.", gain="1.",
                snoise="0.", sigscale=0.1, pclip=-0.5, grow=0)
    error = ""
    try:
        iraf.ccdred.combine(images, "out", mode="h", **{**base, **params})
    except Exception as e:  # noqa: BLE001
        error = str(e)[-200:]
    done = {"params": params, "images": images, "error": error, "files": {}}
    for file in sorted(work.glob("out*.fits")):
        pixels, cards = read(file)
        done["files"][file.name] = {
            "pixels": pixels,
            "cards": {k: cards[k].strip("' ") for k in ("NCOMBINE", "EXPTIME", "SUBSET")
                      if k in cards}}
    out[name] = done
(here / "combine.json").write_text(json.dumps(out))
print(len(out), sum(1 for v in out.values() if v["error"]))
