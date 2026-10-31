"""IRAF 2.18's imextensions and mimstatistics, for tests/test_proto_lists_iraf.py.

proto_images.write(folder) in harwood's python, and then in the one with PyRAF:

    python3 run.py <folder>
"""

import io
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from cases import IMEXT, MIMSTAT
from pyraf import iraf

iraf.images(_doprint=0)
iraf.proto(_doprint=0)
fresh = Path(sys.argv[1]).resolve()
here = Path(__file__).parent.resolve()
work = fresh.parent / "work"


def renew():
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(fresh, work)
    iraf.chdir(str(work))


def run(task, params):
    text = io.StringIO()
    error = ""
    try:
        task(Stdout=text, Stderr=text, mode="h", **params)
    except Exception as e:  # noqa: BLE001
        error = re.sub(r"\s+", " ", str(e))[:200]
    return text.getvalue().split("\n")[:-1], error


def read_mask(pl):
    """A .pl through imcopy, as numbers."""
    iraf.imcopy(str(pl), "omask_copy.fits", verbose="no", Stdout=io.StringIO())
    raw = (work / "omask_copy.fits").read_bytes()
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
    dtype = {16: ">i2", 32: ">i4", -32: ">f4"}[int(cards["BITPIX"])]
    n = int(cards["NAXIS1"]) * int(cards["NAXIS2"])
    return np.frombuffer(raw, dtype, n, k).astype(int).tolist()


out = {"imext": {}, "mimstat": {}}
for name, params in IMEXT.items():
    renew()
    base = dict(output="file", index="1-", extname="", extver="", lindex="yes", lname="no",
                lver="no", ikparams="")
    lines, error = run(iraf.imextensions, {**base, **params})
    out["imext"][name] = {"params": params, "stdout": lines, "error": error,
                          "nimages": None if error else int(iraf.imextensions.nimages)}
for name, params in MIMSTAT.items():
    renew()
    base = dict(imasks="", omasks="", fields="image,npix,mean,stddev,min,max", lower="INDEF",
                upper="INDEF", nclip=0, lsigma=3.0, usigma=3.0, binwidth=0.1, format="yes",
                cache="no")
    lines, error = run(iraf.mimstatistics, {**base, **params})
    done = {"params": params, "stdout": lines, "error": error}
    pl = work / (params.get("omasks", "") + (
        "" if params.get("omasks", "").endswith(".pl") else ".pl"))
    if params.get("omasks") and pl.exists():
        done["omask"] = read_mask(pl.name)
    out["mimstat"][name] = done
(here / "proto.json").write_text(json.dumps(out))
print(len(out["imext"]), len(out["mimstat"]))
