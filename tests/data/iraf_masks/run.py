"""IRAF 2.18's text2mask, badpiximage and ccdmask, for tests/test_masks_iraf.py.

mask_images.write(folder) in harwood's python, and then in the one with PyRAF:

    python3 run.py <folder>

Masks stay here as .pl, images as text of a line per pixel run.
"""

import io
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from cases import BADPIX, CCDMASK, MSKEXPR, PLIO, TEXT2MASK
from pyraf import iraf

iraf.images(_doprint=0)
iraf.proto(_doprint=0)
iraf.noao(_doprint=0)
iraf.imred(_doprint=0)
iraf.ccdred(_doprint=0)
fresh = Path(sys.argv[1]).resolve()
here = Path(__file__).parent.resolve()
work = fresh.parent / "work"


def run(task, *args, **params):
    text = io.StringIO()
    try:
        task(*args, Stdout=text, Stderr=text, mode="h", **params)
    except Exception as e:  # noqa: BLE001
        return re.sub(r"\s+", " ", str(e))[-160:]
    return ""


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
    dtype = {16: ">i2", 32: ">i4", -32: ">f4"}[int(cards["BITPIX"])]
    pixels = np.frombuffer(raw, dtype, n, k).astype(int).tolist()
    return {"pixels": pixels, "bitpix": int(cards["BITPIX"]),
            "object": cards.get("OBJECT", ""), "exptime": cards.get("EXPTIME", ""),
            "datamax": cards.get("DATAMAX", "")}


shutil.rmtree(work, ignore_errors=True)
shutil.copytree(fresh, work)
iraf.chdir(str(work))
out = {"text2mask": {}, "badpix": {}, "ccdmask": {}}
for name, params in TEXT2MASK.items():
    params = {"linterp": 1, "cinterp": 2, "square": 3, "pixel": 4, **params}
    error = run(iraf.text2mask, params.pop("text"), f"t_{name}.pl", params.pop("ncols"),
                params.pop("nlines"), **params)
    out["text2mask"][name] = error
    if not error:
        shutil.copy(work / f"t_{name}.pl", here / f"t_{name}.pl")
for name, params in BADPIX.items():
    params = {"goodvalue": 1, "badvalue": 0, **params}
    target = f"b_{name}.fits"
    error = run(iraf.badpiximage, params.pop("fixfile"), params.pop("template"), target,
                **params)
    out["badpix"][name] = {"error": error, **(read(work / target) if not error else {})}
for name, params in CCDMASK.items():
    base = dict(ncmed=7, nlmed=7, ncsig=15, nlsig=15, lsigma=6.0, hsigma=6.0, ngood=5,
                linterp=2, cinterp=3, eqinterp=2)
    params = {**base, **params}
    error = run(iraf.ccdmask, params.pop("image"), f"c_{name}.pl", **params)
    out["ccdmask"][name] = error
    if not error:
        shutil.copy(work / f"c_{name}.pl", here / f"c_{name}.pl")
for name, (task, params) in PLIO.items():
    params = dict(params)
    if task == "imexpr":
        kw = dict(dims="auto", intype="auto", outtype="auto", refim="auto", bwidth=0,
                  btype="nearest", bpixval=0.0, rangecheck="yes", verbose="no", exprdb="none")
        kw.update({c: "" for c in "abcdefghijklmnopqrstuvwxyz"})
        kw.update(params)
        expr = kw.pop("expr")
        target = kw.pop("output")
        error = run(iraf.imexpr, expr, target, **kw)
    else:
        target = params["output"]
        error = run(iraf.imcopy, params["input"], target, verbose="no")
    out["ccdmask"]["P" + name] = error
    if not error:
        shutil.copy(work / target, here / target)
out["mskexpr"] = {}
for name, params in MSKEXPR.items():
    params = {"refimages": "", "refmasks": "", "depth": 0, "exprdb": "none", "verbose": "yes",
              **params}
    expr = params.pop("expr")
    target = f"x_{name}.pl"
    text = io.StringIO()
    error = ""
    try:
        iraf.mskexpr(expr, target, params.pop("refimages"), Stdout=text, Stderr=text, mode="h",
                     **params)
    except Exception as e:  # noqa: BLE001
        error = re.sub(r"\s+", " ", str(e))[-160:]
    out["mskexpr"][name] = {"error": error, "stdout": text.getvalue().split("\n")[:-1],
                            "file": (work / target).exists()}
    if not error and (work / target).exists():
        shutil.copy(work / target, here / target)
(here / "masks.json").write_text(json.dumps(out))
print({k: [c for c, v in out[k].items() if (v if isinstance(v, str) else v["error"])]
       for k in out})
