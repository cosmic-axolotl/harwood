"""IRAF 2.18's imexpr, for tests/test_imexpr_iraf.py.

expr_images.write(folder) in harwood's python, and then in the one with PyRAF:

    python3 run.py <folder>
"""

import io
import json
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
from cases import CASES
from pyraf import iraf

iraf.images(_doprint=0)
iraf.imutil(_doprint=0)
fresh = Path(sys.argv[1]).resolve()
here = Path(__file__).parent.resolve()
work = fresh.parent / "work"
PARAMS = "abcdefghijklmnopqrstuvwxyz"
KEEP = ("OBJECT", "EXPTIME", "CRPIX1", "CRPIX2", "CD1_1", "CD2_2", "LTV1", "LTM1_1", "NCOMB")


def fits_read(path):
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
    bitpix = int(cards["BITPIX"])
    shape = [int(cards[f"NAXIS{n}"]) for n in range(1, int(cards["NAXIS"]) + 1)]
    dtype = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}[bitpix]
    n = int(np.prod(shape))
    data = np.frombuffer(raw, dtype, n, k)
    bscale, bzero = float(cards.get("BSCALE", 1)), float(cards.get("BZERO", 0))
    values = data.astype(np.float64) * bscale + bzero if (bscale != 1 or bzero != 0) else data
    return {"bitpix": bitpix, "shape": shape, "bzero": bzero,
            "bits": [struct.pack(">d", float(v)).hex() for v in np.ravel(values)],
            "cards": {key: cards[key].strip("' ") for key in KEEP if key in cards}}


out = {}
for name, params in CASES.items():
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(fresh, work)
    shutil.copy(here / "db.dat", work)
    shutil.copy(here / "expr.txt", work)
    iraf.chdir(str(work))
    params = dict(params)
    expr = params.pop("expr")
    target = params.pop("output", "out.fits")
    kw = {k: "" for k in PARAMS}
    kw.update(dims="auto", intype="auto", outtype="auto", refim="auto", bwidth=0,
              btype="nearest", bpixval=0.0, rangecheck="yes", verbose="no", exprdb="none")
    kw.update(params)
    text = io.StringIO()
    error = ""
    try:
        iraf.imexpr(expr, target, Stdout=text, Stderr=text, mode="h", **kw)
    except Exception as e:  # noqa: BLE001
        error = re.sub(r"\s+", " ", str(e))[:240]
    file = work / target.split("[")[0]
    done = {"params": {"expr": expr, "output": target, **params},
            "text": text.getvalue()[-300:], "error": error}
    if file.exists() and not error:
        done.update(fits_read(file))
    out[name] = done
    (here / "imexpr.json").write_text(json.dumps(out), encoding="utf-8")
print(len(out), sum(1 for v in out.values() if v["error"]))
