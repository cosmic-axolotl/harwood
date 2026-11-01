"""IRAF 2.18's artdata tasks and precess, for tests/test_artdata_iraf.py.

art_images.write(folder) in harwood's python, and then in the one with PyRAF:

    python3 run.py <folder>
"""

import io
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from cases import MKNOISE, MKOBJECTS, MKPATTERN, PRECESS, STARLIST
from pyraf import iraf

iraf.noao(_doprint=0)
iraf.astutil(_doprint=0)
iraf.artdata(_doprint=0)
fresh = Path(sys.argv[1]).resolve()
here = Path(__file__).parent.resolve()
work = fresh.parent / "work"


def renew():
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(fresh, work)
    iraf.chdir(str(work))


def run(task, *args, **params):
    text = io.StringIO()
    try:
        task(*args, Stdout=text, Stderr=text, mode="h", **params)
    except Exception as e:  # noqa: BLE001
        return text.getvalue().split("\n")[:-1], re.sub(r"\s+", " ", str(e))[-160:]
    return text.getvalue().split("\n")[:-1], ""


def read(path):
    raw = path.read_bytes()
    cards, order, k, done = {}, [], 0, False
    while not done:
        for i in range(k, k + 2880, 80):
            card = raw[i : i + 80].decode("latin1")
            if card.startswith("END"):
                done = True
                break
            if card[8:10] == "= ":
                cards[card[:8].strip()] = card[10:].split("/")[0].strip()
            order.append(card.rstrip())
        k += 2880
    bitpix = int(cards["BITPIX"])
    shape = [int(cards[f"NAXIS{n}"]) for n in range(1, int(cards["NAXIS"]) + 1)]
    dtype = {16: ">i2", 32: ">i4", -32: ">f4", -64: ">f8"}[bitpix]
    data = np.frombuffer(raw, dtype, int(np.prod(shape)), k).astype(float)
    data = data * float(cards.get("BSCALE", 1)) + float(cards.get("BZERO", 0))
    return {
        "pixels": data.tolist(),
        "bitpix": bitpix,
        "shape": shape,
        "keys": {
            key: cards[key].strip("' ")
            for key in ("OBJECT", "EXPTIME", "RDNOISE", "GAIN", "AIRMASS", "INSTRUME")
            if key in cards
        },
        "comments": [c[8:].rstrip() for c in order if c.startswith("COMMENT")],
    }


out = {"precess": {}, "mkpattern": {}}
for name, params in PRECESS.items():
    renew()
    params = {"stdepoch": 0.0, **params}
    lines, error = run(
        iraf.precess,
        params.pop("input"),
        params.pop("startyear"),
        params.pop("endyear"),
        **params,
    )
    out["precess"][name] = {"stdout": lines, "error": error}
base = dict(
    output="",
    pattern="constant",
    option="replace",
    v1=0.0,
    v2=1.0,
    size=1,
    title="",
    pixtype="real",
    ndim=2,
    ncols=512,
    nlines=512,
    n3=1,
    n4=1,
    n5=1,
    n6=1,
    n7=1,
    header="",
)
for name, params in MKPATTERN.items():
    renew()
    merged = {**base, **params}
    source = merged.pop("input", "new.fits")
    lines, error = run(iraf.mkpattern, source, **merged)
    target = merged["output"] or source.split("[")[0]
    done = {"params": params, "stdout": lines, "error": error}
    if not error and (work / target).exists():
        done.update(read(work / target))
    out["mkpattern"][name] = done
noise = dict(
    output="",
    title="",
    ncols=512,
    nlines=512,
    header="",
    background=0.0,
    gain=1.0,
    rdnoise=0.0,
    poisson="no",
    seed=1,
    cosrays="",
    ncosrays=0,
    energy=30000.0,
    radius=0.5,
    ar=1.0,
    pa=0.0,
    comments="yes",
)
objects = dict(
    output="",
    title="",
    ncols=512,
    nlines=512,
    header="",
    background=1000.0,
    objects="",
    xoffset=0.0,
    yoffset=0.0,
    star="moffat",
    radius=1.0,
    beta=2.5,
    ar=1.0,
    pa=0.0,
    distance=1.0,
    exptime=1.0,
    magzero=7.0,
    gain=1.0,
    rdnoise=0.0,
    poisson="no",
    seed=1,
    comments="yes",
)
for key, table, task, base2 in (
    ("mknoise", MKNOISE, "mknoise", noise),
    ("mkobjects", MKOBJECTS, "mkobjects", objects),
):
    out[key] = {}
    for name, params in table.items():
        renew()
        iraf.flprcache()
        merged = {**base2, **params}
        source = merged.pop("input", "new.fits")
        lines, error = run(getattr(iraf, task), source, **merged)
        target = merged["output"] or source
        done = {"params": params, "stdout": lines, "error": error}
        if not error and (work / target).exists():
            done.update(read(work / target))
        if (work / "new.txt").exists():
            done["written"] = (work / "new.txt").read_text()
        out[key][name] = done
base3 = dict(
    interactive="no",
    spatial="uniform",
    xmin=1.0,
    xmax=512.0,
    ymin=1.0,
    ymax=512.0,
    xcenter="INDEF",
    ycenter="INDEF",
    core_radius=30.0,
    base=0.0,
    sseed=1,
    luminosity="powlaw",
    minmag=-7.0,
    maxmag=0.0,
    mzero=-4.0,
    power=0.6,
    alpha=0.74,
    beta=0.04,
    delta=0.294,
    mstar=1.28,
    lseed=1,
    sfile="",
    lfile="",
    nssample=100,
    sorder=10,
    nlsample=100,
    lorder=10,
)
out["starlist"] = {}
for name, params in STARLIST.items():
    renew()
    iraf.flprcache()
    merged = {**base3, **params}
    lines, error = run(iraf.starlist, "list.txt", merged.pop("nstars"), **merged)
    done = {"params": params, "stdout": lines, "error": error}
    if (work / "list.txt").exists():
        done["text"] = (work / "list.txt").read_text()
    out["starlist"][name] = done
(here / "artdata.json").write_text(json.dumps(out))
print({k: [c for c, v in out[k].items() if v["error"]] for k in out})
