"""What IRAF's odcombine left in each case's folder, as odc.json: its log, and for each
output its header cards, its pixels -- the bits of reals, the values of doubles and
integers -- and its world coordinates; the masks, from their pixel lists, as integers."""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

CARDS = ("OBJECT", "NCOMBINE", "EXPTIME", "ICBPM", "BPM", "APNUM1", "DC-FLAG", "CRVAL1",
         "CDELT1", "LTV1", "BITPIX", "MSTITLE")


def pixels(data):
    if data.dtype.kind == "f" and data.dtype.itemsize == 4:
        return {"kind": "real", "bits": np.asarray(data, np.float32).view(np.uint32)
                .ravel().tolist()}
    if data.dtype.kind == "f":
        return {"kind": "double", "values": np.asarray(data, np.float64).ravel().tolist()}
    return {"kind": "int", "values": np.asarray(data, np.int64).ravel().tolist()}


results = Path(sys.argv[1])
out = {}
for work in sorted(results.iterdir()):
    case = {"said": (work / "said.txt").read_text(encoding="utf-8").splitlines(),
            "outputs": {}, "masks": {}}
    for world in sorted(work.glob("*.world")):
        image = world.with_suffix(".fits")
        header = fits.getheader(image)
        data = fits.getdata(image)
        rows = [r.split() for r in world.read_text(encoding="utf-8").splitlines()
                if r.strip()]
        case["outputs"][image.name] = {
            "cards": {k: header[k] for k in CARDS if k in header},
            "imcmb": [header[k] for k in sorted(header) if k.startswith("IMCMB")],
            "shape": list(data.shape), **pixels(data),
            "world": [float(r[0]) for r in rows]}
    for image in sorted(work.glob("*_pl.fits")):
        data = fits.getdata(image)
        case["masks"][image.name.removesuffix("_pl.fits")] = {
            "shape": list(data.shape), "values": np.asarray(data, np.int64).ravel().tolist()}
    out[work.name] = case
Path(__file__).with_name("odc.json").write_text(json.dumps(out), encoding="utf-8")
