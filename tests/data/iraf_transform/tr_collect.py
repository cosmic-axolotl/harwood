"""What IRAF's fitcoords and transform left in each case, as transform.json: what
was printed, the logfile, the fc database files, and every output image -- its
pixels as their bytes (base64, big-endian as in the file), shape, BITPIX and the
cards transform writes; a mask's pixels from its pl extension."""

import base64
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from tr_cases import FITCOORDS, TRANSFORM, WCS

CARDS = ("DISPAXIS", "DCLOG1", "DC-FLAG", "WCSDIM", "CTYPE1", "CTYPE2", "CRVAL1",
         "CRVAL2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2", "CD1_1", "CD2_2", "LTM1_1",
         "LTM2_2", "LTV1", "LTV2", "WAT0_001", "WAT1_001", "WAT2_001", "BPM", "OBJECT")


def image(path: Path) -> dict:
    with fits.open(path) as hdul:
        hdu = hdul[1] if len(hdul) > 1 else hdul[0]
        data = np.asarray(hdu.data)
        header = hdu.header
        big = data.astype(data.dtype.newbyteorder(">"))
        return {"bitpix": header["BITPIX"], "shape": list(data.shape),
                "dtype": big.dtype.str, "data": base64.b64encode(big.tobytes()).decode(),
                "cards": {k: header[k] for k in CARDS if k in header}}


results = Path(sys.argv[1])
out = {}
for group, cases in (("fitcoords", FITCOORDS), ("transform", {**TRANSFORM, **WCS})):
    for case in cases:
        work = results / group / case
        entry = {"said": (work / "said.txt").read_text(encoding="utf-8").splitlines()}
        if (work / "logfile").exists():
            entry["logfile"] = (work / "logfile").read_text(encoding="utf-8").splitlines()
        entry["fc"] = {p.name: p.read_text(encoding="utf-8").splitlines()
                       for p in sorted((work / "database").glob("fc*"))}
        entry["images"] = {p.name: image(p) for p in sorted(work.glob("o*.fits"))
                           + sorted(work.glob("mo*.fits"))}
        if (work / "w.fits").exists():
            entry["w"] = image(work / "w.fits")
        out[f"{group}/{case}"] = entry
Path(__file__).with_name("transform.json").write_text(json.dumps(out, indent=0),
                                                       encoding="utf-8")
