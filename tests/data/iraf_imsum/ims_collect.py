"""What IRAF's imsum wrote for each case, as imsum.json: the output's stored values
(before BSCALE/BZERO), shape, BITPIX, BZERO, OBJECT, the hparams' cards as IRAF
wrote them, and what it printed. A case IRAF refused has no image."""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from ims_cases import CASES

results = Path(sys.argv[1])
out = {}
for number, case in enumerate(CASES, start=1):
    said = (results / f"o{number:03d}.txt").read_text(encoding="utf-8").splitlines()
    entry = {"said": said}
    path = results / f"o{number:03d}.fits"
    if path.exists():
        with fits.open(path, do_not_scale_image_data=True) as hdul:
            header, data = hdul[0].header, np.asarray(hdul[0].data)
        entry.update(
            bitpix=header["BITPIX"], bzero=header.get("BZERO", 0.0),
            shape=list(data.shape), object=header.get("OBJECT"),
            cards={key: str(header.cards[key]) for key in ("EXPTIME", "AIRMASS")
                   if key in header},
            data=[None if np.isnan(v) else float(v)
                  for v in data.ravel().astype(np.float64)])
    out[case] = entry
Path(__file__).with_name("imsum.json").write_text(json.dumps(out), encoding="utf-8")
