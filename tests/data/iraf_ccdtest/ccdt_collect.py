"""What IRAF's ccdtest made in each case, as ct.json: each image's BITPIX, shape, the
sha256 of its pixels as stored, and its header's cards (but the structure's)."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

SKIP = ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND", "IRAF-TLM", "DATE",
        "ORIGIN", "COMMENT", "HISTORY", "")


def image(path: Path) -> dict:
    with fits.open(path, do_not_scale_image_data=True) as hdul:
        raw = np.ascontiguousarray(hdul[0].data)
        header = hdul[0].header
        return {"bitpix": header["BITPIX"], "shape": list(raw.shape),
                "sha": hashlib.sha256(raw.astype(raw.dtype.newbyteorder(">")).tobytes())
                .hexdigest(),
                "cards": {k: header[k] for k in header if k not in SKIP}}


if __name__ == "__main__":
    results = Path(sys.argv[1])
    gold = {work.name: {p.name: image(p) for p in sorted(work.glob("*.fits"))}
            for work in sorted(p for p in results.iterdir() if p.is_dir())}
    Path(__file__).with_name("ct.json").write_text(json.dumps(gold, indent=0,
                                                              default=str),
                                                   encoding="utf-8")
    print(len(gold))
