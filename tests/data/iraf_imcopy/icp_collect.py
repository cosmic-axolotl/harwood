"""What IRAF's imcopy wrote, as imcopy.json: BITPIX, BZERO and the pixels as stored."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from run import CASES


def stored(path) -> dict:
    with fits.open(path, do_not_scale_image_data=True, memmap=False) as hdul:
        raw = np.ascontiguousarray(hdul[0].data)
        header = hdul[0].header
        return {"bitpix": header["BITPIX"], "bzero": header.get("BZERO"),
                "shape": list(raw.shape),
                "sha": hashlib.sha256(raw.astype(raw.dtype.newbyteorder(">")).tobytes())
                .hexdigest()}


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    gold = {name: stored(folder / f"c_{name}.fits") for name in CASES}
    Path(__file__).with_name("imcopy.json").write_text(json.dumps(gold, indent=1) + "\n",
                                                       encoding="utf-8")
