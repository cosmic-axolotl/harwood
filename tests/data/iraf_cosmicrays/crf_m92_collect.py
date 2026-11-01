"""What IRAF wrote for two real M92 frames, as crf_m92.json. The frames were cleaned by this,
in a scratch folder, with the same parameters as M92 below:

    iraf.cosmicrays(frame, "o<tag>.fits", crmasks="m<tag>.pl", threshold=..., fluxratio=...,
                    npasses=5, window=..., interactive="no", train="no")
    iraf.imcopy("m<tag>.pl", "m<tag>.fits")
"""

import hashlib
import json
import sys
from pathlib import Path

from astropy.io import fits

M92 = {"a": ("m92010.fits", 25.0, 2.0, 5), "b": ("m92014.fits", 25.0, 2.0, 5),
       "c": ("m92010.fits", 10.0, 10.0, 7), "d": ("m92014.fits", 5.0, 30.0, 5)}

if __name__ == "__main__":
    folder = Path(sys.argv[1])
    out = {}
    for tag in M92:
        image = fits.getdata(folder / f"o{tag}.fits").astype(">f4")
        mask = fits.getdata(folder / f"m{tag}.fits") > 0
        out[tag] = {"image": hashlib.sha256(image.tobytes()).hexdigest(),
                    "mask": hashlib.sha256(mask.tobytes()).hexdigest(),
                    "crcor": fits.getheader(folder / f"o{tag}.fits")["CRCOR"]}
    Path(__file__).with_name("crf_m92.json").write_text(json.dumps(out, indent=0))
