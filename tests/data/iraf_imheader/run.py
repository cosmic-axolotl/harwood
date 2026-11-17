"""IRAF 2.18's imheader on imh_inputs.py's images, long and short, as imheader.json:

    python imh_inputs.py <folder>                                     (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder>              (PyRAF)
"""

import io
import json
import os
import sys
from pathlib import Path

from pyraf import iraf

NAMES = ["b8.fits", "i16.fits", "u16.fits", "i32.fits", "f32.fits", "f64.fits", "one.fits",
         "notitle.fits", "mef.fits[1]", "real.fits"]

if __name__ == "__main__":
    iraf.images(_doprint=0)
    iraf.imutil(_doprint=0)
    os.chdir(sys.argv[1])
    out = {}
    for name in NAMES:
        for long in (False, True):
            text = io.StringIO()
            iraf.imheader(name, longheader="yes" if long else "no", Stdout=text)
            out[f"{name} {'long' if long else 'short'}"] = text.getvalue().splitlines()
    Path(__file__).with_name("imheader.json").write_text(json.dumps(out, indent=1) + "\n",
                                                          encoding="utf-8")
