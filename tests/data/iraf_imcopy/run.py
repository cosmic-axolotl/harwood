"""IRAF 2.18's imcopy of each pixel type, as imcopy.json:

    python icp_inputs.py <folder>                                   (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder>            (PyRAF)
    python icp_collect.py <folder>                                  (harwood's)
"""

import os
import sys

CASES = {"b8": "b8.fits", "i16": "i16.fits", "u16": "u16.fits", "i32": "i32.fits",
         "f32": "f32.fits", "f64": "f64.fits", "scaled": "scaled.fits",
         "section": "u16.fits[2:3,*]"}

if __name__ == "__main__":
    from pyraf import iraf

    iraf.images(_doprint=0)
    iraf.imutil(_doprint=0)
    os.chdir(sys.argv[1])
    for name, image in CASES.items():
        iraf.imcopy(image, f"c_{name}.fits", verbose="no")
