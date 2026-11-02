"""IRAF 2.18's t at the edges of an image, a session each, for
tests/test_imexamine_keys_iraf.py, which has what it found. The image is 40 by 30, 1000 y + x.

    HOME=<a scratch folder> python3 run_edges.py <folder with f.fits>
"""

import glob
import io
import os
import sys

from pyraf import iraf

iraf.images(_doprint=0)
iraf.tv(_doprint=0)
os.chdir(sys.argv[1])
iraf.chdir(sys.argv[1])
AT = ("2 15", "39 15", "20 2", "20 3", "20 29", "20 28", "2 2", "39 29", "60 15", "20 40")
for at in AT:
    for name in glob.glob("t.*"):
        os.remove(name)
    with open("k.cur", "w", encoding="utf-8") as file:
        file.write(f"{at} 1 t\n20 15 1 x\n")
    text = io.StringIO()
    error = ""
    try:
        iraf.imexamine("f.fits", imagecur="k.cur", logfile="", keeplog="no", frame=1,
                       use_display="no", graphics="stdvdm", mode="h", output="t", ncoutput=7,
                       nloutput=5, Stdout=text, Stderr=text)
    except Exception as e:  # noqa: BLE001
        error = str(e)[-40:]
    print(at, repr(text.getvalue()), error, sorted(glob.glob("t.*")), flush=True)
