"""How IRAF made the files here: qphot on tests/iraf_field.field(), written as
field.fits with EXPTIME=30, AIRMASS=1.234, FILTER=V and TIME=02:14:33, at the
coordinates of CURSOR in tests/iraf_field.py (field.coo, x y a line):
python3 run_iraf.py DIR < /dev/null."""

import os
import sys

os.chdir(sys.argv[1])

from pyraf import iraf  # noqa: E402

iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.apphot(_doprint=0)
HEADER = dict(exposure="EXPTIME", airmass="AIRMASS", filter="FILTER", obstime="TIME")
CASES = {"qA": (5.0, 10.0, 10.0, "3,5,8", dict(epadu=2.0, **HEADER)),
         "qB": (9.0, 15.0, 5.0, "4", dict(zmag=20.0)),
         "qC": (3.0, 8.0, 12.0, "2,4,6,7.5", dict(epadu=4.0, exposure="EXPTIME"))}
for name, (cbox, annulus, dannulus, apertures, kw) in CASES.items():
    iraf.qphot("field.fits", cbox, annulus, dannulus, apertures, coords="field.coo",
               output=f"{name}.mag", interactive="no", verbose="no", Stdout=f"{name}.out",
               Stderr=f"{name}.err", **kw)
