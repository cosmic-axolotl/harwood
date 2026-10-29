"""How IRAF made the files here: phot on tests/iraf_field.field(), written
as field.fits with EXPTIME=30, AIRMASS=1.234, FILTER=V and TIME=02:14:33,
at the cursor positions (photA) and at the first three and two by the
edges with datamax=600 (photB; the coordinates are in test_phot_format).
Run with the system python3 that has PyRAF:

    python3 run_iraf.py DIR A < /dev/null; python3 run_iraf.py DIR B < /dev/null
"""

import os
import sys

os.chdir(sys.argv[1])

from pyraf import iraf  # noqa: E402

iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.apphot(_doprint=0)
for p in ("datapars", "centerpars", "fitskypars", "photpars"):
    iraf.unlearn(p)
iraf.datapars.setParam("epadu", 2.0)
for k, v in dict(exposure="EXPTIME", airmass="AIRMASS", filter="FILTER",
                 obstime="TIME").items():
    iraf.datapars.setParam(k, v)
iraf.photpars.setParam("apertures", "3,5,8")
iraf.centerpars.setParam("calgorithm", "centroid")
case = sys.argv[2]
if case == "B":
    iraf.datapars.setParam("datamax", 600.0)
iraf.phot("field.fits", coords="field.coo" if case == "A" else "edge.coo",
          output=f"phot{case}.mag", interactive=False, verify=False, verbose=False)
