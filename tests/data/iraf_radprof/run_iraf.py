"""How IRAF made the files here: radprof on tests/iraf_field.field(), written as
field.fits, at the coordinates of CURSOR (field.coo) and some at the edges
(edge.coo): python3 run_iraf.py DIR < /dev/null. The psets are set in memory only."""

import os
import sys

os.chdir(sys.argv[1])

from pyraf import iraf  # noqa: E402

iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.apphot(_doprint=0)
iraf.datapars.fwhmpsf = 2.5
iraf.datapars.sigma = "INDEF"
iraf.datapars.datamin = "INDEF"
iraf.datapars.datamax = "INDEF"
iraf.datapars.epadu = 2.0
iraf.datapars.readnoise = 0.0
iraf.datapars.itime = 1.0
iraf.centerpars.calgorithm = "centroid"
iraf.centerpars.cbox = 5.0
iraf.fitskypars.salgorithm = "centroid"
iraf.fitskypars.annulus = 10.0
iraf.fitskypars.dannulus = 10.0
iraf.photpars.apertures = "3,5"
iraf.photpars.zmag = 25.0

CASES = {"rA": ("field.coo", 12.0, 0.5, dict(order=5, kreject=3.0, nreject=1)),
         "rB": ("field.coo", 8.0, 0.25, dict(order=3, kreject=2.0, nreject=4)),
         "rC": ("field.coo", 15.0, 1.0, dict(order=8, kreject=0.0, nreject=0)),
         "rD": ("edge.coo", 12.0, 0.5, dict(order=5, kreject=3.0, nreject=1)),
         "rE": ("field.coo", 6.0, 0.5, dict(order=1, kreject=3.0, nreject=1))}
for name, (coords, radius, step, kw) in CASES.items():
    iraf.radprof("field.fits", radius, step, coords=coords, output=f"{name}.prf",
                 interactive="no", verify="no", update="no", verbose="no", plotfile="",
                 Stdout=f"{name}.out", Stderr=f"{name}.err", **kw)
