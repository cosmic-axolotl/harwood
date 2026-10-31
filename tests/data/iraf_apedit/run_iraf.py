"""IRAF 2.18 on one interactive apedit case: python3 run_iraf.py NAME, from a directory
with this folder's NAME.cur and NAME.ans and frame.fits, tests/iraf_field.spectra_frame()
with DISPAXIS=2. apedit reads CL's gcur and draws on stdgraph, so the metacode is PyRAF's
StdoutG, in NAME.vdm, for ../iraf_photcal/decode_gki.py --everything."""
import shutil
import sys

from pyraf import iraf

name = sys.argv[1]
shutil.rmtree("database", ignore_errors=True)
iraf.noao(_doprint=0)
iraf.twodspec(_doprint=0)
iraf.apextract(_doprint=0)
iraf.apextract.dispaxis = 2
iraf.apextract.database = "database"
for key, value in dict(lower=-5.0, upper=5.0, b_function="chebyshev", b_order=1,
                       b_sample="-10:-6,6:10", b_naverage=-3, b_niterate=0, b_low_reject=3.0,
                       b_high_reject=3.0, b_grow=0.0).items():
    iraf.apdefault.setParam(key, value)
iraf.apfind.nfind = 2
iraf.apfind.minsep = 5.0
iraf.apfind.maxsep = 1000.0
iraf.apfind.order = "increasing"
iraf.cl.gcur = f"{name}.cur"
iraf.apedit("frame.fits", apertures="", references="", interactive="yes", find="no",
            recenter="no", resize="no", edit="yes", line="INDEF", nsum=10, width=5.0,
            radius=10.0, threshold=0.0, Stdin=f"{name}.ans", Stdout=f"{name}.out",
            Stderr=f"{name}.err", StdoutG=f"{name}.vdm")
