"""IRAF 2.18 on one daofind cursor file, from a directory holding field.fits and
crowded.fits (tests/iraf_field: field(), crowded_field()) and NAME.cur:
python3 run_iraf.py NAME < /dev/null, a case named c_ on the crowded one. What the
keys print is in NAME.out and NAME.err, the stars in NAME.coo, NAME.coo.2...
stdvdm is not used here, but uparm$vdm is worth a look after any IRAF cursor run."""
import sys

from pyraf import iraf

name = sys.argv[1]
iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.apphot(_doprint=0)
# In memory only: unlearn would take away whatever this account has saved.
iraf.datapars.fwhmpsf = 2.5
iraf.datapars.sigma = 6.0
iraf.datapars.readnoise = 0.0
iraf.datapars.epadu = 1.0
iraf.findpars.threshold = 4.0
iraf.findpars.nsigma = 1.5
iraf.findpars.ratio = 1.0
iraf.findpars.theta = 0.0
iraf.findpars.sharplo = 0.2
iraf.findpars.sharphi = 1.0
iraf.findpars.roundlo = -1.0
iraf.findpars.roundhi = 1.0
image = "crowded.fits" if name.startswith("c_") else "field.fits"
iraf.daofind(image, output=f"{name}.coo", icommands=f"{name}.cur", interactive="no",
             verify="no", update="no", verbose="yes", starmap="", skymap="", display="no",
             graphics="stdvdm", Stdout=f"{name}.out", Stderr=f"{name}.err")
