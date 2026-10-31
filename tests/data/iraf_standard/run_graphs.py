"""IRAF 2.18 drawing standard's graphs: python3 run_graphs.py NAME IMAGES CURSOR, from
a directory with this folder's cursor file and mycal, kpnoextinct.dat, and IMAGES as
IRAF's own 1-D spectra (IRAF cannot read harwood's). Metacode into NAME.vdm, for
../iraf_photcal/decode_gki.py --everything."""
import shutil
import sys
from pathlib import Path

from pyraf import iraf

name, images, cursor = sys.argv[1:4]
vdm = Path.home() / ".iraf/uparm/vdm"
vdm.unlink(missing_ok=True)
iraf.noao(_doprint=0)
iraf.onedspec(_doprint=0)
iraf.standard(images, f"{name}.std", samestar="yes", beam_switch="no", apertures="",
              bandwidth="INDEF", bandsep="INDEF", fnuzero=3.68e-20,
              extinction="kpnoextinct.dat", caldir="mycal/", observatory="kpno",
              interact="yes", graphics="stdvdm", cursor=cursor, star_name="fakestar",
              airmass="", exptime="", answer="YES", Stdin="/dev/null",
              Stdout=f"{name}.out", Stderr=f"{name}.err")
shutil.move(vdm, f"{name}.vdm")
