"""IRAF 2.18 on one interactive identify case: python3 run_iraf.py NAME, from a directory
with this folder's NAME.cur and NAME.ans, ../arc_lines.dat, and arc.fits, written from
tests/iraf_field.arc_spectrum(). What the keys print is in NAME.out and NAME.err, the
record in NAME.db, and metacode in NAME.vdm, for ../iraf_photcal/decode_gki.py --everything."""
import shutil
import sys
from pathlib import Path

from pyraf import iraf

name = sys.argv[1]
vdm = Path.home() / ".iraf/uparm/vdm"
vdm.unlink(missing_ok=True)
shutil.rmtree("database", ignore_errors=True)
iraf.noao(_doprint=0)
iraf.onedspec(_doprint=0)
iraf.identify("arc.fits", section="middle line", database="database",
              coordlist="arc_lines.dat", units="", nsum="10", match=-3.0, maxfeatures=50,
              zwidth=100.0, ftype="emission", fwidth=4.0, cradius=5.0, threshold=0.0,
              minsep=2.0, function="spline3", order=1, sample="*", niterate=0,
              low_reject=3.0, high_reject=3.0, grow=0.0, autowrite="no", graphics="stdvdm",
              cursor=f"{name}.cur", crval="", cdelt="", Stdin=f"{name}.ans",
              Stdout=f"{name}.out", Stderr=f"{name}.err")
shutil.move(vdm, f"{name}.vdm")
record = Path("database/idarc")
if record.exists():
    shutil.copy(record, f"{name}.db")
