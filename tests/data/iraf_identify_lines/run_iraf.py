"""IRAF 2.18's identify moving among the lines of an image: python3 run_iraf.py NAME IMAGE
NSUM [RECORDS], from a directory with NAME.cur and NAME.ans, arc_lines.dat, arc3.fits
(multispec_night()'s arc3, written as a multispec image by harwood's apall --format
multispec writer) and arc2d.fits (its long slit); RECORDS, a record file the database
starts from. What the keys print is in NAME.out and NAME.err, the records in NAME.db.
ms1 ran on arc3 with nsum 1, ms10 with nsum 10, slit on arc2d with nsum 10, msdb on arc3
with nsum 1 from ms1.db."""
import shutil
import sys
from pathlib import Path

from pyraf import iraf

name, image, nsum = sys.argv[1], sys.argv[2], sys.argv[3]
shutil.rmtree("database", ignore_errors=True)
if len(sys.argv) > 4:
    Path("database").mkdir()
    shutil.copy(sys.argv[4], f"database/id{image}")
iraf.noao(_doprint=0)
iraf.onedspec(_doprint=0)
iraf.identify(f"{image}.fits", section="middle line", database="database",
              coordlist="arc_lines.dat", units="", nsum=nsum, match=-3.0, maxfeatures=50,
              zwidth=100.0, ftype="emission", fwidth=4.0, cradius=5.0, threshold=0.0,
              minsep=2.0, function="spline3", order=1, sample="*", niterate=0,
              low_reject=3.0, high_reject=3.0, grow=0.0, autowrite="no", graphics="stdvdm",
              cursor=f"{name}.cur", crval="4160.", cdelt="0.29", Stdin=f"{name}.ans",
              Stdout=f"{name}.out", Stderr=f"{name}.err")
record = Path(f"database/id{image}")
if record.exists():
    shutil.copy(record, f"{name}.db")
