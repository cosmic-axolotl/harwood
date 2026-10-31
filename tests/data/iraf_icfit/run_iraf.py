"""IRAF 2.18 on one interactive icfit case, through fit1d, which is icfit on a line:
python3 run_iraf.py NAME, from a directory with this folder's NAME.cur and NAME.ans
and o.fits, the y of ../iraf_icfit_titles.json as its second line of three. What the
keys print is in NAME.out and NAME.err; metacode into NAME.vdm, for
../iraf_photcal/decode_gki.py --everything. A real fit, where most of harwood's are
double: figures past the sixth can differ."""
import shutil
import sys
from pathlib import Path

from pyraf import iraf

name = sys.argv[1]
vdm = Path.home() / ".iraf/uparm/vdm"
vdm.unlink(missing_ok=True)
Path("out.fits").unlink(missing_ok=True)
iraf.noao(_doprint=0)
iraf.twodspec(_doprint=0)
iraf.longslit(_doprint=0)
iraf.fit1d("o.fits[*,2]", "out.fits", type="fit", axis=1, interactive="yes", sample="*",
           naverage=1, function="spline3", order=3, low_reject=0.0, high_reject=0.0,
           niterate=1, grow=0.0, graphics="stdvdm", cursor=f"{name}.cur", Stdin=f"{name}.ans",
           Stdout=f"{name}.out", Stderr=f"{name}.err")
shutil.move(vdm, f"{name}.vdm")
