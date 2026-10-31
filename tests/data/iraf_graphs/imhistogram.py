"""IRAF 2.18 drawing imhistogram's plots: python3 imhistogram.py, from a directory with
field.fits, tests/iraf_field.field() with OBJECT = 'a field'. The metacode of each goes
to NAME.vdm, for ../iraf_photcal/decode_gki.py --everything, and from there imh_NAME.json."""
import shutil
from pathlib import Path

from pyraf import iraf

vdm = Path.home() / ".iraf/uparm/vdm"
iraf.images(_doprint=0)
iraf.imutil(_doprint=0)
CASES = {"line": {"plot_type": "line"}, "box": {"plot_type": "box"},
         "log": {"plot_type": "line", "logy": "yes"},
         "nolog": {"plot_type": "box", "logy": "no"},
         "cum": {"hist_type": "cumulative", "plot_type": "line"},
         "dif": {"hist_type": "difference", "plot_type": "line"},
         "sec": {"hist_type": "second_difference", "plot_type": "box"},
         "bins": {"z1": 100.0, "z2": 140.0, "nbins": 40, "plot_type": "box"},
         "binw": {"z1": 100.0, "z2": 140.0, "binwidth": 2.0, "plot_type": "line"},
         "closed": {"z1": 100.0, "z2": 140.0, "nbins": 8, "top_closed": "yes",
                    "plot_type": "line"}}
for name, settings in CASES.items():
    vdm.unlink(missing_ok=True)
    iraf.imhistogram("field.fits", device="stdvdm", listout="no",
                     **{"logy": "yes", **settings})
    iraf.gflush()
    shutil.move(vdm, f"{name}.vdm")
