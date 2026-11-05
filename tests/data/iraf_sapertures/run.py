"""IRAF 2.18's sapertures, for tests/test_sapertures_iraf.py. It works in place, so
each case runs on a fresh copy of ../iraf_slist's spectra.

    python ../iraf_slist/sl_inputs.py <folder>                      (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF)
    python sap_collect.py <results>                                  (harwood's)
"""

import io
import shutil
import sys
from pathlib import Path

from pyraf import iraf
from sap_cases import CASES, TABLE

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    base = dict(apertures="", apidtable="", wcsreset="no", verbose="yes", beam="INDEF",
                dtype="INDEF", w1="INDEF", dw="INDEF", z="INDEF", aplow="INDEF",
                aphigh="INDEF", title="INDEF")
    for name, params in CASES.items():
        work = results / name
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        iraf.chdir(str(work))
        (work / "apid.dat").write_text(TABLE, encoding="utf-8")
        params = {**base, **{k: ("yes" if v else "no") if isinstance(v, bool) else v
                             for k, v in params.items()}}
        text = io.StringIO()
        iraf.sapertures("in", Stdout=text, Stderr=text, **params)
        (work / "said.txt").write_text(text.getvalue(), encoding="utf-8")
        print(name, flush=True)
