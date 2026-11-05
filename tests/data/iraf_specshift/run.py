"""IRAF 2.18's specshift, for tests/test_specshift_iraf.py. It works in place, so
each case runs on a fresh copy of ../iraf_dopcor's spectrum.

    python ../iraf_dopcor/dop_inputs.py <folder>                    (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF)
    python collect.py <results>                                     (harwood's)
"""

import io
import shutil
import sys
from pathlib import Path

from pyraf import iraf
from ss_cases import CASES

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    for name, (z, params) in CASES.items():
        work = results / name
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        iraf.chdir(str(work))
        if z is not None:
            iraf.dopcor("in", "in", redshift=str(z), Stdout=io.StringIO())
        text = io.StringIO()
        iraf.specshift("in", Stdout=text, Stderr=text, **params)
        (work / "said.txt").write_text(text.getvalue(), encoding="utf-8")
        print(name, flush=True)
