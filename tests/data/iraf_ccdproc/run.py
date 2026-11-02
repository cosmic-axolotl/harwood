"""IRAF 2.18's ccdproc, for tests/test_ccdproc_more_iraf.py.

    python ccdp_inputs.py <folder>            (harwood's)
    python3 run.py <folder> <results>         (the one with PyRAF)
    python ccdp_collect.py <results>          (harwood's: results to ccdp.json)
"""

import shutil
import sys
from pathlib import Path

from ccdp_cases import CASES
from pyraf import iraf

iraf.noao(_doprint=0)
iraf.imred(_doprint=0)
iraf.ccdred(_doprint=0)
fresh = Path(sys.argv[1]).resolve()
results = Path(sys.argv[2]).resolve()
only = sys.argv[3:]
BASE = dict(output="", ccdtype="", max_cache=0, noproc="no", fixpix="no", overscan="no",
            trim="no", zerocor="no", darkcor="no", flatcor="no", illumcor="no", fringecor="no",
            readcor="no", scancor="no", readaxis="line", fixfile="", biassec="", trimsec="",
            zero="", dark="", flat="", illum="", fringe="", minreplace=1.0,
            scantype="shortscan", nscan=1, interactive="no", function="legendre", order=1,
            sample="*", naverage=1, niterate=1, low_reject=3.0, high_reject=3.0, grow=0.0)
for name, (image, params) in CASES.items():
    if only and name not in only:
        continue
    work = results / name
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(fresh, work)
    iraf.chdir(str(work))
    iraf.ccdred.instrument = ""
    iraf.ccdred.pixeltype = "real real"
    iraf.ccdred.verbose = "no"
    iraf.ccdred.logfile = ""
    iraf.ccdred.plotfile = ""
    iraf.ccdred.backup = ""
    iraf.ccdred.ssfile = "subsets"
    error = ""
    try:
        iraf.flprcache()
        iraf.ccdproc(image, mode="h", Stderr="err.txt", **{**BASE, **params})
    except Exception as e:  # noqa: BLE001
        error = str(e)[-300:]
    (work / "error.txt").write_text(error, encoding="utf-8")
    print(name, error[-160:].replace("\n", " "), flush=True)
