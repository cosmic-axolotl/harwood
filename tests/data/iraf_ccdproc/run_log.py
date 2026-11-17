"""IRAF 2.18's ccdproc with ccdred's verbose and logfile on, for the log's gabarito:

    python ccdp_inputs.py <folder>                  (harwood's)
    python3 run_log.py <folder> <results>           (the one with PyRAF)
    python ccdp_log_collect.py <results>            (harwood's: results to ccdp_log.json)

The same cases as run.py; each case's logfile, its standard error (where verbose
writes), and the text of every processing flag in the header of the image."""

import shutil
import sys
from pathlib import Path

from ccdp_cases import CASES
from pyraf import iraf
from run import BASE

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.imred(_doprint=0)
    iraf.ccdred(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    only = sys.argv[3:]
    for name, (image, params) in CASES.items():
        if only and name not in only:
            continue
        work = results / name
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        iraf.chdir(str(work))
        iraf.ccdred.instrument = ""
        iraf.ccdred.pixeltype = "real real"
        iraf.ccdred.verbose = "yes"
        iraf.ccdred.logfile = "logfile"
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
