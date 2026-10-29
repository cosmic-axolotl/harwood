"""IRAF's chkconfig on CASE.cfg, quiet (CASE.q) and verbose (CASE.v).

Run with the system python3 and PyRAF, one process per case:
    python3 run_iraf.py DIR CASE < /dev/null > CASE.log 2>&1
yacc's "syntax error" goes to the process's standard error, so to CASE.log;
CASE.err is "syntax error" when it was there. f8 (a last line ending in
'#' without a newline) sends IRAF into an endless loop and is not here.
"""

import os
import sys

os.chdir(sys.argv[1])

from pyraf import iraf  # noqa: E402

iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.photcal(_doprint=0)
case = sys.argv[2]
for verbose in (False, True):
    out = f"{case}.{'v' if verbose else 'q'}"
    if os.path.exists(out):
        os.remove(out)
    iraf.chkconfig(case + ".cfg", verbose=verbose, Stdout=out)
