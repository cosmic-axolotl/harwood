"""IRAF's catalog task (mkcatalog without the editor) given answers at its
prompts:
    python3 run_iraf.py DIR c1 c1.cols c1.out1   # the header and fc1.dat
    cp c1 c1.hdr
    python3 run_iraf.py DIR c1 c1.recs c1.out2   # the records, appended
A piped file has one end, which ends the column entry, so the records are
given in a second run, which finds the catalog and adds to it.
"""

import os
import sys

os.chdir(sys.argv[1])

from pyraf import iraf  # noqa: E402

iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.photcal(_doprint=0)
cat, answers, out = sys.argv[2:5]
iraf.catalog(cat, review=False, verify=False, Stdin=answers, Stdout=out,
             Stderr=out + ".err")
