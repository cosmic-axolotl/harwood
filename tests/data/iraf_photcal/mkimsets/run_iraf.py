"""How IRAF made the files here: mkimsets on the .mag.1 files in ../apfile.

Run from ../apfile, with the system python3 that has PyRAF:

    for c in A B C D; do python3 ../mkimsets/run_iraf.py . $c < /dev/null; done

and move the sets*, setsA.out ... and obsparA files here.
"""

import os
import sys

os.chdir(sys.argv[1])

from pyraf import iraf  # noqa: E402

iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.photcal(_doprint=0)
c = sys.argv[2]
kw = dict(edit=False, rename=False, review=False)
if c == "A":
    iraf.mkimsets("@all.lis", "B,V", "setsA", imobsparams="obsparA",
                  Stdout="setsA.out", Stderr="setsA.err", **kw)
elif c == "B":
    iraf.mkimsets("@all.lis", "V,B", "setsB", Stdout="setsB.out", Stderr="setsB.err", **kw)
elif c == "C":
    iraf.mkimsets("@../mkimsets/part.lista", "B,V,R", "setsC", Stdout="setsC.out",
                  Stderr="setsC.err", **kw)
elif c == "D":
    iraf.mkimsets("@all.lis", "B", "setsD", sort="XAIRMASS", Stdout="setsD.out",
                  Stderr="setsD.err", **kw)
