"""How IRAF made the files here: obsfile on the automatico's .mkapfile files.

The .mkapfile files are mkapfile's best magnitudes with an eleventh column,
the star's name, which the automatico pipeline adds. sets2 holds one of
each kind of bad line; shifts and obsparams name an image that is in no
set. Run with the system python3 that has PyRAF:

    for c in A B C D E F G; do python3 run_iraf.py . $c < /dev/null; done
"""

import os
import sys

os.chdir(sys.argv[1])

from cases import CASES  # noqa: E402
from pyraf import iraf  # noqa: E402

iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.photcal(_doprint=0)
c = sys.argv[2]
sets, cols, kw = CASES[c]
iraf.obsfile("@mkapfile.lista", cols, "B,V", sets, "obs" + c, verify=False,
             Stdout="obs" + c + ".out", Stderr="obs" + c + ".err", **kw)
