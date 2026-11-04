"""How IRAF made the files here: obsfile on the automatico's .mkapfile files.

The .mkapfile files are mkapfile's best magnitudes with an eleventh column,
the star's name, which the automatico pipeline adds. sets2 holds one of
each kind of bad line; shifts and obsparams name an image that is in no
set. Run with the system python3 that has PyRAF:

    for c in A B C D E F G S1 S2 S3 S4 S5; do python3 run_iraf.py . $c < /dev/null; done

S1 to S5 type a table at the terminal: what was typed is the task's standard input.
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
kw = dict(kw)
typed = {"Stdin": kw.pop("answers")} if "answers" in kw else {}
iraf.obsfile("@mkapfile.lista", cols, "B,V", sets, "obs" + c,
             verify=kw.pop("verify", False), Stdout="obs" + c + ".out",
             Stderr="obs" + c + ".err", **typed, **kw)
