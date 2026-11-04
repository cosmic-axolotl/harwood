"""How IRAF made the files here: mkapfile on the Skinakas .mag.1 files.

Run with the system python3 that has PyRAF, one process per case -- the
fit keeps a static variable from one run to the next inside a process:

    for f in *.fits.mag.1; do python3 run_iraf.py . one:${f%.fits.mag.1} < /dev/null; done
    for c in M P5 P4; do python3 run_iraf.py . $c < /dev/null; done
    for c in O O2 OS; do python3 run_iraf.py . $c < /dev/null; done

O and O2 take the images' filter, exposure, airmass and time from obs1.dat, and
the airmass alone from obs2.dat; OS has them typed at the terminal (OS.ans, its
standard input), with verify on. What they print is in O.out, O2.out and OS.out.
"""

import os
import sys

os.chdir(sys.argv[1])

from pyraf import iraf  # noqa: E402

iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.photcal(_doprint=0)
case = sys.argv[2]
common = dict(interactive=False, verify=False)
if case.startswith("one:"):
    r = case[4:]
    iraf.mkapfile(r + ".fits.mag.1", 20, r + ".apercors", smallap=1, largeap=11,
                  magfile=r + ".mkapfile", logfile=r + ".mkapfile.log", **common)
elif case == "M":
    iraf.mkapfile("@all.lis", 20, "M.apercors", magfile="M.mkapfile", logfile="M.log",
                  **common)
elif case == "P5":
    iraf.mkapfile("@all.lis", 20, "P5.apercors", smallap=2, largeap=15, maglim=0.05,
                  nparams=5, magfile="P5.mkapfile", logfile="P5.log", **common)
elif case == "P4":
    iraf.mkapfile("@all.lis", 20, "P4.apercors", smallap=3, largeap=0, nparams=4,
                  xwings=0.1, swings=1.5, magfile="P4.mkapfile", logfile="P4.log", **common)
elif case == "O":
    iraf.mkapfile("@all.lis", 20, "O.apercors", magfile="O.mkapfile", logfile="O.log",
                  obsparams="obs1.dat", Stdout="O.out", **common)
elif case == "O2":
    iraf.mkapfile("@all.lis", 20, "O2.apercors", magfile="O2.mkapfile", logfile="O2.log",
                  obsparams="obs2.dat", obscolumns="0 0 2 0", Stdout="O2.out", **common)
elif case == "OS":
    iraf.mkapfile("@all.lis", 20, "OS.apercors", magfile="OS.mkapfile", logfile="OS.log",
                  obsparams="STDIN", interactive=False, verify=True, Stdin="OS.ans",
                  Stdout="OS.out")
