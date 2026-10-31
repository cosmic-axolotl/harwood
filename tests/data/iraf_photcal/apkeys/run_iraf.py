"""IRAF 2.18 on one interactive mkapfile case, in a fresh process because the
fit keeps a static between runs: python3 run_iraf.py NAME < /dev/null. stdvdm
takes the graphics, so no screen is needed; what the keys print is in NAME.out."""
import sys

from pyraf import iraf

name = sys.argv[1]
iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.photcal(_doprint=0)
iraf.mkapfile("@all.lis", 20, f"{name}.apercors", magfile=f"{name}.mkapfile",
              logfile=f"{name}.log", interactive="yes", verify="no",
              gcommands=f"{name}.cur", graphics="stdvdm", Stdout=f"{name}.out",
              Stderr=f"{name}.err")
