"""IRAF 2.18 drawing the graphs of one interactive mkapfile case: python3
run_iraf.py NAME, from a directory holding this folder's NAME.cur, all.lis and the
.mag.1 files of ../apfile, in a fresh process because the fit keeps a static.
stdvdm writes the metacode to uparm$vdm and adds to it, so it goes first and is
moved aside after; decode.py makes NAME.json of it."""
import shutil
import sys
from pathlib import Path

import yaml
from pyraf import iraf

name = sys.argv[1]
vdm = Path.home() / ".iraf/uparm/vdm"
vdm.unlink(missing_ok=True)
iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.photcal(_doprint=0)
iraf.mkapfile("@all.lis", 20, f"{name}.apercors", magfile=f"{name}.mkapfile",
              logfile=f"{name}.log", interactive="yes", verify="no",
              gcommands=f"{name}.cur", graphics="stdvdm", Stdout=f"{name}.out",
              Stderr=f"{name}.err", **yaml.safe_load(Path("CASES.yaml").read_text())[name])
shutil.move(vdm, f"{name}.vdm")
