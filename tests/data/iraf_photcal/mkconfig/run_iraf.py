"""IRAF's mkconfig, without the editor, then chkconfig:
    python3 run_iraf.py DIR m1 landolt obsA landolt ""
    python3 run_iraf.py DIR m2 c1 obsA mine ""
with fobsA.dat (from ../obsfile), fc1.dat and tmine.dat in DIR.
"""

import os
import sys

os.chdir(sys.argv[1])

from pyraf import iraf  # noqa: E402

iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.photcal(_doprint=0)
config, catalog, observations, transform, template = sys.argv[2:7]
iraf.mkconfig(config, catalog, observations, transform, template=template, edit=False,
              check=True, verbose=True, Stdout=config + ".out", Stderr=config + ".err")
