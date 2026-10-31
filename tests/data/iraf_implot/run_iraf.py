"""IRAF 2.18 drawing implot's plots: python3 run_iraf.py NAME, from a directory with
the images of make_frames.py and this folder's NAME.cur (the cursor file) and
NAME.2nd (the second positions a, e, p and s read from CL's gcur, not from the
cursor file). Metacode into NAME.vdm, for ../iraf_photcal/decode_gki.py --over-lines;
what it prints into NAME.out."""
import shutil
import sys
from pathlib import Path

import yaml
from pyraf import iraf

name = sys.argv[1]
vdm = Path.home() / ".iraf/uparm/vdm"
vdm.unlink(missing_ok=True)
case = yaml.safe_load(Path("CASES.yaml").read_text())[name]
iraf.plot(_doprint=0)
if Path(f"{name}.2nd").exists():
    iraf.cl.gcur = f"{name}.2nd"
iraf.implot(",".join(case["images"]), line=case.get("line", 20),
            wcs=case.get("wcs", "logical"), coords=f"{name}.cur", device="stdvdm",
            Stdout=f"{name}.out", Stderr=f"{name}.err")
iraf.gflush()
shutil.move(vdm, f"{name}.vdm")
