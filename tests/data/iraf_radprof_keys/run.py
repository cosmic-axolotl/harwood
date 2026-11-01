"""IRAF 2.18's radprof by the cursor, for tests/test_radprof_iraf.py: an icommands file a case.

    HOME=<a scratch folder with .iraf/uparm> python3 run.py <folder with field.fits, field.coo>
"""

import json
import os
import sys
from pathlib import Path

from pyraf import iraf

CASES = {
    "list": "0 0 1 n\n0 0 1 n\n0 0 1 m\n0 0 1 q\n",
    "rest": "0 0 1 l\n0 0 1 q\n",
    "cursor": "41 45 1 \\040\n110 53 1 f\n110 53 1 \\040\n161 141 1 c\n161 141 1 s\n"
              "161 141 1 o\n0 0 1 q\n",
    "colon": "0 0 1 :m 2\n110 53 1 \\040\n0 0 1 :n 3\n0 0 1 :radius 8\n0 0 1 :order 3\n"
              "161 141 1 \\040\n0 0 1 :step 1\n41 45 1 \\040\n0 0 1 q\n",
    "rewind": "0 0 1 n\n0 0 1 r\n0 0 1 n\n0 0 1 :n 9\n0 0 1 q\n",
}
os.chdir(sys.argv[1])
iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.apphot(_doprint=0)
iraf.datapars.fwhmpsf = 2.5
iraf.datapars.sigma = "INDEF"
iraf.datapars.epadu = 2.0
iraf.centerpars.calgorithm = "centroid"
iraf.centerpars.cbox = 5.0
iraf.fitskypars.salgorithm = "centroid"
iraf.fitskypars.annulus = 10.0
iraf.fitskypars.dannulus = 10.0
iraf.photpars.apertures = "3,5"
iraf.photpars.zmag = 25.0
gold = {}
for name, keys in CASES.items():
    Path(f"{name}.cur").write_text(keys)
    Path(f"{name}.prf").unlink(missing_ok=True)
    error = ""
    try:
        iraf.flprcache()
        iraf.radprof("field.fits", 12.0, 0.5, coords="field.coo", output=f"{name}.prf",
                     order=5, kreject=3.0, nreject=1, interactive="yes",
                     icommands=f"{name}.cur", gcommands="", radplots="no",
                     graphics="stdvdm", display="stdvdm",
                     verify="no", update="no", verbose="no", plotfile="", mode="h",
                     Stdout=f"{name}.out", Stderr=f"{name}.err")
    except Exception as e:  # noqa: BLE001
        error = str(e)[-300:]
    written = Path(f"{name}.prf")
    gold[name] = {"keys": keys, "error": error,
                  "prf": written.read_text() if written.exists() else ""}
    print(name, repr(error[-120:]), len(gold[name]["prf"]))
(Path(__file__).parent / "radprof_keys.json").write_text(json.dumps(gold, indent=0))
