"""IRAF 2.18's imcombine with offsets, masks and project, for test_imcombine_geometry_iraf.py.

    python imcgeo_inputs.py <folder>          (harwood's)
    python3 run.py <folder> <results>         (the one with PyRAF)
    python imcgeo_collect.py <results>        (harwood's: results to imcgeo.json)
"""

import shutil
import sys
from pathlib import Path

from imcgeo_cases import ALL, CASES, OUTPUTS
from pyraf import iraf

iraf.images(_doprint=0)
iraf.immatch(_doprint=0)
fresh = Path(sys.argv[1]).resolve()
results = Path(sys.argv[2]).resolve()
only = sys.argv[3:]
BASE = dict(headers="", imcmb="$I", logfile="", combine="average", reject="none", project="no",
            outtype="real", outlimits="", offsets="none", masktype="none", maskvalue="0",
            blank=0.0, scale="none", zero="none", weight="none", statsec="", expname="EXPTIME",
            lthreshold="INDEF", hthreshold="INDEF", nlow=1, nhigh=1, nkeep=1, mclip="yes",
            lsigma=3.0, hsigma=3.0, rdnoise="0.", gain="1.", snoise="0.", sigscale=0.1,
            pclip=-0.5, grow=0.0)
for name, params in CASES.items():
    if only and name not in only:
        continue
    work = results / name
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(fresh, work)
    iraf.chdir(str(work))
    params = dict(params)
    images = params.pop("images", ALL)
    error = ""
    try:
        iraf.flprcache()
        iraf.imcombine(images, "out.fits", sigmas=OUTPUTS["sigmas"],
                       rejmasks=OUTPUTS["rejmasks"],
                       nrejmasks=OUTPUTS["nrejmasks"], bpmasks=OUTPUTS["bpmasks"],
                       expmasks=OUTPUTS["expmasks"], mode="h", Stderr="err.txt",
                       **{**BASE, **params})
    except Exception as e:  # noqa: BLE001
        error = str(e)[-200:]
    (work / "error.txt").write_text(error, encoding="utf-8")
    print(name, error[-90:], flush=True)
