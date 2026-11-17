"""IRAF 2.18's ccdred.combine and its presets with the package's logfile and verbose on,
for the log's gabarito (tests/test_ccdcombine_log_iraf.py):

    python make.py <folder>; python3 run_log.py <folder> <results>      (PyRAF)

Each case's logfile and what it printed (verbose: the same log on the standard output),
the date as cnvdate writes it cut off by the test."""

import json
import shutil
import sys
from pathlib import Path

from cases import CASES
from pyraf import iraf

BASE = dict(plfile="", sigma="", ccdtype="", subsets="no", delete="no", clobber="no",
            combine="average", reject="none", project="no", outtype="real", offsets="none",
            masktype="none", maskvalue=0.0, blank=0.0, scale="none", zero="none",
            weight="none", statsec="", lthreshold="INDEF", hthreshold="INDEF", nlow=1,
            nhigh=1, nkeep=1, mclip="yes", lsigma=3.0, hsigma=3.0, rdnoise="0.", gain="1.",
            snoise="0.", sigscale=0.1, pclip=-0.5, grow=0)
PRESETS = {"zerocombine": dict(ccdtype="", process="no"),
           "darkcombine": dict(ccdtype="", process="no"),
           "flatcombine": dict(ccdtype="", subsets="no", process="no")}

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.imred(_doprint=0)
    iraf.ccdred(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    out = {}
    jobs = [(name, "combine", params) for name, params in CASES.items()]
    jobs += [(f"preset_{t}", t, p) for t, p in PRESETS.items()]
    for name, task, params in jobs:
        work = results / name
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        iraf.chdir(str(work))
        iraf.ccdred.instrument = ""
        iraf.ccdred.ssfile = "subsets"
        iraf.ccdred.verbose = "yes"
        iraf.ccdred.logfile = "logfile"
        params = dict(params)
        images = params.pop("images", "@all.lis")
        printed = work / "printed.txt"
        error = ""
        try:
            if task == "combine":
                iraf.ccdred.combine(images, "out", mode="h", Stdout=str(printed),
                                    **{**BASE, **params})
            else:
                iraf.unlearn(task)
                getattr(iraf, task)(images, output="out", mode="h", Stdout=str(printed),
                                    **params)
        except Exception as e:  # noqa: BLE001
            error = str(e)[-200:]
        out[name] = {"task": task, "params": params, "images": images, "error": error,
                     "logfile": (work / "logfile").read_text(encoding="utf-8")
                     if (work / "logfile").exists() else "",
                     "printed": printed.read_text(encoding="utf-8")
                     if printed.exists() else ""}
        print(name, error[-80:], flush=True)
    Path(__file__).with_name("combine_log.json").write_text(json.dumps(out, indent=0),
                                                             encoding="utf-8")
