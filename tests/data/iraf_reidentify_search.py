"""Made tests/data/iraf_reidentify_search: python3 iraf_reidentify_search.py DIR, with
arc3.fits and arc3b.fits (multispec_night()'s arcs, written by harwood's apextract
multispec writer with extras off), arc2d.fits and arc_lines.dat in DIR/work, under the
system Python that has PyRAF; then each case's database/ and the log were frozen.

IRAF 2.18's reidentify with a nonzero search, after autoidentify."""

import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(sys.argv[1]).resolve()
from pyraf import iraf  # noqa: E402

iraf.noao(_doprint=0)
iraf.onedspec(_doprint=0)
CASES = {
    "search_ms": ("arc3", "arc3,arc3b", {"shift": "INDEF", "search": "5.", "nlost": 40}),
    "search_2d": ("arc2d", "arc2d", {"trace": False, "step": "25", "nsum": "5", "nlost": 30,
                                     "shift": "INDEF", "search": "3."}),
    "search_trace": ("arc2d", "arc2d", {"trace": True, "step": "20", "nsum": "10",
                                        "nlost": 30, "shift": "INDEF", "search": "2."}),
}
out = {}
for case, (reference, images, params) in CASES.items():
    place = HERE / "out" / case
    shutil.rmtree(place, ignore_errors=True)
    shutil.copytree(HERE / "work", place)
    os.chdir(place)
    iraf.chdir(str(place))
    for task in ("autoidentify", "reidentify", "identify", "aidpars"):
        iraf.unlearn(task)
    iraf.autoidentify(reference, 4160.0, 0.29, coordlist="arc_lines.dat", interactive="no",
                      nsum=10 if reference == "arc2d" else 1, verbose="no",
                      logfile="", plotfile="", Stdout="/dev/null")
    call = {"interactive": "no", "verbose": "yes", "coordlist": "arc_lines.dat",
            "logfiles": "log", "plotfile": ""}
    call.update(params)
    iraf.reidentify(reference, images, Stdout="/dev/null", Stderr="/dev/null", **call)
    log = Path("log").read_text().splitlines()[2:] if Path("log").exists() else []
    out[case] = {"reference": reference, "images": images, "params": params, "log": log}
    os.chdir(HERE)
(HERE / "out" / "cases.json").write_text(json.dumps(out, indent=1))
