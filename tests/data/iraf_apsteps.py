"""Made tests/data/iraf_apsteps.json and .npz: python3 iraf_apsteps.py DIR, with the
frames A.fits, B.fits and arc.fits of tests/iraf_field in DIR/work, under the system
Python that has PyRAF; then cases.json and the .ms.fits in DIR/out were frozen.

IRAF 2.18's apfind, apresize, aprecenter, aptrace and apsum, one after the other,
each with its own parameter file's defaults (interactive=no), on the frames in
work/. Writes cases.json: per case, the steps run, the apertures in the database
after the last step, the RECENTER log line, and the extracted bands (if any)."""

import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(sys.argv[1]).resolve()
WORK = HERE / "work"
OUT = HERE / "out"

from pyraf import iraf  # noqa: E402

iraf.noao(_doprint=0)
iraf.twodspec(_doprint=0)
iraf.apextract(_doprint=0)
for task in ("apfind", "apresize", "aprecenter", "aptrace", "apsum", "apedit", "apall",
             "apdefault", "apparams"):
    iraf.unlearn(task)

CASES = {
    "find": [("apfind", "A", {})],
    "resize": [("apfind", "A", {}), ("apresize", "A", {})],
    "trace": [("apfind", "A", {}), ("apresize", "A", {}), ("aptrace", "A", {})],
    "sum": [("apfind", "A", {}), ("apresize", "A", {}), ("aptrace", "A", {}),
            ("apsum", "A", {})],
    "sum_variance": [("apfind", "A", {}), ("apresize", "A", {}), ("aptrace", "A", {}),
                     ("apsum", "A", {"background": "fit", "weights": "variance",
                                     "extras": "yes", "trace": "no"})],
    "sum_untraced": [("apfind", "A", {}), ("apsum", "A", {"trace": "no"})],
    "recenter": [("apall", "A", {}), ("aprecenter", "B", {"references": "A"}),
                 ("apsum", "B", {})],
    "sum_arc": [("apall", "A", {}), ("apsum", "arc", {"references": "A", "trace": "no"})],
}
APALL = {"interactive": "no", "find": "yes", "recenter": "yes", "resize": "yes",
         "edit": "no", "trace": "yes", "fittrace": "no", "extract": "yes",
         "extras": "yes", "review": "no", "nfind": 2, "format": "multispec"}


def apertures(image):
    text = (Path("database") / f"ap{image}").read_text()
    found = []
    for block in text.split("begin")[1:]:
        values = {}
        for line in block.splitlines():
            words = line.split()
            if words and words[0] in ("center", "low", "high"):
                values[words[0]] = [float(w) for w in words[1:3]]
        found.append([values["center"][0], values["low"][0], values["high"][0]])
    return found


results = {}
for case, steps in CASES.items():
    place = OUT / case
    shutil.rmtree(place, ignore_errors=True)
    place.mkdir(parents=True)
    for name in ("A", "B", "arc"):
        shutil.copy(WORK / f"{name}.fits", place / f"{name}.fits")
    os.chdir(place)
    iraf.chdir(str(place))
    iraf.apextract.database = "database"
    iraf.apextract.logfile = "ap.log"
    iraf.apextract.verbose = "no"
    for task, image, params in steps:
        call = {"interactive": "no"}
        if task == "apall":
            call = dict(APALL)
        elif task in ("apfind",):
            call["nfind"] = 2
        if task in ("apsum",):
            call.update(edit="no", fittrace="no", review="no", output=f"{image}.ms")
        if task in ("aptrace",):
            call.update(fittrace="no")
        call.update(params)
        getattr(iraf, task)(image, Stdout="/dev/null", Stderr="/dev/null", **call)
    last_image = steps[-1][1]
    entry = {"steps": [[t, i, p] for t, i, p in steps],
             "apertures": apertures(last_image)}
    log = [line for line in Path("ap.log").read_text().splitlines() if "RECENTER" in line] \
        if Path("ap.log").exists() else []
    entry["log"] = log[-1].split(": ", 1)[-1].strip() if log else None
    extracts = steps[-1][0] in ("apsum", "apall")
    entry["extracted"] = f"{last_image}.ms.fits" if extracts else None
    results[case] = entry
    os.chdir(HERE)

(OUT / "cases.json").write_text(json.dumps(results, indent=1))
print(json.dumps({k: (v["apertures"], v["log"]) for k, v in results.items()}, indent=1))
