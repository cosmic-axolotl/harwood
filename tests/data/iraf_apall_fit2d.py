"""Made tests/data/iraf_apall_fit2d.json and .npz: python3 iraf_apall_fit2d.py DIR, with
A.fits (tests/iraf_field.spectra_frame(), DISPAXIS=2) and T.fits (its transpose,
DISPAXIS=1) in DIR/work, under the system Python that has PyRAF; then the .ms.fits in
DIR/out were frozen.

IRAF 2.18's apall with pfit=fit2d (Marsh's profile); polysep and polyorder are set in
apall1, the parameter file apall reads them from."""

import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(sys.argv[1]).resolve()
from pyraf import iraf  # noqa: E402

iraf.noao(_doprint=0)
iraf.twodspec(_doprint=0)
iraf.apextract(_doprint=0)
CASES = {
    "f2_variance": ("A", {"weights": "variance", "pfit": "fit2d", "readnoise": "5.",
                          "gain": "2."}),
    "f2_fit": ("A", {"weights": "variance", "pfit": "fit2d", "background": "fit",
                     "readnoise": "5.", "gain": "2."}),
    "f2_clean": ("A", {"clean": "yes", "pfit": "fit2d", "background": "fit",
                       "readnoise": "5.", "gain": "2."}),
    "f2_d1": ("T", {"weights": "variance", "pfit": "fit2d", "readnoise": "5.", "gain": "1."}),
    "f2_order": ("A", {"weights": "variance", "pfit": "fit2d", "polysep": 0.7,
                       "polyorder": 4, "readnoise": "5.", "gain": "2."}),
}
out = {}
for case, (image, params) in CASES.items():
    place = HERE / "out" / case
    shutil.rmtree(place, ignore_errors=True)
    place.mkdir(parents=True)
    shutil.copy(HERE / "work" / f"{image}.fits", place / f"{image}.fits")
    os.chdir(place)
    iraf.chdir(str(place))
    for task in ("apall", "apparams", "apdefault", "apall1"):
        iraf.unlearn(task)
    iraf.apextract.database = "database"
    iraf.apextract.logfile = "ap.log"
    call = {"interactive": "no", "find": "yes", "recenter": "yes", "resize": "yes",
            "edit": "no", "trace": "yes", "fittrace": "no", "extract": "yes",
            "extras": "yes", "review": "no", "nfind": 2, "format": "multispec",
            "output": f"{image}.ms"}
    marsh = {k: v for k, v in params.items() if k in ("polysep", "polyorder")}
    call.update({k: v for k, v in params.items() if k not in marsh})
    for key, value in marsh.items():
        setattr(iraf.apall1, key, value)
    iraf.apall(image, Stdout="/dev/null", Stderr="/dev/null", **call)
    out[case] = {"image": image, "params": params}
    os.chdir(HERE)
(HERE / "out" / "cases.json").write_text(json.dumps(out, indent=1))
