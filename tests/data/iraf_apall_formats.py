"""Made tests/data/iraf_apall_formats.json and .npz: python3 iraf_apall_formats.py DIR, with
the frames A, T, L and W that the .json describes in DIR/work, under the system Python that
has PyRAF; then the files in DIR/out were frozen."""

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
for task in ("apall", "apparams", "apdefault"):
    iraf.unlearn(task)
CASES = {
    "ms": ("A", {"format": "multispec", "extras": "no"}),
    "ms_extras": ("A", {"format": "multispec", "weights": "variance", "background": "fit"}),
    "ms_one": ("A", {"format": "multispec", "extras": "no", "nfind": 1}),
    "ms_one_extras": ("A", {"format": "multispec", "nfind": 1, "background": "fit"}),
    "onedspec": ("A", {"format": "onedspec", "extras": "no"}),
    "onedspec_extras": ("A", {"format": "onedspec", "background": "fit"}),
    "onedspec_subaps": ("A", {"format": "onedspec", "extras": "no", "nsubaps": 2}),
    "echelle": ("A", {"format": "echelle", "extras": "no"}),
    "echelle_subaps": ("A", {"format": "echelle", "extras": "no", "nsubaps": 2}),
    "strip": ("A", {"format": "strip", "extras": "no"}),
    "ms_d1": ("T", {"format": "multispec", "extras": "no"}),
    "strip_d1": ("T", {"format": "strip", "extras": "no"}),
    "strip_variance": ("A", {"format": "strip", "extras": "no", "weights": "variance",
                             "readnoise": "5.", "gain": "2."}),
    "strip_background": ("A", {"format": "strip", "extras": "no", "background": "fit"}),
    "ms_ltv": ("L", {"format": "multispec", "extras": "no"}),
    "ms_wcs": ("W", {"format": "multispec", "extras": "no"}),
    "onedspec_wcs": ("W", {"format": "onedspec", "extras": "no"}),
}
for case, (image, params) in CASES.items():
    place = HERE / "out" / case
    shutil.rmtree(place, ignore_errors=True)
    place.mkdir(parents=True)
    shutil.copy(HERE / "work" / f"{image}.fits", place / f"{image}.fits")
    os.chdir(place)
    iraf.chdir(str(place))
    iraf.apextract.database = "database"
    call = {"interactive": "no", "find": "yes", "recenter": "yes", "resize": "yes",
            "edit": "no", "trace": "yes", "fittrace": "no", "extract": "yes",
            "extras": "yes", "review": "no", "nfind": 2}
    call.update(params)
    iraf.apall(image, Stdout="/dev/null", Stderr="/dev/null", **call)
    os.chdir(HERE)
(HERE / "out" / "cases.json").write_text(json.dumps(CASES, indent=1))
