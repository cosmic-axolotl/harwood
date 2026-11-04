"""Made tests/data/iraf_apall_subaps.json and .npz: python3 iraf_apall_subaps.py DIR,
with A.fits (tests/iraf_field.spectra_frame(), DISPAXIS=2) and T.fits (its transpose,
DISPAXIS=1) in DIR/work, under the system Python that has PyRAF; then the .ms.fits in
DIR/out were frozen.

IRAF 2.18's apall with nsubaps on tests/iraf_field.spectra_frame() (A, DISPAXIS=2)
and its transpose (T, DISPAXIS=1); everything else as iraf_apall.json's cases."""
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
    "s2_none": ("A", {"nsubaps": 2}),
    "s3_none": ("A", {"nsubaps": 3}),
    "s3_fit": ("A", {"nsubaps": 3, "background": "fit"}),
    "s2_median": ("A", {"nsubaps": 2, "background": "median"}),
    "s2_variance": ("A", {"nsubaps": 2, "weights": "variance", "readnoise": "5.",
                          "gain": "2."}),
    "s3_clean": ("A", {"nsubaps": 3, "clean": "yes", "readnoise": "5.", "gain": "2.",
                       "background": "fit"}),
    "s2_d1": ("T", {"nsubaps": 2, "background": "average"}),
}
out = {}
for case, (image, params) in CASES.items():
    place = HERE / "out" / case
    shutil.rmtree(place, ignore_errors=True)
    place.mkdir(parents=True)
    shutil.copy(HERE / "work" / f"{image}.fits", place / f"{image}.fits")
    os.chdir(place)
    iraf.chdir(str(place))
    iraf.apextract.database = "database"
    iraf.apextract.logfile = "ap.log"
    call = {"interactive": "no", "find": "yes", "recenter": "yes", "resize": "yes",
            "edit": "no", "trace": "yes", "fittrace": "no", "extract": "yes",
            "extras": "yes", "review": "no", "nfind": 2, "format": "multispec",
            "output": f"{image}.ms"}
    call.update(params)
    iraf.apall(image, Stdout="/dev/null", Stderr="/dev/null", **call)
    out[case] = {"image": image, "params": params}
    os.chdir(HERE)
(HERE / "out" / "cases.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
