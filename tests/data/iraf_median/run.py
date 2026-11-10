"""IRAF 2.18's median, for tests/test_median_iraf.py:

    python med_inputs.py <folder>                                    (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF)
    python med_collect.py <results>                                  (harwood's)

Every case in one copy of the images, its output o<number>.fits in the order of
med_cases.CASES, and what median printed in o<number>.txt."""

import io
import shutil
import sys
from pathlib import Path

from med_cases import CASES
from pyraf import iraf

if __name__ == "__main__":
    iraf.images(_doprint=0)
    iraf.imfilter(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    shutil.rmtree(results, ignore_errors=True)
    shutil.copytree(fresh, results)
    iraf.chdir(str(results))
    for number, (case, params) in enumerate(CASES.items(), start=1):
        params = dict(params)
        given = params.pop("input")
        root, _, section = given.partition("[")
        image = given if ".fits" in given else \
            f"{root}.fits" + (f"[{section}" if section else "")
        for key in ("zloreject", "zhireject"):
            if key in params and params[key] is None:
                params[key] = iraf.INDEF
        params["verbose"] = "yes" if params.get("verbose") else "no"
        text = io.StringIO()
        try:
            iraf.median(image, f"o{number:03d}.fits", params.pop("xwindow"),
                        params.pop("ywindow"), Stdout=text, Stderr=text, **params)
        except Exception as error:  # noqa: BLE001 -- IRAF's refusal is a result
            text.write(f"ERROR: {error}\n")
        (results / f"o{number:03d}.txt").write_text(text.getvalue(), encoding="utf-8")
        print(case, flush=True)
