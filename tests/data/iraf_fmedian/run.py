"""IRAF 2.18's fmedian, for tests/test_fmedian_iraf.py:

    python ../iraf_fmedian/fmd_inputs.py <folder>                    (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF)
    python fmd_collect.py <results>                                  (harwood's)

Every case in one copy of the images, its output o<number>.fits in the order of
fmd_cases.CASES, and what fmedian printed in o<number>.txt."""

import io
import shutil
import sys
from pathlib import Path

from fmd_cases import CASES
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
        root, _, section = params.pop("input").partition("[")
        image = f"{root}.fits" + (f"[{section}" if section else "")
        for key, value in list(params.items()):
            if value is None:
                params[key] = iraf.INDEF
            elif isinstance(value, bool):
                params[key] = "yes" if value else "no"
        params.setdefault("verbose", "no")
        text = io.StringIO()
        try:
            iraf.fmedian(image, f"o{number:03d}.fits", params.pop("xwindow"),
                    params.pop("ywindow"), Stdout=text, Stderr=text, **params)
        except Exception as error:  # noqa: BLE001 -- IRAF's refusal is a result
            text.write(f"ERROR: {error}\n")
        (results / f"o{number:03d}.txt").write_text(text.getvalue(), encoding="utf-8")
        print(case, flush=True)
