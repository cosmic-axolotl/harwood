"""IRAF 2.18's disptrans, for tests/test_disptrans_iraf.py: what it says, each line's
world coordinates and the pixels.

    python ../iraf_dopcor/dop_inputs.py <folder>                    (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF)
    python dt_collect.py <results>                                  (harwood's)
"""

import io
import shutil
import sys
from pathlib import Path

from dt_cases import CASES
from pyraf import iraf

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    base = dict(error=0.01, linearize="no", verbose="yes", air="none", t=15.0, p=760.0,
                f=4.0)
    for name, params in CASES.items():
        work = results / name
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        iraf.chdir(str(work))
        params = {**base, **{k: ("yes" if v else "no") if isinstance(v, bool) else v
                             for k, v in params.items()}}
        text = io.StringIO()
        try:
            iraf.disptrans("in", "o", Stdout=text, Stderr=text, **params)
        except Exception as error:  # noqa: BLE001
            text.write("ERROR " + str(error).splitlines()[-1] + "\n")
        (work / "said.txt").write_text(text.getvalue(), encoding="utf-8")
        if not (work / "o.fits").exists():
            continue
        for line in (1, 2, 3):
            listed = io.StringIO()
            iraf.listpixels(f"o[*,{line}]", wcs="world", formats="%.14g", Stdout=listed)
            (work / f"world{line}.txt").write_text(listed.getvalue(), encoding="utf-8")
        print(name, flush=True)
