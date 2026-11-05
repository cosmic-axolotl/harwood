"""IRAF 2.18's sflip, for tests/test_sflip_iraf.py, on ../iraf_dopcor's spectrum:
the output's pixels, and listpixels' world and physical coordinates of each line.

    python ../iraf_dopcor/dop_inputs.py <folder>                    (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF)
    python collect.py <results>                                     (harwood's)
"""

import io
import shutil
import sys
from pathlib import Path

from pyraf import iraf
from sf_cases import CASES

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    for name, params in CASES.items():
        work = results / name
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        iraf.chdir(str(work))
        params = dict(params)
        output = params.pop("output", "o")
        flags = {k: "yes" if v else "no" for k, v in params.items()}
        text = io.StringIO()
        iraf.sflip("in", output, Stdout=text, Stderr=text, **flags)
        (work / "said.txt").write_text(text.getvalue(), encoding="utf-8")
        for line in (1, 2, 3):
            for wcs in ("world", "physical"):
                listed = io.StringIO()
                iraf.listpixels(f"{output}[*,{line}]", wcs=wcs, formats="%.12g",
                                Stdout=listed)
                (work / f"{wcs}{line}.txt").write_text(listed.getvalue(), encoding="utf-8")
        print(name, flush=True)
