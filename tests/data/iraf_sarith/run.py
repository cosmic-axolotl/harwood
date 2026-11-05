"""IRAF 2.18's sarith, for tests/test_sarith_iraf.py: what it says, and each output's
lines and bands -- their aperture, beam, title, world coordinates and pixels.

    python sa_inputs.py <folder>                                    (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF)
    python sa_collect.py <results>                                  (harwood's)
"""

import io
import shutil
import sys
from pathlib import Path

from pyraf import iraf
from sa_cases import CASES

BASE = dict(input2="", w1="INDEF", w2="INDEF", apertures="", bands="", beams="",
            apmodulus=0, reverse="no", ignoreaps="no", format="multispec", renumber="no",
            offset=0, clobber="no", merge="no", rebin="yes", errval=0.0, verbose="yes")


def run(params, work):
    params = {**BASE, **{k: ("yes" if v else "no") if isinstance(v, bool) else v
                         for k, v in params.items()}}
    text = io.StringIO()
    try:
        iraf.sarith(params.pop("input1"), params.pop("op"), params.pop("input2"), "o",
                    Stdout=text, Stderr=text, **params)
    except Exception as error:  # noqa: BLE001
        text.write("ERROR " + str(error).splitlines()[-1] + "\n")
    return text.getvalue()


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
        then = params.pop("then", None)
        said = run(params, work)
        if then is not None:
            said += "----\n" + run(then, work)
        (work / "said.txt").write_text(said, encoding="utf-8")
        for image in sorted(work.glob("o*.fits")):
            stem = image.stem
            listed = io.StringIO()
            iraf.listpixels(stem, wcs="world", formats="%.14g", Stdout=listed)
            (work / f"{stem}.world").write_text(listed.getvalue(), encoding="utf-8")
        print(name, flush=True)
