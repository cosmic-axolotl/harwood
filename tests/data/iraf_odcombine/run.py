"""IRAF 2.18.1's odcombine, for tests/test_odcombine_iraf.py: its log, and each output's
pixels, world coordinates and header.

    sh build.sh <scratch>                                            (the gabarito)
    python odc_inputs.py <folder>                                    (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <scratch>/odc <folder> <results>
    python odc_collect.py <results>                                  (harwood's)
"""

import io
import shutil
import sys
from pathlib import Path

from odc_cases import CASES
from pyraf import iraf


def run(params):
    params = {k: ("yes" if v else "no") if isinstance(v, bool) else v
              for k, v in params.items()}
    text = io.StringIO()
    iraf.odcombine.unlearn()
    try:
        iraf.odcombine(params.pop("input"), params.pop("output", "o"), logfile="STDOUT",
                       Stdout=text, Stderr=text, **params)
    except Exception as error:  # noqa: BLE001
        text.write("ERROR " + str(error).splitlines()[-1] + "\n")
    return text.getvalue()


if __name__ == "__main__":
    built = Path(sys.argv[1]).resolve()
    fresh = Path(sys.argv[2]).resolve()
    results = Path(sys.argv[3]).resolve()
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    iraf.task(odcombine=str(built / "xx_odcombine.e"))
    for name, params in CASES.items():
        work = results / name
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        iraf.chdir(str(work))
        (work / "said.txt").write_text(run(dict(params)), encoding="utf-8")
        inputs = {p.name for p in fresh.iterdir()}
        for mask in sorted(work.glob("*.pl")):
            # The masks are pixel lists; their values as an image of integers.
            iraf.imcopy(mask.name, mask.stem + "_pl.fits", verbose="no")
        for image in sorted(work.glob("*.fits")):
            if image.name in inputs:
                continue
            if image.name.endswith("_pl.fits"):
                continue
            listed = io.StringIO()
            iraf.listpixels(image.name[:-5], wcs="world", formats="%.14g", Stdout=listed)
            (work / f"{image.name[:-5]}.world").write_text(listed.getvalue(),
                                                          encoding="utf-8")
        print(name, sorted(p.name for p in work.iterdir() if p.name not in inputs),
              flush=True)
