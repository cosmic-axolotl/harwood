"""IRAF 2.18's scombine, for tests/test_scombine_iraf.py: its log, and each output's
lines -- aperture, beam, title, world coordinates and pixels -- and header.

    python scb_inputs.py <folder>                                    (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF)
    python scb_collect.py <results>                                  (harwood's)
"""

import io
import shutil
import sys
from pathlib import Path

from pyraf import iraf
from scb_cases import CASES


def run(params):
    params = {k: ("yes" if v else "no") if isinstance(v, bool) else v
              for k, v in params.items()}
    text = io.StringIO()
    iraf.scombine.unlearn()
    try:
        iraf.scombine(params.pop("input"), params.pop("output", "o"), logfile="STDOUT",
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
        (work / "said.txt").write_text(run(dict(params)), encoding="utf-8")
        for image in sorted(work.glob("o*.fits")) + sorted(work.glob("no.fits")):
            listed = io.StringIO()
            iraf.listpixels(image.stem, wcs="world", formats="%.14g", Stdout=listed)
            (work / f"{image.stem}.world").write_text(listed.getvalue(), encoding="utf-8")
        print(name, flush=True)
