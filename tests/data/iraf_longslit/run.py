"""IRAF 2.18's response and illumination, for tests/test_longslit_iraf.py:

    python ls_inputs.py <folder>                                     (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF;
    a HOME of its own: response saves its parameters, and unlearn resets them)
    python ls_collect.py <results>                                   (harwood's)

Each case on a fresh copy of the images (response writes into an existing image),
interactive=no; its output o.fits, or the pre-existing image it wrote into."""

import io
import shutil
import sys
from pathlib import Path

from ls_cases import ILLUMINATION, RESPONSE
from pyraf import iraf


def name(image):
    root, _, section = image.partition("[")
    return f"{root}.fits" + (f"[{section}" if section else "")


if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.twodspec(_doprint=0)
    iraf.longslit(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    for task, cases in (("response", RESPONSE), ("illumination", ILLUMINATION)):
        for case, params in cases.items():
            work = results / case
            shutil.rmtree(work, ignore_errors=True)
            shutil.copytree(fresh, work)
            iraf.chdir(str(work))
            params = dict(params)
            # response writes its fit back to its parameter file (clputi): each
            # case starts from the defaults, not from the case before.
            iraf.unlearn(task)
            iraf.longslit.dispaxis = params.pop("dispaxis", 1)
            output = name(params.pop("output", "o"))
            text = io.StringIO()
            try:
                if task == "response":
                    iraf.response(name(params.pop("calibration")),
                                  name(params.pop("normalization")), output,
                                  interactive="no", Stdout=text, Stderr=text, **params)
                else:
                    iraf.illumination(name(params.pop("images")), output,
                                      interactive="no", Stdout=text, Stderr=text,
                                      **params)
            except Exception as error:  # noqa: BLE001 -- IRAF's refusal is a result
                text.write(f"ERROR: {error}\n")
            (work / "said.txt").write_text(text.getvalue(), encoding="utf-8")
            print(case, flush=True)
