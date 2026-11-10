"""IRAF 2.18's imreplace, for tests/test_imreplace_iraf.py:

    python imr_inputs.py <folder>                                    (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF)
    python imr_collect.py <results>                                  (harwood's)

Each case on a fresh copy of the images, which imreplace edits in place."""

import io
import shutil
import sys
from pathlib import Path

from imr_cases import CASES
from pyraf import iraf

if __name__ == "__main__":
    iraf.images(_doprint=0)
    iraf.imutil(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    for name, params in CASES.items():
        work = results / name
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        iraf.chdir(str(work))
        params = dict(params)
        image = params.pop("image")
        if "[" not in image:
            image += ".fits"
        else:
            root, section = image.split("[", 1)
            image = f"{root}.fits[{section}"
        value = params.pop("value")
        text = io.StringIO()
        try:
            iraf.imreplace(image, value, lower=params.get("lower", iraf.INDEF),
                           upper=params.get("upper", iraf.INDEF),
                           radius=params.get("radius", 0.0),
                           imaginary=params.get("imaginary", 0.0),
                           Stdout=text, Stderr=text)
        except Exception as error:  # noqa: BLE001 -- IRAF's refusal is a result
            text.write(f"ERROR: {error}\n")
        (work / "said.txt").write_text(text.getvalue(), encoding="utf-8")
        print(name, flush=True)
