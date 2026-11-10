"""IRAF 2.18's imsum, for tests/test_imsum_iraf.py:

    python ims_inputs.py <folder>                                    (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF)
    python ims_collect.py <results>                                  (harwood's)

Every case in one copy of the images, its output o<number>.fits in the order of
ims_cases.CASES, and what imsum printed in o<number>.txt."""

import io
import shutil
import sys
from pathlib import Path

from ims_cases import CASES
from pyraf import iraf


def name(image):
    root, _, section = image.partition("[")
    return f"{root}.fits" + (f"[{section}" if section else "")


if __name__ == "__main__":
    iraf.images(_doprint=0)
    iraf.imutil(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    shutil.rmtree(results, ignore_errors=True)
    shutil.copytree(fresh, results)
    iraf.chdir(str(results))
    for number, (case, params) in enumerate(CASES.items(), start=1):
        params = dict(params)
        inputs = ",".join(name(i) for i in params.pop("input"))
        flags = {k: ("yes" if v else "no") if isinstance(v, bool) else v
                 for k, v in params.items()}
        text = io.StringIO()
        try:
            iraf.imsum(inputs, f"o{number:03d}.fits", Stdout=text, Stderr=text, **flags)
        except Exception as error:  # noqa: BLE001 -- IRAF's refusal is a result
            text.write(f"ERROR: {error}\n")
        (results / f"o{number:03d}.txt").write_text(text.getvalue(), encoding="utf-8")
        print(case, flush=True)
