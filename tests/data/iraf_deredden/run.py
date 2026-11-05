"""IRAF 2.18's deredden, for tests/test_deredden_iraf.py.

    python der_inputs.py <folder>                                   (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF)
    python der_collect.py <results>                                 (harwood's)
"""

import io
import shutil
import sys
from pathlib import Path

from der_cases import CASES
from pyraf import iraf

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    BASE = dict(R=3.1, type="E(B-V)", apertures="", override=False, uncorrect=True)
    for name, (source, chain) in CASES.items():
        work = results / name
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        iraf.chdir(str(work))
        for number, step in enumerate(chain, 1):
            params = {**BASE, **step}
            params = {k: ("yes" if v else "no") if isinstance(v, bool) else v
                      for k, v in params.items()}
            text = io.StringIO()
            try:
                iraf.deredden(source, f"o{number}", Stdout=text, Stderr=text, **params)
            except Exception as error:  # noqa: BLE001
                text.write(f"ERROR: {str(error)[-200:]}\n")
            (work / f"o{number}.txt").write_text(text.getvalue(), encoding="utf-8")
            source = f"o{number}"
        print(name, flush=True)
