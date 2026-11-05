"""IRAF 2.18's dopcor, for tests/test_dopcor_iraf.py.

    python dop_inputs.py <folder>                          (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF)
    python dop_collect.py <results>                        (harwood's: dop.json)
"""

import io
import shutil
import sys
from pathlib import Path

from dop_cases import CASES
from pyraf import iraf

iraf.noao(_doprint=0)
iraf.onedspec(_doprint=0)
fresh = Path(sys.argv[1]).resolve()
results = Path(sys.argv[2]).resolve()
BASE = dict(isvelocity=False, add=False, dispersion=True, flux=False, factor=3.0,
            apertures="", verbose=False)

for name, chain in CASES.items():
    work = results / name
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(fresh, work)
    iraf.chdir(str(work))
    source = "in"
    for number, step in enumerate(chain, 1):
        params = {**BASE, **step}
        for key, value in params.items():
            if isinstance(value, bool):
                params[key] = "yes" if value else "no"
        text = io.StringIO()
        try:
            iraf.dopcor(source, f"o{number}", Stdout=text, Stderr=text, **params)
        except Exception as error:  # noqa: BLE001
            text.write(f"ERROR: {str(error)[-200:]}\n")
        (work / f"o{number}.txt").write_text(text.getvalue(), encoding="utf-8")
        source = f"o{number}"
    print(name, flush=True)
