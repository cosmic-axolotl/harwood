"""IRAF 2.18's slist, for tests/test_slist_iraf.py: what it prints, in sl.json.

    python sl_inputs.py <folder>                                    (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <here>  (with PyRAF)
"""

import io
import json
import sys
from pathlib import Path

from pyraf import iraf
from sl_cases import CASES

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    iraf.chdir(str(Path(sys.argv[1]).resolve()))
    out = {}
    for name, params in CASES.items():
        params = {k: ("yes" if v else "no") if isinstance(v, bool) else v
                  for k, v in params.items()}
        text = io.StringIO()
        iraf.slist(Stdout=text, Stderr=text, **params)
        out[name] = text.getvalue().splitlines()
    (Path(sys.argv[2]) / "sl.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
