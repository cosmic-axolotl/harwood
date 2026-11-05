"""IRAF 2.18's lcalib, for tests/test_lcalib_iraf.py: what it prints, on both
streams, in lc.json.

    HOME=<scratch with .iraf/uparm> python3 run.py <scratch> <here>     (with PyRAF)
"""

import io
import json
import sys
from pathlib import Path

from lc_cases import CASES
from pyraf import iraf

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    work = Path(sys.argv[1]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    iraf.chdir(str(work))
    out = {}
    for name, params in CASES.items():
        stdout, stderr = io.StringIO(), io.StringIO()
        iraf.lcalib(Stdout=stdout, Stderr=stderr, **params)
        out[name] = {"stdout": stdout.getvalue().splitlines(),
                     "stderr": stderr.getvalue().splitlines()}
    (Path(sys.argv[2]) / "lc.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
