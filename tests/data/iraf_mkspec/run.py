"""IRAF 2.18's mkspec, for tests/test_mkspec_iraf.py: each case's pixels and title,
in mk.json.

    HOME=<scratch with .iraf/uparm> python3 run.py <scratch> <here>     (with PyRAF)
"""

import io
import json
import sys
from pathlib import Path

from mk_cases import CASES
from pyraf import iraf

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    work = Path(sys.argv[1]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    iraf.chdir(str(work))
    out = {}
    for name, params in CASES.items():
        defaults = dict(constant=0.0, start_level=0.0, end_level=0.0, start_wave=1.0,
                        end_wave=2.0, temperature=1.0)
        iraf.mkspec(name, **{**defaults, **params})
        listed = io.StringIO()
        iraf.listpixels(name, formats="", Stdout=listed)
        title = io.StringIO()
        iraf.hselect(name, "title", "yes", Stdout=title)
        rows = [line.split() for line in listed.getvalue().splitlines() if line.strip()]
        out[name] = {"pixels": [r[-1] for r in rows], "title": title.getvalue().strip()}
    (Path(sys.argv[2]) / "mk.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
