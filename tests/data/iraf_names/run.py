"""IRAF 2.18's names, for tests/test_names_iraf.py: what it prints, in nm.json.

    HOME=<scratch with .iraf/uparm> python3 run.py <scratch> <here>     (with PyRAF)
"""

import io
import json
import sys
from pathlib import Path

from nm_cases import CASES
from pyraf import iraf

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    work = Path(sys.argv[1]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    iraf.chdir(str(work))
    iraf.artdata(_doprint=0)
    for number in (2, 5):
        iraf.mkpattern(f"sp.{number:04d}", ncols=10, nlines=1, ndim=1)
    out = {}
    for name, params in CASES.items():
        params = {k: ("yes" if v else "no") if isinstance(v, bool) else v
                  for k, v in params.items()}
        text = io.StringIO()
        try:
            iraf.names(Stdout=text, Stderr=text, **params)
        except Exception as error:  # noqa: BLE001
            text.write("ERROR " + str(error).splitlines()[-1] + "\n")
        out[name] = text.getvalue().splitlines()
    (Path(sys.argv[2]) / "nm.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
