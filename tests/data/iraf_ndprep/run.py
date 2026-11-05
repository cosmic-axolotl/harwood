"""IRAF 2.18's ndprep, for tests/test_ndprep_iraf.py: each case's image, its pixels
and its dispersion axis's world coordinates.

    HOME=<scratch with .iraf/uparm> python3 run.py <scratch>       (with PyRAF)
    python nd_collect.py <scratch>                                  (harwood's)
"""

import io
import sys
from pathlib import Path

from nd_cases import CASES
from pyraf import iraf

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    work = Path(sys.argv[1]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    iraf.chdir(str(work))
    for name, params in CASES.items():
        params = {k: ("yes" if v else "no") if isinstance(v, bool) else v
                  for k, v in params.items()}
        iraf.ndprep(output=name, **params)
        listed = io.StringIO()
        section = name if params.get("nspace", 0) == 0 else (
            f"{name}[*,1]" if params.get("dispaxis") == 1 else f"{name}[1,*]")
        iraf.listpixels(section, wcs="world", formats="%.12g", Stdout=listed)
        (work / f"{name}.world").write_text(listed.getvalue(), encoding="utf-8")
        print(name, flush=True)
