"""IRAF 2.18's scoords, for tests/test_scoords_iraf.py: what it says and the world
coordinates it leaves, in sc.json.

    HOME=<scratch with .iraf/uparm> python3 run.py <scratch> <here>     (with PyRAF)
"""

import io
import json
import sys
from pathlib import Path

from pyraf import iraf
from sc_cases import CASES, FILES, N

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    work = Path(sys.argv[1]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    iraf.chdir(str(work))
    for name, text in FILES.items():
        (work / name).write_text(text, encoding="utf-8")
    out = {}
    for case, (image, coords) in CASES.items():
        iraf.mkspec(image, image, N, 1, 1, constant=5.0)
        text = io.StringIO()
        iraf.scoords(image, coords, label="Wavelength", units="Angstroms", verbose="yes",
                     Stdout=text, Stderr=text)
        listed = io.StringIO()
        iraf.listpixels(image, wcs="world", formats="%.15g", Stdout=listed)
        rows = [r.split() for r in listed.getvalue().splitlines() if r.strip()]
        out[case] = {"said": text.getvalue().splitlines(),
                     "world": [float(r[0]) for r in rows]}
    (Path(sys.argv[2]) / "sc.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
