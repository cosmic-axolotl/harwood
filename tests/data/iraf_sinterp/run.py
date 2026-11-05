"""IRAF 2.18's sinterp, for tests/test_sinterp_iraf.py: what it prints, and the
spectrum --make-image writes, in si.json.

    HOME=<scratch with .iraf/uparm> python3 run.py <scratch> <here>     (with PyRAF)
"""

import io
import json
import sys
from pathlib import Path

from pyraf import iraf
from si_cases import CASES, POINTS, TABLE

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    work = Path(sys.argv[1]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    iraf.chdir(str(work))
    (work / "table.dat").write_text(TABLE, encoding="utf-8")
    (work / "points.dat").write_text(POINTS, encoding="utf-8")
    out = {}
    for name, params in CASES.items():
        base = dict(x1=0.0, x2=0.0, dx=0.0, npts=0, curve_gen="no", make_image="no",
                    order=5, image=name)
        params = {**base, **{k: ("yes" if v else "no") if isinstance(v, bool) else v
                             for k, v in params.items()}}
        text = io.StringIO()
        iraf.sinterp("table.dat", "points.dat", Stdout=text, Stderr=text, **params)
        case = {"said": text.getvalue().splitlines()}
        if params["make_image"] == "yes":
            listed = io.StringIO()
            iraf.listpixels(name, wcs="world", formats="%.10g", Stdout=listed)
            rows = [r.split() for r in listed.getvalue().splitlines() if r.strip()]
            case["world"] = [float(r[0]) for r in rows]
            case["pixels"] = [float(r[1]) for r in rows]
        out[name] = case
    (Path(sys.argv[2]) / "si.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
