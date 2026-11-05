"""IRAF 2.18's rspectext, for tests/test_rspectext_iraf.py: each case's spectrum (its
pixels, header and listpixels' world coordinates) in rs.json.

    HOME=<scratch with .iraf/uparm> python3 run.py <scratch> <here>     (with PyRAF)
"""

import io
import shutil
import json
import sys
from pathlib import Path

from pyraf import iraf
from rs_cases import CASES, text

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    work = Path(sys.argv[1]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    iraf.chdir(str(work))
    (work / "spec.txt").write_text(text(), encoding="utf-8")
    out = {}
    order = [name for name in CASES if name != "header"] + ["header", "exists"]
    for name in order:
        source, params = CASES.get(name, CASES["linear"])
        if name == "header":
            Path(work / "hdr.txt").unlink(missing_ok=True)
            iraf.wspectext("o_linear", "hdr.txt", header="yes", wformat="")
        params = {k: ("yes" if v else "no") if isinstance(v, bool) else v
                  for k, v in params.items()}
        target = "o_linear" if name == "exists" else f"o_{name}"
        said = io.StringIO()
        try:
            iraf.rspectext(source, target, Stdout=said, Stderr=said, **params)
        except Exception as error:  # noqa: BLE001
            said.write(str(error).splitlines()[-1] + "\n")
        case = {"said": said.getvalue().splitlines()}
        if name != "exists":
            listed = io.StringIO()
            iraf.listpixels(target, wcs="world", formats="%.12g", Stdout=listed)
            rows = [row.split() for row in listed.getvalue().splitlines() if row.strip()]
            case["world"] = [float(r[0]) for r in rows]
            case["flux"] = [float(r[1]) for r in rows]
            keys = io.StringIO()
            iraf.hselect(target, "title,DC-FLAG,CA-FLAG,EX-FLAG", "yes", Stdout=keys)
            case["keys"] = keys.getvalue().rstrip("\n").split("\t")
        out[name] = case
        print(name, flush=True)
    shutil.copy(work / "hdr.txt", Path(sys.argv[2]) / "hdr.txt")
    (Path(sys.argv[2]) / "rs.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
