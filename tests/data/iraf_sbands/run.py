"""IRAF 2.18's sbands, for tests/test_sbands_iraf.py: what it wrote, in sb.json.

    python ../iraf_dopcor/dop_inputs.py <folder>                    (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <here>  (with PyRAF)
"""

import json
import sys
from pathlib import Path

from pyraf import iraf
from sb_cases import BANDS, CASES, FILTER

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    work = Path(sys.argv[1]).resolve()
    iraf.chdir(str(work))
    (work / "bands.dat").write_text(BANDS, encoding="utf-8")
    (work / "filt.dat").write_text(FILTER, encoding="utf-8")
    out = {}
    for name, params in CASES.items():
        params = {k: ("yes" if v else "no") if isinstance(v, bool) else v
                  for k, v in params.items()}
        target = work / f"{name}.out"
        target.unlink(missing_ok=True)
        iraf.sbands("in", str(target), "bands.dat", **params)
        out[name] = target.read_text(encoding="utf-8").splitlines()
    (Path(sys.argv[2]) / "sb.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
