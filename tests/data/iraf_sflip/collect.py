"""What IRAF's sflip left in each case's folder, as sflip.json: for each line, its
pixels, and their world and physical coordinates."""

import json
import sys
from pathlib import Path

results = Path(sys.argv[1])
out = {}
for work in sorted(results.iterdir()):
    case = {"said": (work / "said.txt").read_text(encoding="utf-8").splitlines()}
    for line in (1, 2, 3):
        rows = {wcs: [[float(v) for v in row.split()]
                      for row in (work / f"{wcs}{line}.txt").read_text(
                          encoding="utf-8").splitlines() if row.strip()]
                for wcs in ("world", "physical")}
        case[str(line)] = {"world": [r[0] for r in rows["world"]],
                           "physical": [r[0] for r in rows["physical"]],
                           "flux": [r[-1] for r in rows["world"]]}
    out[work.name] = case
Path(__file__).with_name("sflip.json").write_text(json.dumps(out), encoding="utf-8")
