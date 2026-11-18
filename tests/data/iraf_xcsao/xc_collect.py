"""What RVSAO's xcsao did in each case, as xcsao.json: its standard output and
error, the xcsao.log it wrote (without its first line, the time stamp) and, when
save_vel asked, the spectra's headers after it (their cards, the blank ones out)."""

import json
import sys
from pathlib import Path

from xc_cases import CASES

results = Path(sys.argv[1])
target = Path(__file__).with_name("xcsao.json")
# cases not run this time keep what they had
gold = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
for case in CASES:
    work = results / case
    if not work.exists():
        continue
    log = work / "xcsao.log"
    gold[case] = {"out": (work / "out.txt").read_text(encoding="utf-8", errors="replace"),
                  "err": (work / "err.txt").read_text(encoding="utf-8", errors="replace"),
                  "log": log.read_text(encoding="utf-8") if log.exists() else None}
    if CASES[case][1].get("save_vel"):
        from astropy.io import fits

        gold[case]["head"] = {
            name: [c.image for c in fits.getheader(work / name).cards if c.image.strip()]
            for name in CASES[case][0].split(",") if (work / name).exists()}
Path(__file__).with_name("xcsao.json").write_text(json.dumps(gold, indent=0),
                                                   encoding="utf-8")
