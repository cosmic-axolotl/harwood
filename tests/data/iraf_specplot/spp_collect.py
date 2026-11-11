"""What IRAF's specplot printed in each case, as specplot.json: both streams, the
logfile, and from the metacode each frame's window, viewport and texts."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from spp_cases import CASES

results = Path(sys.argv[1])
decoder = Path(__file__).resolve().parent.parent / "iraf_photcal" / "decode_gki.py"
out = {}
for case in CASES:
    work = results / case
    entry = {"out": (work / "out.txt").read_text(encoding="utf-8"),
             "err": (work / "err.txt").read_text(encoding="utf-8"), "frames": []}
    log = work / "sp.log"
    entry["log"] = log.read_text(encoding="utf-8").splitlines() if log.exists() else None
    if (work / "vdm").exists():
        shutil.copy(work / "vdm", work / "graph.vdm")
        subprocess.run([sys.executable, str(decoder), str(work / "graph.vdm"),
                        "--everything"], check=True)
        frames = json.loads((work / "graph.json").read_text(encoding="utf-8"))
        entry["frames"] = [{"window": f["windows"][0][:4], "viewport": f["windows"][0][4:],
                            "text": f["text"]} for f in frames]
    out[case] = entry
Path(__file__).with_name("specplot.json").write_text(json.dumps(out, indent=0),
                                                      encoding="utf-8")
