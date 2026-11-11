"""What IRAF's splot left in each case, as splot.json: what it printed on both
streams, the save_file splot.log (its date lines are compared without the date),
and from its metacode (decoded by ../iraf_photcal/decode_gki.py) each frame's
window, viewport and texts."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from sp_cases import CASES

results = Path(sys.argv[1])
out = {}
for case in CASES:
    work = results / case
    entry = {"out": (work / "out.txt").read_text(encoding="utf-8"),
             "err": (work / "err.txt").read_text(encoding="utf-8")}
    log = work / "splot.log"
    entry["log"] = log.read_text(encoding="utf-8").splitlines() if log.exists() else None
    entry["frames"] = []
    if (work / "vdm").exists():
        shutil.copy(work / "vdm", work / "graph.vdm")
        decoder = Path(__file__).resolve().parent.parent / "iraf_photcal" / "decode_gki.py"
        subprocess.run([sys.executable, str(decoder), str(work / "graph.vdm"),
                        "--everything"], check=True)
        frames = json.loads((work / "graph.json").read_text(encoding="utf-8"))
        entry["frames"] = [{"window": f["windows"][0][:4], "viewport": f["windows"][0][4:],
                            "text": f["text"]} for f in frames]
    out[case] = entry
Path(__file__).with_name("splot.json").write_text(json.dumps(out, indent=0),
                                                   encoding="utf-8")
