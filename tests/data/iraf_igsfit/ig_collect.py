"""What IRAF left in each case's folder, as igsfit.json: what it printed (the banner's
date dropped), the error stream, the fc records (their dates dropped), the deletion
lists (every *.db there), the logfile, and the frames of the graph and of the plot log as
../iraf_photcal/decode_gki.py reads them -- per frame the first WCS's window and
viewport, the texts, and the marks (kind, x, y in GKI)."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "iraf_photcal"))
from decode_gki import everything  # noqa: E402

BANNER = re.compile(r"^(Community IRAF|harwood) .*$", re.M)


def frames(path):
    if not path.exists():
        return []
    out = []
    for frame in everything(path):
        out.append({"window": frame["windows"][0] if frame["windows"] else None,
                    "text": frame["text"],
                    "marks": [m[:3] for m in frame["marks"]]})
    return out


def text(path, drop=None):
    if not path.exists():
        return None
    body = path.read_text(encoding="utf-8")
    return drop.sub("", body) if drop else body


results = Path(sys.argv[1])
out = {}
for work in sorted(p for p in results.iterdir() if p.is_dir()):
    out[work.name] = {
        "said": text(work / "said.txt", BANNER),
        "err": text(work / "err.txt"),
        "fc": {p.name: text(p, re.compile(r"^# .*$", re.M))
               for p in sorted((work / "database").glob("fc*"))},
        "deletions": {p.name: text(p) for p in sorted(work.glob("*.db"))},
        "logfile": text(work / "logfile", BANNER),
        "frames": frames(work / "case.vdm"),
        "plotfile": frames(work / "plotfile"),
    }
Path(__file__).with_name("igsfit.json").write_text(json.dumps(out, indent=0) + "\n",
                                                    encoding="utf-8")
