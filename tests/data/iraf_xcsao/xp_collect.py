"""What RVSAO's xcsao with its graph left in each case's folder, as xcplot.json: what
it said and warned of, each spectrum it rewrote (its cards and pixels) and every
frame of the graph from the metacode, as ../iraf_emsao/ep_collect.py reads them.

    python xp_collect.py <results> <fresh spectra>"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "iraf_emsao"))
from ep_collect import frames, spectra  # noqa: E402

if __name__ == "__main__":
    results, fresh = Path(sys.argv[1]), Path(sys.argv[2])
    out = {}
    for work in sorted(p for p in results.iterdir() if p.is_dir()):
        out[work.name] = {"said": (work / "out.txt").read_text(encoding="utf-8"),
                          "warned": (work / "err.txt").read_text(encoding="utf-8"),
                          "spectra": spectra(work, fresh),
                          "frames": frames(work / "g.vdm")}
    Path(__file__).with_name("xcplot.json").write_text(
        json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
