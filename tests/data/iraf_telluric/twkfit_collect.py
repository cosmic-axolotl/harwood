"""What IRAF's telluric and skytweak by hand left in each case's folder, as twkfit.json:
what they said and warned of, the bits of the pixels of every image written (c*), and
the frames of the graphs as ../iraf_photcal/decode_gki.py reads them -- per frame the
window and viewport of each graph drawn, the texts, and the lines: the curves (more
than 30 points, or cut by the window's edge) and the sample's marks (2 points)."""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "iraf_photcal"))
from decode_gki import everything  # noqa: E402

UNITY = [0.0, 1.0, 0.0, 1.0]


def frames(path):
    if not path.exists():
        return []
    out = []
    for frame in everything(path):
        windows = [w for w in frame["windows"][:2] if w[4:] != UNITY]
        if not windows:
            continue
        out.append({"windows": windows, "logs": frame.get("logs", [])[:len(windows)],
                    "text": frame["text"], "lines": [p for _, p in frame["lines"]]})
    return out


results = Path(sys.argv[1])
out = {}
for work in sorted(p for p in results.iterdir() if p.is_dir()):
    images = {}
    for image in sorted(work.glob("c*.fits")):
        if image.read_bytes()[:6] != b"SIMPLE":
            images[image.stem] = None
            continue
        data = np.asarray(fits.getdata(image), np.float32)
        images[image.stem] = {"shape": list(data.shape),
                              "bits": data.view(np.uint32).ravel().tolist()}
    out[work.name] = {"said": (work / "out.txt").read_text(encoding="utf-8"),
                      "warned": (work / "err.txt").read_text(encoding="utf-8"),
                      "images": images, "frames": frames(work / "g.vdm")}
Path(__file__).with_name("twkfit.json").write_text(
    json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
