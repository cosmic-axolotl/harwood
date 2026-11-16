"""What IRAF's illumination by hand left in each case's folder, as illum_bins.json:
what it said and warned of, the bits of the pixels of each illumination written (o*),
and the frames of the bins' graph (those titled 'Set illumination bins') as
../iraf_photcal/decode_gki.py reads them: the window and viewport, the texts, the
lines -- the spectrum and the bins' marks, those erased too."""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "iraf_photcal"))
from decode_gki import everything  # noqa: E402


def frames(path):
    if not path.exists():
        return []
    return [{"window": frame["windows"][0], "text": frame["text"],
             "lines": [p for _, p in frame["lines"]]}
            for frame in everything(path) if "Set illumination bins" in frame["text"]]


results = Path(sys.argv[1])
out = {}
for work in sorted(p for p in results.iterdir() if p.is_dir()):
    images = {}
    for image in sorted(work.glob("o*.fits")):
        data = np.asarray(fits.getdata(image), np.float32)
        images[image.stem] = {"shape": list(data.shape),
                              "bits": data.view(np.uint32).ravel().tolist()}
    out[work.name] = {"said": (work / "out.txt").read_text(encoding="utf-8"),
                      "warned": (work / "err.txt").read_text(encoding="utf-8"),
                      "images": images, "frames": frames(work / "g.vdm")}
Path(__file__).with_name("illum_bins.json").write_text(
    json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
