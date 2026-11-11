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
    inputs = {"one.fits", "slit.fits", "ms.fits", "graph.vdm"}
    entry["written"] = {}
    for path in sorted(work.glob("*.fits")):
        if path.name in inputs and path.stat().st_mtime <= (work / "cursor").stat().st_mtime:
            continue
        from astropy.io import fits
        import base64
        import numpy as np
        with fits.open(path) as hdul:
            data = np.asarray(hdul[0].data)
            big = data.astype(data.dtype.newbyteorder(">"))
            h = hdul[0].header
            cards = {k: h[k] for k in h if k.startswith(("CRVAL", "CRPIX", "CDELT", "CD1_",
                                                          "CD2_", "DC-FLAG", "WAT", "LTV",
                                                          "LTM", "EX-FLAG", "CA-FLAG"))}
        entry["written"][path.name] = {"shape": list(data.shape), "dtype": big.dtype.str,
                                       "data": base64.b64encode(big.tobytes()).decode(),
                                       "cards": cards}
    entry["frames"] = []
    if (work / "vdm").exists():
        shutil.copy(work / "vdm", work / "graph.vdm")
        decoder = Path(__file__).resolve().parent.parent / "iraf_photcal" / "decode_gki.py"
        subprocess.run([sys.executable, str(decoder), str(work / "graph.vdm"),
                        "--everything"], check=True)
        frames = json.loads((work / "graph.json").read_text(encoding="utf-8"))
        entry["frames"] = [{"window": f["windows"][0][:4], "viewport": f["windows"][0][4:],
                            "text": f["text"]} for f in frames]
        entry["texts"] = sorted({s for f in frames for s in f["text"]})
        entry["lines"] = [len(f["lines"]) for f in frames]
    out[case] = entry
Path(__file__).with_name("splot.json").write_text(json.dumps(out, indent=0),
                                                   encoding="utf-8")
