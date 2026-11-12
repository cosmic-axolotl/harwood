"""What IRAF's rvcorrect did in each case, as rvcorrect.json: both streams, the
parameters it recorded (repr of the double), and the cards imupdate wrote."""

import json
import sys
from pathlib import Path

from astropy.io import fits
from rv_cases import CASES

results = Path(sys.argv[1])
gold = {}
for case in CASES:
    work = results / case
    entry = {"out": (work / "out.txt").read_text(encoding="utf-8"),
             "err": (work / "err.txt").read_text(encoding="utf-8", errors="replace"),
             "params": json.loads((work / "params.json").read_text(encoding="utf-8"))
             if (work / "params.json").exists() else {}}
    cards = {}
    for image in sorted(work.glob("*.fits")):
        h = fits.getheader(image)
        found = {k: h.cards[k].image for k in ("HJD", "VHELIO", "VLSR", "VSUN") if k in h}
        if found:
            cards[image.name] = found
    entry["cards"] = cards
    gold[case] = entry
Path(__file__).with_name("rvcorrect.json").write_text(json.dumps(gold, indent=0),
                                                       encoding="utf-8")
