"""IRAF's results, from run.py's folder, as iek.json: what it printed, the curves it drew in
each frame of the metacode, and the sha256 of the sections the t key wrote."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from iek_cases import CASES

sys.path.insert(0, str(Path(__file__).parent.parent / "iraf_photcal"))
from decode_gki import instructions  # noqa: E402

CLEAR, LINE = 6, 9


def frames(path):
    out, now = [], []
    for code, words in instructions(path):
        if code == CLEAR:
            if now:
                out.append(now)
            now = []
        elif code == LINE and len(words) > 40:
            now.append([int(word) for word in words[1:]])
    if now:
        out.append(now)
    return out


if __name__ == "__main__":
    results = Path(sys.argv[1])
    gold = {}
    for name in CASES:
        folder = results / name
        sections = {}
        for path in sorted(folder.glob("f.0*.fits")):
            data = np.asarray(fits.getdata(path)).astype(">f4")
            sha = hashlib.sha256(data.tobytes()).hexdigest()
            sections[path.name] = [list(data.shape), sha]
        gold[name] = {"text": (folder / "stdout.txt").read_text(), "sections": sections,
                      "log": (folder / "out.log").read_text()
                      if (folder / "out.log").exists() else "",
                      "frames": frames(folder / "metacode") if (folder / "metacode").exists()
                      and (folder / "metacode").stat().st_size else []}
    (Path(__file__).parent / "iek.json").write_text(json.dumps(gold, indent=0))
    print(len(gold))
