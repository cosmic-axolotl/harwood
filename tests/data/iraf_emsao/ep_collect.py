"""What RVSAO's emsao with its graph left in each case's folder, as emplot.json: what it
said and warned of, each spectrum it rewrote (the header's cards as written, INDEF
and all, blank ones left out, and the pixels' bits), and every frame of the graph
from the metacode (g.vdm), in the order drawn:

    windows  the WCS used at the end of the frame: x1 x2 y1 y2 and the viewport
    lines    [style, x, y, x, y, ...] in GKI units (0..32767)
    texts    [x, y, text, up, size, path, hjust, vjust, font, quality, color]

A frame is a clear of the screen, or a workstation opened anew (gopen after gclose
with APPEND) with something drawn on it.

    python ep_collect.py <results> <fresh spectra>"""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

OPEN, CLEAR, LINE, TEXT, PLSET, TXSET, WCS = 1, 6, 9, 11, 15, 17, 26


def frames(path: Path) -> list:
    if not path.exists():
        return []
    words = np.fromfile(path, dtype="<i2")
    out, frame, style, txset, windows = [], None, 1, [], []
    i = 0
    while i < len(words):
        opcode, length = int(words[i + 1]), int(words[i + 2])
        data = words[i + 3:i + length]
        i += length
        if opcode in (OPEN, CLEAR):
            if frame is None or frame["lines"] or frame["texts"]:
                frame = {"windows": windows, "lines": [], "texts": []}
                out.append(frame)
            style = 1
        elif opcode == WCS:
            values = np.frombuffer(data[1:].tobytes(), dtype="<f4")[::2]
            windows = [[round(float(v), 7) for v in values[k:k + 8]]
                       for k in range(0, len(values), 12)
                       if values[k] != values[k + 1] and list(values[k:k + 8]) !=
                       [0, 1, 0, 1, 0, 1, 0, 1]]
            if frame is not None:
                frame["windows"] = windows
        elif frame is None:
            continue
        elif opcode == PLSET:
            style = int(data[0])
        elif opcode == TXSET:
            txset = [int(v) for v in data[:9]]
        elif opcode == TEXT:
            text = "".join(chr(c) for c in data[3:3 + int(data[2])])
            frame["texts"].append([int(data[0]), int(data[1]), text, *txset])
        elif opcode == LINE:
            points = data[1:1 + 2 * int(data[0])].astype(int) & 0xffff
            frame["lines"].append([style, *points.tolist()])
    return [f for f in out if f["lines"] or f["texts"]]


def spectra(work: Path, fresh: Path) -> dict:
    out = {}
    for image in sorted(work.glob("*.fits")):
        before = fresh / image.name
        if before.exists() and before.read_bytes() == image.read_bytes():
            continue
        raw = image.read_bytes()
        cards = []
        for k in range(0, len(raw), 80):
            card = raw[k:k + 80].decode("ascii")
            if card.startswith("END "):
                break
            if card.strip():
                cards.append(card)
        with fits.open(image, do_not_scale_image_data=True) as hdul:
            data = np.asarray(hdul[0].data, np.float32)
        out[image.name] = {"cards": cards,
                           "shape": list(data.shape),
                           "bits": data.view(np.uint32).ravel().tolist()}
    return out


if __name__ == "__main__":
    results, fresh = Path(sys.argv[1]), Path(sys.argv[2])
    out = {}
    for work in sorted(p for p in results.iterdir() if p.is_dir()):
        out[work.name] = {"said": (work / "out.txt").read_text(encoding="utf-8"),
                          "warned": (work / "err.txt").read_text(encoding="utf-8"),
                          "spectra": spectra(work, fresh),
                          "frames": frames(work / "g.vdm")}
    Path(__file__).with_name("emplot.json").write_text(
        json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
