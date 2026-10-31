"""The cursor files of the interactive mkapfile cases, written beside their
photometry: python make_cursors.py, from a directory holding all.lis and the
.mag.1 files of ../apfile. A cursor on a point is the point's own coordinates,
worked out by harwood's non-interactive arithmetic, already held to IRAF's."""
import sys
from pathlib import Path

from harwood.core.photcal import apfile

P0 = [1.2, 0.1, 0.5, 0.9, 0.0]
files = Path("all.lis").read_text().split()
images = apfile.read_photfiles(files, 20, 0.10)
fit = apfile.fit_model(images, P0, 3)
first = images[0]
curve = apfile.evaluate(first, fit.params, 20)
star = first.stars[2]
j = min(10, star.nap - 1)               # a point on the third star of the first image
rbar, mag = float(curve.rbar[j]), float(star.mag[j])
res = mag - float(curve.adopt[j])
xs, ys, mag1 = float(star.x), float(star.y), float(star.mag[0])
total = 0.0                             # the same point on the cumulative graph
for i in range(19, j - 1, -1):
    total += float(star.mag[i]) if i + 1 <= star.nap else float(curve.adopt[i])
rcum = float(star.rap[j - 1])

ALL = ["m", "r", "b", "x", "y", "a", "g", "w"]
CASES = {
    "keys": [(0, 0, k) for k in ALL]
            + [(0, 0, f": {c}") for c in (
                "show parameters", "show model", "show seeing", "show", "image",
                "image stan0051cr.fits", "image nosuch.fits", "nparams", "swings",
                "pwings", "pgauss", "rgescale", "xwings", "smallap", "largeap",
                "bogus")],
    "noop": [(0, 0, "f")],
    "twice": [(0, 0, "f"), (0, 0, "f"), (0, 0, "w")],
    "del": [(rbar, mag, "d"), (0, 0, "w"), (0, 0, "f"), (0, 0, "w")],
    "stale": [(rbar, mag, "d"), (0, 0, "w"), (0, 0, ": show model"), (0, 0, ": show seeing")],
    "undel": [(rbar, mag, "d"), (rbar, mag, "u"), (0, 0, "f")],
    "delres": [(0, 0, "r"), (rbar, res, "d"), (0, 0, "f")],
    "delb": [(0, 0, "b"), (mag1, res, "d"), (0, 0, "f")],
    "delx": [(0, 0, "x"), (xs, res, "d"), (0, 0, "f")],
    "dely": [(0, 0, "y"), (ys, res, "d"), (0, 0, "f")],
    "near": [(rbar, mag, "c"), (0, 0, "r"), (rbar, res, "c"), (0, 0, "b"), (mag1, res, "c"),
             (0, 0, "x"), (xs, res, "c"), (0, 0, "y"), (ys, res, "c")],
    "cuma": [(0, 0, "a"), (rcum, total, "c"), (0, 0, "w")],
    "dela": [(0, 0, "a"), (rcum, total, "d"), (0, 0, "w"), (0, 0, "f"), (0, 0, "w")],
    "undela": [(0, 0, "a"), (rcum, total, "d"), (rcum, total, "u"), (0, 0, "f")],
    "n5": [(0, 0, ": nparams 5"), (0, 0, "f"), (0, 0, ": show model"), (0, 0, ": nparams")],
    "n1": [(0, 0, ": nparams 9"), (0, 0, ": nparams"), (0, 0, ": nparams 1"), (0, 0, "f")],
    "start": [(0, 0, f": {c}") for c in ("swings 1.5", "pwings 0.2", "pgauss 0.4",
                                         "rgescale 0.8", "xwings 0.1", "show parameters")]
             + [(0, 0, "f"), (0, 0, ": show model")],
    "clamp": [(0, 0, f": {c}") for c in ("swings 0.5", "swings", "pwings 2", "pwings",
                                         "pgauss -1", "pgauss", "rgescale -2", "rgescale",
                                         "smallap 99", "smallap", "largeap 0", "largeap")],
    "aps": [(0, 0, ": smallap 2"), (0, 0, ": largeap 15"), (0, 0, "w")],
    "walk": [(0, 0, "p")] + [(0, 0, "n")] * 12 + [(0, 0, "w")] + [(0, 0, "p")] * 12,
    "other": [(0, 0, ": image stan0058cr.fits"), (0, 0, "w"), (0, 0, "n"), (0, 0, "w"),
              (0, 0, "f"), (0, 0, "w")],
}
for name, lines in CASES.items():
    text = "".join(f"{x:.7g} {y:.7g} 1 {k}\n" for x, y, k in lines) + "0 0 1 q\n"
    Path(f"{name}.cur").write_text(text)
print(len(CASES), rbar, mag, res, xs, ys, mag1, file=sys.stderr)
