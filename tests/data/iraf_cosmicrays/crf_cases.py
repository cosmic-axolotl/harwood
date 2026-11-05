"""The cases of IRAF's cosmicrays gabarito."""

QUIT = "1 1 1 q\\n"

CASES = {
    "default": dict(),
    "one-pass": dict(npasses=1),
    "two-passes": dict(npasses=2),
    "low": dict(threshold=10.0),
    "high": dict(threshold=200.0),
    "ratio-low": dict(fluxratio=0.5),
    "ratio-high": dict(fluxratio=20.0),
    "window7": dict(window="7"),
    "window7-one": dict(window="7", npasses=1),
    "window7-low": dict(window="7", threshold=10.0, fluxratio=10.0),
    "short": dict(input="s.fits"),
    "short-high": dict(input="s.fits", fluxratio=20.0, threshold=10.0),
    "short7": dict(input="s.fits", window="7", fluxratio=20.0),
    "train": dict(train="yes", objects="train.cur", interactive="yes", savefile="saved.txt",
                  cursor="keys.cur", keys=QUIT),
    "train-graph": dict(train="yes", objects="traing.cur", interactive="yes",
                        savefile="saved.txt", cursor="keys.cur",
                        keys="1000 50 1 t\n" + QUIT + QUIT + QUIT),
    "train-more": dict(train="yes", objects="trainm.cur", interactive="yes",
                       savefile="saved.txt", cursor="keys.cur", keys=QUIT + QUIT + QUIT),
}

#: Sessions with the graph, cursor lines in flux and ratio
_POINTS = "".join(f"{x} {y} 1 \\040\n" for x in (11, 40, 150, 500, 1500, 4000, 20000)
                  for y in (0.1, 1, 3, 8, 12, 16, 19, 25))
SESSIONS = {
    "threshold": "1000 5 1 t\n" + QUIT,
    "threshold-all": "1000 50 1 t\n" + QUIT,
    "threshold-low": "1000 0.05 1 t\n" + QUIT,
    "delete": "1500 17 1 d\n1500 17 1 d\n1 1 1 q\n",
    "undelete": "60 0.6 1 u\n" + QUIT,
    "delete-undo": "1500 17 1 d\n1500 17 1 u\n60 0.6 1 u\n60 0.6 1 d\n" + QUIT,
    "region": "100 2 1 e\n100000 20 1 e\n" + QUIT,
    "region-un": "1 0.001 1 v\n100000 20 1 v\n" + QUIT,
    "region-flip": "100000 20 1 e\n100 2 1 e\n" + QUIT,
    "show": "1000 2 1 a\n1000 2 1 a\n" + QUIT,
    "show-train": "1500 17 1 d\n1000 2 1 a\n60 0.6 1 u\n1000 50 1 t\n" + QUIT,
    "points": _POINTS + QUIT,
    "surface": "1500 17 1 s\n1 1 1 d\n1500 17 1 \\040\n" + QUIT,
    "redraw": "1500 17 1 r\n1500 17 1 z\n1500 17 1 \\040\n" + QUIT,
    "window": "1 1 1 w\n1000 8 1 z\n" + _POINTS + QUIT,
    "window-e": "1 1 1 w\n100 5 1 e\n3000 19 1 e\n" + _POINTS + QUIT,
    "window-auto": "1 1 1 w\n1 1 1 a\n" + _POINTS + QUIT,
    "window-flip": "1 1 1 w\n1 1 1 f\n1 1 1 w\n1 1 1 g\n" + _POINTS + QUIT,
    "window-shift": "1 1 1 w\n1 1 1 r\n1 1 1 w\n1 1 1 u\n" + _POINTS + QUIT,
    "unknown": "1000 5 1 x\n1000 5 1 :nothing\n1000 5 1 \\040\n" + QUIT,
    # gtools: on a flipped axis j sets the edge on the left of the screen, and a region
    # sorted in flux and compared in NDC holds nothing
    "window-flip-edge": "1 1 1 w\n1 1 1 f\n1 1 1 w\n1000 1 1 j\n" + _POINTS + QUIT,
    "window-flip-region": "1 1 1 w\n1 1 1 f\n100 2 1 e\n100000 20 1 e\n" + QUIT,
    "colon-window": "1 1 1 :/xwindow 100 5000\n1 1 1 :/ywindow 0 20\n1 1 1 r\n"
                    + _POINTS + QUIT,
    "colon-transpose": "1 1 1 :/transpose\n1 1 1 r\n" + _POINTS + QUIT,
    "colon-linear": "1 1 1 :/xtransform linear\n1 1 1 r\n" + _POINTS + QUIT,
    "colon-clip": "1 1 1 :/clip 0.6 0.6\n1 1 1 r\n" + _POINTS + QUIT,
    "eof": "1000 5 1 t\n",
    "eof-region": "100 2 1 e\n",
}
