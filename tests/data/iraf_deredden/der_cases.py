"""The cases of IRAF's deredden gabarito: each a chain of runs, the first on its
input, each next on the output before it, o1, o2..."""

CASES = {
    "ebv": ("in", [dict(value=0.3)]),
    "av": ("in", [dict(value=1.0, type="A(V)", R=5.0)]),
    "c": ("in", [dict(value=0.5, type="c")]),
    "apertures": ("in", [dict(value=0.3, apertures="1,3")]),
    "again": ("in", [dict(value=0.3), dict(value=0.2)]),
    "override": ("in", [dict(value=0.3), dict(value=0.5, R=4.0, override=True)]),
    "samer": ("in", [dict(value=0.3), dict(value=0.5, override=True)]),
    "keepold": ("in", [dict(value=0.3), dict(value=0.5, override=True, uncorrect=False)]),
    "far": ("far", [dict(value=0.3)]),
}
