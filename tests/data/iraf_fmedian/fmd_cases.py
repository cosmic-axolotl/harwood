"""The cases of IRAF's fmedian gabarito: input, xwindow, ywindow, and the hidden
parameters when not their defaults (None for INDEF)."""

CASES = {}
for image in ("a", "s", "d", "narrow", "tiny", "one", "col", "nan", "lim"):
    for xw, yw in ((3, 3), (7, 1), (1, 7), (5, 3), (4, 4), (9, 9), (1, 1)):
        CASES[f"{image}-{xw}x{yw}"] = dict(input=image, xwindow=xw, ywindow=yw)
for image in ("a", "s", "one"):
    for extra, params in (("h255", dict(hmin=0, hmax=255)),
                          ("h15", dict(hmin=0, hmax=15)),
                          ("hneg", dict(hmin=-50, hmax=50)),
                          ("z", dict(zmin=-100.0, zmax=200.0)),
                          ("zh", dict(zmin=0.0, zmax=40.0, hmin=0, hmax=400)),
                          ("unmap", dict(unmap=False)),
                          ("unmap-h255", dict(unmap=False, hmin=0, hmax=255)),
                          ("identity", dict(zmin=-32768.0, zmax=32767.0)),
                          ("rej", dict(zloreject=0.0, zhireject=30.0)),
                          ("rejlo", dict(zloreject=10.5)),
                          ("rejall", dict(zloreject=900.0, zhireject=1000.0)),
                          ("rejrev", dict(zloreject=20.0, zhireject=15.0))):
        for xw, yw in ((3, 3), (7, 1), (5, 3)):
            CASES[f"{image}-{extra}-{xw}x{yw}"] = dict(input=image, xwindow=xw,
                                                       ywindow=yw, **params)
for image in ("a", "narrow", "s", "one"):
    for boundary, constant in (("constant", 0.0), ("constant", 37.5), ("reflect", 0.0),
                               ("wrap", 0.0)):
        for xw, yw in ((3, 3), (7, 1), (5, 5)):
            CASES[f"{image}-{boundary}{constant:g}-{xw}x{yw}"] = dict(
                input=image, xwindow=xw, ywindow=yw, boundary=boundary,
                constant=constant)
CASES["lim-constant"] = dict(input="lim", xwindow=3, ywindow=3, boundary="constant",
                             constant=1000.0)
for section in ("a[3:18,2:15]", "a[2:20:3,3:15:2]", "a[18:3,*]"):
    CASES[f"{section}"] = dict(input=section, xwindow=5, ywindow=3)
CASES["verbose"] = dict(input="a", xwindow=4, ywindow=3, verbose=True)
CASES["verbose-rej"] = dict(input="a", xwindow=3, ywindow=3, zloreject=0.0,
                            verbose=True)
CASES["verbose-identity"] = dict(input="s", xwindow=3, ywindow=3, zmin=-32768.0,
                                 zmax=32767.0, verbose=True)
CASES["verbose-z"] = dict(input="a", xwindow=3, ywindow=3, zmin=-100.0, verbose=True)
