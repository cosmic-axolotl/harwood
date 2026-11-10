"""The cases of IRAF's median gabarito: input, xwindow, ywindow, and the hidden
parameters when not their defaults (zloreject, zhireject None for INDEF)."""

WINDOWS = [(3, 3), (7, 1), (1, 7), (5, 3), (3, 5), (4, 4), (2, 6), (1, 1), (9, 9),
           (11, 3), (25, 1)]
CASES = {}
for image in ("a", "s", "narrow", "tiny", "one", "col", "nan", "d"):
    for xw, yw in WINDOWS:
        CASES[f"{image}-{xw}x{yw}"] = dict(input=image, xwindow=xw, ywindow=yw)
for image in ("a", "narrow", "one", "s"):
    for boundary, constant in (("constant", 0.0), ("constant", 37.5), ("reflect", 0.0),
                               ("wrap", 0.0)):
        for xw, yw in ((3, 3), (7, 1), (5, 5), (9, 3)):
            CASES[f"{image}-{boundary}{constant:g}-{xw}x{yw}"] = dict(
                input=image, xwindow=xw, ywindow=yw, boundary=boundary,
                constant=constant)
for image in ("a", "s", "one"):
    for zlo, zhi in ((0.0, None), (None, 30.0), (0.0, 30.0), (10.5, 12.5),
                     (-1000.0, -900.0), (900.0, 1000.0), (20.0, 15.0)):
        for xw, yw in ((3, 3), (7, 1), (5, 3)):
            CASES[f"{image}-z{zlo}-{zhi}-{xw}x{yw}"] = dict(
                input=image, xwindow=xw, ywindow=yw, zloreject=zlo, zhireject=zhi)
CASES["verbose"] = dict(input="a", xwindow=4, ywindow=3, verbose=True)
CASES["verbose-z"] = dict(input="a", xwindow=3, ywindow=3, zloreject=0.0, verbose=True)
CASES["verbose-zhi"] = dict(input="a", xwindow=3, ywindow=3, zhireject=30.0, verbose=True)
CASES["section"] = dict(input="a[3:18,2:15]", xwindow=5, ywindow=3)
CASES["section-step"] = dict(input="a[*:2,*:2]", xwindow=3, ywindow=3)
for section in ("a[3:18,2:15]", "a[2:20:3,3:15:2]", "a[18:3,*]", "a[1:6,1:5]",
                "a[20:23,13:17]", "a[*,-*]", "s[4:12:2,2:10:3]"):
    for boundary, constant in (("nearest", 0.0), ("constant", 37.5), ("reflect", 0.0),
                               ("wrap", 0.0)):
        CASES[f"{section}-{boundary}"] = dict(input=section, xwindow=5, ywindow=3,
                                              boundary=boundary, constant=constant)
CASES["narrow-rows"] = dict(input="narrow", xwindow=3, ywindow=9)
CASES["tiny-5x5"] = dict(input="tiny", xwindow=5, ywindow=5)
CASES["lists-mismatch"] = dict(input="a.fits,s.fits", xwindow=3, ywindow=3)
