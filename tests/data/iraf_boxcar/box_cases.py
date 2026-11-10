"""The cases of IRAF's boxcar gabarito: input, xwindow, ywindow, and boundary and
constant when not their defaults."""

CASES = {}
for image in ("a", "s", "d", "narrow", "tiny", "one", "col", "nan"):
    for xw, yw in ((3, 3), (7, 1), (1, 7), (5, 3), (4, 4), (2, 3), (3, 2), (1, 1), (9, 9),
                   (6, 1), (1, 6)):
        CASES[f"{image}-{xw}x{yw}"] = dict(input=image, xwindow=xw, ywindow=yw)
for image in ("a", "narrow", "s", "one"):
    for boundary, constant in (("constant", 0.0), ("constant", 37.5), ("reflect", 0.0),
                               ("wrap", 0.0)):
        for xw, yw in ((3, 3), (7, 1), (5, 5), (4, 2), (9, 3)):
            CASES[f"{image}-{boundary}{constant:g}-{xw}x{yw}"] = dict(
                input=image, xwindow=xw, ywindow=yw, boundary=boundary,
                constant=constant)
for section in ("a[3:18,2:15]", "a[2:20:3,3:15:2]", "a[18:3,*]", "a[*,-*]",
                "s[4:12:2,2:10:3]"):
    for boundary in ("nearest", "reflect"):
        CASES[f"{section}-{boundary}"] = dict(input=section, xwindow=5, ywindow=3,
                                              boundary=boundary)
CASES["cube"] = dict(input="cube", xwindow=3, ywindow=3)
