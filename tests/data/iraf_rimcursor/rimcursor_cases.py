"""The inputs of rimcursor's gabarito."""

CURSOR = (
    "1.5 2.5\n"
    "10 20 101\n"
    "10.25 20.75 101 a\n"
    "33.3 44.4 101 \\040\n"
    "5 6 101 :command with words\n"
    "7.125 8.5 102 b\n"
    "9 9 abc\n"
    "1e2 2d1 101 x y   \n"
    "  3 4   101    z\n"
)
BAD = {"one": "1\n", "word": "abc 3\n", "blank": "1 2\n\n3 4\n", "comment": "1 2\n# c\n3 4\n"}

CASES = {
    "noimage": dict(image=""),
    "logical": dict(image="tan.fits"),
    "physical": dict(image="tan.fits", wcs="physical"),
    "world": dict(image="tan.fits", wcs="world"),
    "section-log": dict(image="tan.fits[11:40,6:20]"),
    "section-phy": dict(image="tan.fits[11:40,6:20]", wcs="physical"),
    "section-wor": dict(image="tan.fits[11:40,6:20]", wcs="world"),
    "step": dict(image="lin.fits[*:2,*:3]", wcs="physical"),
    "lin-world": dict(image="lin.fits", wcs="world"),
    "attrs": dict(image="attrs.fits", wcs="world"),
    "wx": dict(image="tan.fits", wcs="world", wxformat="%12.2H", wyformat="%12.1h"),
    "wf": dict(image="lin.fits", wcs="world", wxformat="%8.3f"),
    "world-17": dict(image="tan.fits", wcs="world", wxformat="%0.17g", wyformat="%0.17g"),
    "section-17": dict(image="tan.fits[11:40,6:20]", wcs="world", wxformat="%0.17g",
                       wyformat="%0.17g"),
    "lin-17": dict(image="lin.fits", wcs="world", wxformat="%0.17g", wyformat="%0.17g"),
    "step-17": dict(image="tan.fits[3:59:2,*:3]", wcs="world", wxformat="%0.17g",
                    wyformat="%0.17g"),
    "flip-17": dict(image="tan.fits[-*,*]", wcs="world", wxformat="%0.17g", wyformat="%0.17g"),
    "lin-step-17": dict(image="lin.fits[3:59:2,*:3]", wcs="world", wxformat="%0.17g",
                        wyformat="%0.17g"),
    "wy": dict(image="lin.fits", wyformat="%g"),
    "none": dict(image="nothing.fits"),
    "badwcs": dict(image="tan.fits", wcs="bogus"),
}
