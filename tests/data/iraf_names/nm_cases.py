"""The cases of IRAF's names gabarito, and the images `check` finds (sp.0002, sp.0005)."""

CASES = {
    "simple": dict(input="sp", records="1-3"),
    "list": dict(input="sp", records="5,1-3,2"),
    "step": dict(input="sp,arc", records="10-20x4"),
    "reverse": dict(input="sp", records="7-4"),
    "zero": dict(input="sp", records="0-3"),
    "afterzero": dict(input="sp", records="2,0,5"),
    "open": dict(input="sp", records="998-"),
    "extension": dict(input="sp.imh,sp.pl,sp.fits,sp[1:10]", records="1"),
    "append": dict(input="sp", records="1,3", append="[1:500]"),
    "none": dict(input="sp,arc", records=""),
    "check": dict(input="sp", records="1-6", check=True),
    "toomany": dict(input="sp", records="1-2000"),
    "bad": dict(input="sp", records="a-3"),
}
