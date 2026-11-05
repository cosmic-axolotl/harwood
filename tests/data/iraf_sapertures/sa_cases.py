"""The cases of IRAF's sapertures gabarito, on ../iraf_slist's `in` (apertures 1 and 3
linear, 2 logarithmic, beam 5 on aperture 3, titles first, second and none): each the
parameters of one run, all verbose."""

TABLE = """# ap beam dtype w1 dw z aplow aphigh title
1 11 0 4100. 1.5 0. 6. 15. table one
2 -4 the second, short
3 INDEF INDEF INDEF INDEF 0.002 INDEF INDEF short of nine
"""
CASES = {
    "nothing": dict(),
    "beam": dict(apertures="1,3", beam=7),
    "linear": dict(apertures="1", w1=5000.0, dw=2.0),
    "tolog": dict(apertures="3", dtype=1, w1=3.6, dw=0.0001),
    "redshift": dict(z=0.01),
    "limits": dict(aplow=-2.0, aphigh=4.0),
    "title": dict(apertures="2", title="a new title"),
    "pixels": dict(dtype=-1),
    "reset": dict(wcsreset=True, beam=9),
    "table": dict(apidtable="apid.dat"),
}
