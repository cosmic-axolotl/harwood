"""The cases of IRAF's imreplace gabarito: image (with a section if any), value, and
lower, upper, radius, imaginary when not IRAF's defaults (None is INDEF)."""

INTS = ("s16", "u16", "i32", "b8", "scaled")
FLOATS = ("r32", "r64")
WINDOWS = {
    "all": {},
    "below0": {"upper": 0.0},
    "above0": {"lower": 0.0},
    "lowneg": {"lower": -201.0},
    "upneg": {"upper": 201.0},
    "fracneg": {"lower": -100.5, "upper": -20.5},
    "fracpos": {"lower": 20.3, "upper": 100.7},
    "fraclow": {"lower": -100.5},
    "fracup": {"upper": -20.5},
    "reversed": {"lower": 50.0, "upper": -50.0},
    "point": {"lower": 7.0, "upper": 7.0},
}

CASES = {}
for image in INTS + FLOATS + ("one", "cube"):
    for window, limits in WINDOWS.items():
        CASES[f"{image}-{window}"] = dict(image=image, value=0.0, **limits)
for image in INTS + FLOATS:
    CASES[f"{image}-value+"] = dict(image=image, value=2.7, upper=0.0)
    CASES[f"{image}-value-"] = dict(image=image, value=-2.7, lower=0.0)
CASES["s16-big"] = dict(image="s16", value=99999.0, lower=100.0)
CASES["s16-imaginary"] = dict(image="s16", value=3.0, lower=100.0, imaginary=5.0)
CASES["s16-section"] = dict(image="s16[3:6,2:5]", value=-1.0, upper=0.0)
CASES["r32-section"] = dict(image="r32[2:9:2,*]", value=1000.0, lower=-50.0, upper=50.0)
for image in ("s16", "r32", "r64", "u16", "one", "cube"):
    for radius in (0.5, 1.0, 1.5, 2.5, 6.0):
        CASES[f"{image}-r{radius}"] = dict(image=image, value=0.0, lower=150.0,
                                           radius=radius)
    CASES[f"{image}-r1.5-upper"] = dict(image=image, value=9.0, upper=-180.0, radius=1.5)
    CASES[f"{image}-r2-both"] = dict(image=image, value=-9.0, lower=-30.0, upper=30.0,
                                     radius=2.0)
    CASES[f"{image}-r2-all"] = dict(image=image, value=4.0, radius=2.0)
CASES["s16-r20"] = dict(image="s16", value=0.0, upper=-220.0, radius=20.0)
CASES["s16-r1-neg"] = dict(image="s16", value=1.0, radius=1.0, upper=-32760.0)
CASES["s16-r1.5-section"] = dict(image="s16[2:9,3:8]", value=0.0, lower=150.0,
                                 radius=1.5)
for image in ("odd32", "odd64"):
    CASES[f"{image}-none"] = dict(image=image, value=7.0, lower=1e30, upper=2e30)
    CASES[f"{image}-zero"] = dict(image=image, value=7.0, lower=0.0, upper=0.0)
    CASES[f"{image}-big"] = dict(image=image, value=7.0, lower=1e30)
    CASES[f"{image}-tiny"] = dict(image=image, value=7.0, lower=-1e-30, upper=1e-30)
    CASES[f"{image}-r1"] = dict(image=image, value=7.0, lower=4.5, radius=1.0)
    CASES[f"{image}-r1.5"] = dict(image=image, value=7.0, upper=-3.5, radius=1.5)
    CASES[f"{image}-r4"] = dict(image=image, value=7.0, lower=4.5, radius=4.0)
