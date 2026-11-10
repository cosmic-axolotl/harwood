"""The cases of IRAF's imsum gabarito: the inputs (names without .fits, sections
allowed) and imsum's parameters when not its defaults."""


def r(*ks):
    return [f"r{k:02d}" for k in ks]


def s(*ks):
    return [f"s{k:02d}" for k in ks]


CASES = {}
for n in range(1, 9):
    for option in ("sum", "average", "median"):
        CASES[f"r{n}-{option}"] = dict(input=r(*range(1, n + 1)), option=option)
for n in (3, 4, 5, 6, 7):
    for low, high in ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (0.25, 0.25),
                      (0.4, 0), (0, 0.6), (1.5, 0.9)):
        if (int(low * n) if low < 1 else int(low)) + \
                (int(high * n) if high < 1 else int(high)) >= n:
            continue
        for option in ("sum", "average"):
            CASES[f"r{n}-{option}-{low}-{high}"] = dict(input=r(*range(1, n + 1)),
                                                        option=option, low_reject=low,
                                                        high_reject=high)
CASES["r3-reject-all"] = dict(input=r(1, 2, 3), low_reject=2, high_reject=1)
CASES["r3-median-ignores"] = dict(input=r(1, 2, 3), option="median", low_reject=2,
                                  high_reject=2)
for option in ("sum", "average", "median"):
    CASES[f"s4-{option}"] = dict(input=s(1, 2, 3, 4), option=option)
    CASES[f"s5-{option}-real"] = dict(input=s(1, 2, 3, 4, 5), option=option,
                                      calctype="real")
    CASES[f"s5-{option}-r-s"] = dict(input=s(1, 2, 3, 4, 5), option=option,
                                     calctype="real", pixtype="short")
    CASES[f"r5-{option}-short"] = dict(input=r(1, 2, 3, 4, 5), option=option,
                                       calctype="short")
    CASES[f"r5-{option}-r-i"] = dict(input=r(1, 2, 3, 4, 5), option=option,
                                     pixtype="int")
    CASES[f"r5-{option}-d"] = dict(input=r(1, 2, 3, 4, 5), option=option,
                                   calctype="double")
    CASES[f"s4-{option}-ushort"] = dict(input=s(1, 2, 3, 4), option=option,
                                        pixtype="ushort", calctype="int")
CASES["s4-average-1-1"] = dict(input=s(1, 2, 3, 4), option="average", low_reject=1,
                               high_reject=1)
CASES["s4-average-int-1-0"] = dict(input=s(1, 2, 3, 4), option="average", low_reject=1,
                                   calctype="int")
CASES["s-u"] = dict(input=["s01", "u01"])
CASES["u-u"] = dict(input=["u01", "u02"], option="average")
CASES["u-s-i"] = dict(input=["u01", "s01", "i01"])
CASES["s-r"] = dict(input=["s01", "r01"], option="average")
CASES["i-d"] = dict(input=["i01", "d01"])
CASES["r-d"] = dict(input=["r01", "d01"], option="average")
CASES["d-s"] = dict(input=["d01", "s01"], option="median")
CASES["b-b"] = dict(input=["b01", "b01"], option="sum")
CASES["b-s"] = dict(input=["b01", "s01"], option="sum")
CASES["long"] = dict(input=s(1, 2), calctype="long", pixtype="long")
for n in (15, 16, 20):
    CASES[f"r{n}-sum"] = dict(input=r(*range(1, n + 1)))
    CASES[f"r{n}-average"] = dict(input=r(*range(1, n + 1)), option="average")
    CASES[f"s{n}-average-r-s"] = dict(input=s(*range(1, n + 1)), option="average",
                                      calctype="real", pixtype="short")
    CASES[f"s{n}-sum"] = dict(input=s(*range(1, n + 1)))
    CASES[f"r{n}-median"] = dict(input=r(*range(1, n + 1)), option="median")
    CASES[f"r{n}-average-2-3"] = dict(input=r(*range(1, n + 1)), option="average",
                                      low_reject=2, high_reject=3)
CASES["one-average"] = dict(input=["one1", "one2", "one3"], option="average")
CASES["one-median"] = dict(input=["one1", "one2", "one3"], option="median")
CASES["cube-sum"] = dict(input=["cube1", "cube2", "cube3"])
CASES["cube-1-1"] = dict(input=["cube1", "cube2", "cube3"], low_reject=1, high_reject=1)
CASES["section"] = dict(input=["r01[2:5,*]", "r02[3:6,*]", "r03[1:4,*]"],
                        option="average")
CASES["section-step"] = dict(input=["s01[*:2,2:4]", "s02[*:2,1:3]"])
CASES["sizes"] = dict(input=["r01", "wide"])
CASES["dims"] = dict(input=["r01", "one1"])
CASES["hparams-sum"] = dict(input=r(1, 2, 3), hparams="EXPTIME,AIRMASS")
CASES["hparams-average"] = dict(input=r(1, 2, 3), option="average",
                                hparams="EXPTIME,AIRMASS")
CASES["hparams-median"] = dict(input=r(1, 2, 3), option="median",
                               hparams="EXPTIME,AIRMASS")
CASES["hparams-pattern"] = dict(input=r(1, 2, 3, 4, 5, 6, 7), option="average",
                                hparams="*TIME,AIR*")
CASES["title"] = dict(input=r(1, 2), title="the two summed")
CASES["verbose"] = dict(input=r(1, 2, 3), option="average", low_reject=1, verbose=True,
                        hparams="EXPTIME", pixtype="real", calctype="double")
CASES["verbose-defaults"] = dict(input=s(1, 2), verbose=True)
CASES["badtype"] = dict(input=r(1, 2), calctype="x")
for pixtype in ("ushort", "int", "long"):
    CASES[f"verbose-{pixtype}"] = dict(input=s(1, 2), verbose=True, pixtype=pixtype,
                                       calctype=pixtype if pixtype != "ushort" else "")
CASES["verbose-u"] = dict(input=["u01", "u02"], verbose=True)
CASES["verbose-b"] = dict(input=["b01", "s01"], verbose=True)
