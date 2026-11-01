"""The inputs of curfit's gabarito: lists written here, images by curfit_images.py."""

LISTS = {
    "xy.txt": "# c\n1 2.5\n2 3.7\n3 7.1\n4 9.9\n5 16.2\n6 18.8\n7 27.3\n8 31.5\n",
    "unsorted.txt": "5 16.2\n1 2.5\n3 7.1\n3 8\n8 31.5\n2 3.7\n7 27.3\n4 9.9\n6 18.8\n3 7.5\n",
    "y.txt": "2.5\n3.7\n7.1\n\n  # not a comment\n9.9\n16.2\n18.8\n27.3\n31.5\n40.1\n48.2\n",
    "w.txt": "1 2.5 0.5\n2 3.7 0.25\n3 7.1 2\n4 9.9 0\n5 16.2 1\n6 18.8 1\n7 27.3\n"
    "8 31.5 0.1\n9 35.5 3 7\n10 44. 0.7\n",
    "zero.txt": "1 0\n2 3.7\n3 0\n4 -9.9\n5 16.2\n6 -18.8\n7 27.3\n8 31.5\n",
    "wide.txt": "\n".join(
        f"{3000 + 37.25 * i:.3f} "
        f"{1.0e4 + 3.5 * i - 0.0121 * i * i + (7 if i % 9 == 0 else 0):.4f}"
        for i in range(60)
    )
    + "\n",
    "one.txt": "3 4\n",
    "same.txt": "3 4\n3 5\n3 6\n",
    "two.txt": "1 4\n2 5\n",
    "bad.txt": "1 2\n3\nx y\n4 5\n5 6\n6 8\n",
    "empty.txt": "# nothing\n",
}

KW = dict(function="legendre", order=4, weighting="uniform", axis=1, listdata=False,
          verbose=False, power=False, calctype="double")


def case(name, **kw):
    return {"input": name, **{**KW, **kw}}


CASES = {}
for calc in ("real", "double"):
    for fn in ("legendre", "chebyshev", "spline3", "spline1"):
        for o in (1, 2, 3, 5, 8):
            CASES[f"xy-{fn}-{o}-{calc}"] = case(
                "xy.txt", function=fn, order=o, calctype=calc, verbose=True, power=True
            )
            CASES[f"wide-{fn}-{o}-{calc}"] = case(
                "wide.txt", function=fn, order=o, calctype=calc, power=True
            )
    for wt in ("uniform", "user", "statistical", "instrumental"):
        for f in ("w.txt", "zero.txt", "unsorted.txt", "y.txt"):
            CASES[f"{f}-{wt}-{calc}"] = case(f, weighting=wt, calctype=calc, verbose=True)
    CASES[f"list-{calc}"] = case("xy.txt", calctype=calc, listdata=True, power=True)
    CASES[f"many-{calc}"] = case("xy.txt,unsorted.txt", calctype=calc, order=2)
    for f in ("one.txt", "same.txt", "two.txt", "bad.txt", "empty.txt", "missing.txt"):
        CASES[f"{f}-{calc}"] = case(f, calctype=calc, verbose=True)
    for wt in ("uniform", "statistical", "user"):
        for ax in (1, 2):
            CASES[f"im-{wt}-{ax}-{calc}"] = case(
                "im2", weighting=wt, axis=ax, calctype=calc, verbose=True, power=True, order=3
            )
    CASES[f"im3-1-{calc}"] = case("im3", axis=1, calctype=calc, order=3)
    CASES[f"im3-3-{calc}"] = case("im3", axis=3, calctype=calc, order=2)
    CASES[f"im-noaxis-{calc}"] = case("im2", axis=3, calctype=calc)
    CASES[f"section-{calc}"] = case("im2[10:40,5:20]", axis=2, calctype=calc, order=3)
    CASES[f"line-{calc}"] = case("im2[*,7]", calctype=calc, order=3)
    CASES[f"fits-{calc}"] = case("fits.fits", calctype=calc, order=3)

# what only a session sets: colon commands to a cursor file
SESSIONS = {}
for calc in ("real", "double"):
    for fn in ("legendre", "chebyshev", "spline3"):
        for name, cmds in {
            "sample": {"sample": "4000:4800,5000:5100"},
            "reject": {"low_reject": 2, "high_reject": 2, "niterate": 3},
            "grow": {"low_reject": 2, "high_reject": 1.5, "niterate": 5, "grow": 60},
            "avg": {"naverage": 3},
            "med": {"naverage": -4},
            "all": {
                "sample": "3100:5000",
                "naverage": 2,
                "low_reject": 2.5,
                "high_reject": 2.5,
                "niterate": 2,
                "grow": 40,
            },
        }.items():
            kw = dict(function=fn, calctype=calc, order=5, verbose=True, power=True)
            SESSIONS[f"{name}-{fn}-{calc}"] = {"cmds": cmds, **case("wide.txt", **kw)}
