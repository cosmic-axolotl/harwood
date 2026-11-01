"""Sessions of curfit with keys, for points of wide.txt in curfit_cases.py."""

BASE = dict(function="legendre", order=5, weighting="uniform", axis=1, listdata=False,
            verbose=True, power=False)
ONE, TWO = "3372.500 10033.7900", "4490.000 10120.6400"
THREE, FOUR = "4117.500 10094.1100", "5048.750 10155.8975"

KEYS = {
    "delete": (f"{ONE} 1 d\n{THREE} 1 d\n0 0 1 f\n0 0 1 q\n", dict(BASE, calctype="real")),
    "undelete": (f"{ONE} 1 d\n{TWO} 1 d\n{ONE} 1 u\n0 0 1 f\n0 0 1 q\n",
                 dict(BASE, calctype="double")),
    "order": ("0 0 1 :order 3\n0 0 1 f\n0 0 1 q\n", dict(BASE, calctype="double")),
    "function": ("0 0 1 :function chebyshev\n0 0 1 :order 4\n0 0 1 f\n0 0 1 q\n",
                 dict(BASE, calctype="real")),
    "both": (f"{FOUR} 1 d\n0 0 1 :sample 3100:5000\n0 0 1 :low_reject 2\n"
             "0 0 1 :high_reject 2\n0 0 1 :niterate 2\n0 0 1 f\n0 0 1 q\n",
             dict(BASE, calctype="double")),
    "nofit": (f"{ONE} 1 d\n0 0 1 q\n", dict(BASE, calctype="double")),
    "bare": (":order 3\nf\nq\n", dict(BASE, calctype="real")),
}
