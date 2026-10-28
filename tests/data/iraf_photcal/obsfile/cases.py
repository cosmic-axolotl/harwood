"""The obsfile cases, shared by run_iraf.py and the test."""

ID = "1,6,7,2,3,4,5,8,9,11"
NOID = "1,6,7,2,3,4,5,8,9,0"
CASES = {
    "A": ("imsets.mkimsets", ID, dict(tolerance=8.0)),
    "B": ("imsets.mkimsets", NOID, dict(tolerance=8.0)),
    "C": ("imsets.mkimsets", NOID, dict(tolerance=0.0, wrap=False)),
    "D": ("imsets.mkimsets", ID, dict(tolerance=3.0, allfilters=False,
                                      apercors="apercors.all", normtime=True,
                                      minmagerr=0.02, shifts="shifts")),
    "E": ("sets2", ID, dict(tolerance=8.0, obsparams="obsparams", shifts="shifts")),
    "F": ("sets2", NOID, dict(tolerance=0.0, allfilters=False)),
    "G": ("sets2", NOID, dict(tolerance=8.0, allfilters=False, wrap=False,
                              apercors="apercors.all")),
}
