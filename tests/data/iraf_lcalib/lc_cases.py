"""The cases of IRAF's lcalib gabarito: each option on one star of onedstds$redcal/,
the extinction table, the list of stars, and a blackbody."""

CASES = {
    "bands": dict(option="bands", star_name="feige24", caldir="onedstds$redcal/"),
    "mags": dict(option="mags", star_name="feige24", caldir="onedstds$redcal/"),
    "fnu": dict(option="fnu", star_name="feige24", caldir="onedstds$redcal/"),
    "flam": dict(option="flam", star_name="g191b2b", caldir="onedstds$redcal/",
                 fnuzero=3.5e-20),
    "ext": dict(option="ext", star_name="", extinction="onedstds$kpnoextinct.dat"),
    "stars": dict(option="stars", star_name="", caldir="onedstds$redcal/"),
    "blackbody": dict(option="flam", star_name="V", caldir="onedstds$blackbody/", mag=10.3,
                      magband="V", teff="A0V"),
}
