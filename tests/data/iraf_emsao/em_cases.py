"""The cases of the emsao gabarito: the spectra, emsao's parameters and contpars'
(the continuum fit). Every case runs with displot=no, curmode=no and the log to
STDOUT and emsao.log; the spectra are em_inputs.py's."""

NONE = {"vel_corr": "none"}

CASES = {
    "ngc": ("ngc.fits", NONE, {}),
    "nopos": ("nopos.fits", NONE, {}),
    "blue": ("blue.fits", NONE, {}),
    "far": ("far.fits", NONE, {}),
    "single": ("single.fits", NONE, {}),
    "single_sigline": ("single.fits", {**NONE, "sigline": 25.0}, {}),
    "none": ("none.fits", NONE, {}),
    "subtracted": ("subtracted.fits", NONE, {}),
    "loglinear": ("loglinear.fits", NONE, {}),
    "ms": ("ms.fits", {**NONE, "specnum": "1-2"}, {}),
    "ms_two": ("ms.fits", {**NONE, "specnum": "2"}, {}),
    "two_files": ("ngc.fits,blue.fits", NONE, {}),
    # velocity corrections
    "barycentric": ("dated.fits", {"vel_corr": "barycentric"}, {}),
    "heliocentric": ("dated.fits", {"vel_corr": "heliocentric"}, {}),
    "slashdate": ("slashdate.fits", {"vel_corr": "barycentric"}, {}),
    "mjd": ("mjd.fits", {"vel_corr": "barycentric"}, {}),
    "observatory": ("observatory.fits", {"vel_corr": "barycentric"}, {}),
    "observatory_helio": ("observatory.fits", {"vel_corr": "heliocentric"}, {}),
    "no_date": ("ngc.fits", {"vel_corr": "barycentric"}, {}),
    "file": ("filebcv.fits", {"vel_corr": "file"}, {}),
    "hfile": ("filebcv.fits", {"vel_corr": "hfile"}, {}),
    "file_hcv_only": ("filehcv.fits", {"vel_corr": "file"}, {}),
    "file_none": ("dated.fits", {"vel_corr": "file"}, {}),
    # the initial velocity
    "guess": ("ngc.fits", {**NONE, "vel_init": "guess", "czguess": 790.0}, {}),
    "zguess": ("blue.fits", {**NONE, "vel_init": "zguess", "czguess": 0.0498}, {}),
    "guess_off": ("ngc.fits", {**NONE, "vel_init": "guess", "czguess": 1500.0}, {}),
    "guess_zero": ("blue.fits", {**NONE, "vel_init": "guess", "czguess": 0.0}, {}),
    # searching and fitting
    "nsmooth": ("ngc.fits", {**NONE, "nsmooth": 3}, {}),
    "linesig": ("ngc.fits", {**NONE, "linesig": 4.0}, {}),
    "wspan": ("blue.fits", {**NONE, "wspan": 5.7}, {}),
    "npfit": ("ngc.fits", {**NONE, "npfit": 3}, {}),
    "esmooth": ("ngc.fits", {**NONE, "esmooth": 2}, {}),
    "disperr": ("ngc.fits", {**NONE, "disperr": 0.1}, {}),
    "lsmin": ("ngc.fits", {**NONE, "lsmin": 6.0}, {}),
    "range": ("blue.fits", {**NONE, "st_lambda": 3800.0, "end_lambda": 5400.0}, {}),
    "range_negative": ("ngc.fits", {**NONE, "st_lambda": -30.0, "end_lambda": -40.0}, {}),
    "mylines": ("ngc.fits", {**NONE, "emlines": "mylines.dat", "linedir": "",
                             "emsearch": "mysearch.dat"}, {}),
    "mylines_path": ("ngc.fits", {**NONE, "emlines": "mylines.dat", "linedir": "",
                                  "emsearch": "mysearch.dat",
                                  "ablines": "rvsao$lib/ablines.dat",
                                  "emcombine": "rvsao$lib/emcomb.dat"}, {}),
    "missing_list": ("ngc.fits", {**NONE, "emlines": "nothere.dat"}, {}),
    "nocombine": ("ngc.fits", {**NONE, "emcombine": ""}, {}),
    # the continuum
    "legendre": ("ngc.fits", NONE, {"c_function": "legendre", "order": 3}),
    "chebyshev": ("blue.fits", NONE, {"c_function": "chebyshev", "order": 5,
                                      "niterate": 3}),
    "noreject": ("ngc.fits", NONE, {"s_low_reject": 0.0, "s_high_reject": 0.0}),
    "spline1": ("ngc.fits", NONE, {"c_function": "spline1", "order": 4, "grow": 0.0}),
}
