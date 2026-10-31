"""What ccdred.combine is asked, over seven frames. All of it is a preset's business:
the defaults, each rejection, ccdred's scaling and what imcombine has that the *combine
scripts do not pass on: grow and weight."""

CASES = {"defaults": {}}
for kind in ("minmax", "sigclip", "avsigclip", "pclip"):
    CASES[f"reject_{kind}"] = {"reject": kind}
CASES["reject_ccdclip"] = {"reject": "ccdclip", "rdnoise": "8.", "gain": "1."}
CASES["reject_crreject"] = {"reject": "crreject", "rdnoise": "8.", "gain": "1."}
CASES["median"] = {"combine": "median"}
CASES["median_sigclip"] = {"combine": "median", "reject": "sigclip"}
CASES["scale_exposure"] = {"scale": "exposure"}
CASES["scale_mean"] = {"scale": "mean"}
CASES["scale_median"] = {"scale": "median"}
CASES["zero_median"] = {"zero": "median"}
CASES["weight_exposure"] = {"weight": "exposure"}
CASES["weight_mean"] = {"weight": "mean", "scale": "mean"}
CASES["grow1"] = {"reject": "minmax", "grow": 1}
CASES["grow2_sig"] = {"reject": "sigclip", "lsigma": 2.0, "hsigma": 2.0, "grow": 2}
CASES["grow1_keep"] = {"reject": "minmax", "grow": 1, "nkeep": 4}
CASES["grow_avsig"] = {"reject": "avsigclip", "lsigma": 1.5, "hsigma": 1.5, "grow": 1}
CASES["grow_ccd"] = {"reject": "ccdclip", "rdnoise": "8.", "lsigma": 1.5, "hsigma": 1.5,
                     "grow": 1, "nkeep": 3}
CASES["grow_median"] = {"combine": "median", "reject": "sigclip", "lsigma": 1.5, "hsigma": 1.5,
                        "grow": 1}
CASES["grow_pclip"] = {"reject": "pclip", "grow": 1}
CASES["thresholds"] = {"lthreshold": 50.0, "hthreshold": 600.0}
CASES["blank"] = {"lthreshold": 1.0e6, "blank": 7.5}
CASES["nkeep"] = {"reject": "sigclip", "lsigma": 1.0, "hsigma": 1.0, "nkeep": 5}
CASES["mclip_no"] = {"reject": "sigclip", "mclip": "no"}
CASES["section"] = {"images": "f0.fits[5:20,5:20],f1.fits[5:20,5:20],f2.fits[5:20,5:20]"}
