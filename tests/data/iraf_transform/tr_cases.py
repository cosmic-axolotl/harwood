"""The cases of IRAF's fitcoords and transform gabarito: each a fresh copy of the
inputs, the fits it needs made first by IRAF's own fitcoords (interactive=no).

FITCOORDS: name -> (images, parameters); TRANSFORM: name -> (fits made first,
each (images, parameters), and the transform's own parameters)."""

FITCOORDS = {
    "default": ("arc", {}),
    "legendre": ("arc", {"function": "legendre", "xorder": 3, "yorder": 4}),
    "low": ("arc", {"xorder": 2, "yorder": 2}),
    "physical": ("arcl", {}),
    "axis1": ("arct", {"xorder": 4, "yorder": 5}),
    "combine": ("arc,arcl", {"combine": "yes", "fitname": "both"}),
    "fitname": ("arc", {"fitname": "f"}),
    "deletions": ("arc", {"deletions": "dels.db"}),
    "single": ("one", {"xorder": 3}),
    "twice": ("arc", {"_again": {"xorder": 3, "yorder": 3}}),
    "missing": ("gal", {}),
    "extension": ("arc.fits", {}),
}

_ARC = (("arc", {}),)

TRANSFORM = {
    "flux": (_ARC, {"input": "gal", "output": "o", "fitnames": "arc"}),
    "noflux_linear": (_ARC, {"input": "arc", "output": "o", "fitnames": "arc",
                             "flux": "no", "interptype": "linear"}),
    "nearest": (_ARC, {"input": "gal", "output": "o", "fitnames": "arc",
                       "interptype": "nearest"}),
    "poly3": (_ARC, {"input": "gal", "output": "o", "fitnames": "arc",
                     "interptype": "poly3"}),
    "poly5": (_ARC, {"input": "gal", "output": "o", "fitnames": "arc",
                     "interptype": "poly5"}),
    "blank": (_ARC, {"input": "gal", "output": "o", "fitnames": "arc", "y1": 6280.0,
                     "y2": 6670.0, "dy": 1.0, "blank": -1.0}),
    "edges": (_ARC, {"input": "gal", "output": "o", "fitnames": "arc", "y1": 6280.0,
                     "y2": 6670.0, "dy": 1.0, "x1": -3.0, "x2": 66.0}),
    "ylog": (_ARC, {"input": "gal", "output": "o", "fitnames": "arc", "ylog": "yes"}),
    "ny": (_ARC, {"input": "gal", "output": "o", "fitnames": "arc", "ny": 200}),
    "dy": (_ARC, {"input": "gal", "output": "o", "fitnames": "arc", "dy": 2.0}),
    "y2_from": (_ARC, {"input": "gal", "output": "o", "fitnames": "arc", "dy": 1.5,
                       "ny": 150, "y2": 6600.0}),
    "y1_given": (_ARC, {"input": "gal", "output": "o", "fitnames": "arc", "dy": 1.5,
                        "ny": 150, "y1": 6350.0}),
    "ylog_dy": (_ARC, {"input": "gal", "output": "o", "fitnames": "arc", "ylog": "yes",
                       "dy": 0.0001}),
    "x_axis": (_ARC, {"input": "gal", "output": "o", "fitnames": "arc", "x1": 5.0,
                      "x2": 55.0, "nx": 26, "xlog": "yes"}),
    "axis1": ((("arct", {"xorder": 4, "yorder": 5}),),
              {"input": "arct", "output": "o", "fitnames": "arct"}),
    "physical": ((("arcl", {}),), {"input": "arcl", "output": "o", "fitnames": "arcl"}),
    "two_fits": ((("arc", {}), ("arc", {"fitname": "leg", "function": "legendre",
                                        "xorder": 4, "yorder": 5})),
                 {"input": "gal", "output": "o", "fitnames": "arc,legarc"}),
    "two_images": (_ARC, {"input": "gal,arc", "output": "o,o2", "fitnames": "arc"}),
    "integer": (_ARC, {"input": "gali", "output": "o", "fitnames": "arc"}),
    "masks": (_ARC, {"input": "gal", "output": "o", "fitnames": "arc",
                     "minput": "bpm.fits", "moutput": "mo.fits"}),
    "mask_out": (_ARC, {"input": "gal", "output": "o", "fitnames": "arc", "y1": 6280.0,
                        "y2": 6670.0, "dy": 1.0, "moutput": "mo.fits"}),
    "twice": ((("arc", {}), ("arc", {"xorder": 3, "yorder": 3})),
              {"input": "gal", "output": "o", "fitnames": "arc"}),
    "single": ((("one", {"xorder": 3}),),
               {"input": "gal", "output": "o", "fitnames": "one"}),
}

#: transform by the image's own world system (fitnames empty): the galaxy first
#: rectified as in "flux" (w.fits, a linear WCS), or a section of it copied (with
#: LTV), then resampled. name -> (section copied or "", the parameters).
WCS = {
    "wcs_linear": ("", {"y1": 6300.0, "y2": 6640.0, "dy": 1.0}),
    "wcs_default": ("", {}),
    "wcs_log": ("", {"ylog": "yes", "flux": "no"}),
    "wcs_x": ("", {"x1": 3.0, "x2": 58.0, "nx": 40, "interptype": "linear"}),
    "wcs_section": ("[5:55,10:290]", {"y1": 6320.0, "y2": 6600.0, "dy": 0.8,
                                      "blank": 0.0}),
}
