"""The cases of IRAF's fit1d and background gabarito: task, input, output and the
parameters (interactive=no always). Each on a fresh copy of the inputs."""

SKY = "1:18 43:61"

CASES = {
    "background": ("background", "sky", "o", {"sample": SKY, "order": 2}),
    "background_reject": ("background", "sky", "o", {"sample": SKY, "order": 3,
                                                     "low_reject": 2.5,
                                                     "high_reject": 2.5,
                                                     "niterate": 3}),
    "background_grow": ("background", "sky", "o", {"sample": SKY, "order": 2,
                                                   "low_reject": 3, "high_reject": 3,
                                                   "grow": 1.5}),
    "background_axis2": ("background", "sky", "o", {"axis": 2, "order": 4,
                                                    "function": "legendre",
                                                    "high_reject": 3}),
    "background_spline3": ("background", "sky", "o", {"sample": SKY, "order": 2,
                                                      "function": "spline3",
                                                      "high_reject": 3}),
    "background_naverage": ("background", "sky", "o", {"sample": SKY, "order": 2,
                                                       "naverage": 3,
                                                       "high_reject": 3}),
    "background_median": ("background", "sky", "o", {"sample": SKY, "order": 2,
                                                     "naverage": -3}),
    "fit_spline3": ("fit1d", "sky", "o", {"type": "fit", "order": 5}),
    "fit_spline1": ("fit1d", "sky", "o", {"type": "fit", "function": "spline1",
                                          "order": 4, "low_reject": 2,
                                          "high_reject": 2, "niterate": 2}),
    "ratio": ("fit1d", "sky", "o", {"type": "ratio", "function": "chebyshev",
                                    "order": 3, "axis": 2}),
    "difference_1d": ("fit1d", "spec", "o", {"type": "difference", "order": 6,
                                             "high_reject": 3, "niterate": 2}),
    "bpm": ("fit1d", "sky", "o", {"type": "difference", "function": "legendre",
                                  "order": 3, "bpm": "bpm.fits"}),
    "bpm_axis2": ("fit1d", "sky", "o", {"type": "fit", "function": "legendre",
                                        "order": 3, "axis": 2, "bpm": "bpm.fits",
                                        "high_reject": 3}),
    "section": ("fit1d", "sky[5:55,10:100]", "o", {"type": "difference",
                                                   "function": "legendre", "order": 2}),
    "section_ratio": ("fit1d", "sky[5:55,10:100]", "o", {"type": "ratio",
                                                         "function": "legendre",
                                                         "order": 2}),
    "existing": ("fit1d", "sky[5:55,10:100]", "old", {"type": "fit",
                                                      "function": "legendre",
                                                      "order": 2}),
    "inplace": ("fit1d", "sky", "sky", {"type": "difference", "function": "legendre",
                                        "order": 2}),
    "integer": ("fit1d", "skyi", "skyi", {"type": "difference", "function": "legendre",
                                          "order": 2}),
    "nodegree": ("fit1d", "sky", "o", {"type": "fit", "function": "legendre",
                                       "order": 4, "sample": "3:4"}),
}

#: Interactive cases: the answers to 'Fit line =' and the cursor file, keys as
#: IRAF's cursor files hold them ('x y wcs key' or ':command'), each fit up to q.
INTERACTIVE = {
    "by_hand": ("background", "sky", "o",
                {"order": 2, "_answers": "60\n",
                 "_cursor": "3 0 1 s\n17 0 1 s\n44 0 1 s\n59 0 1 s\n0 0 1 f\n0 0 1 q\n"}),
    "by_hand_two": ("background", "sky", "o",
                    {"order": 2, "_answers": "60\n30 40\n",
                     "_cursor": ":order 3\n:function legendre\n0 0 1 f\n0 0 1 q\n"
                                "2 0 1 s\n15 0 1 s\n46 0 1 s\n60 0 1 s\n0 0 1 f\n"
                                "0 0 1 q\n"}),
    "by_hand_reset": ("fit1d", "sky", "o",
                      {"type": "fit", "sample": "1:20 40:61", "_answers": "200\n",
                       "_cursor": "0 0 1 t\n:high_reject 3\n0 0 1 f\n0 0 1 q\n"}),
    "by_hand_axis2": ("background", "sky", "o",
                      {"axis": 2, "order": 2, "_answers": "31 33\n",
                       "_cursor": "5 0 1 s\n100 0 1 s\n0 0 1 f\n0 0 1 q\n"}),
}
