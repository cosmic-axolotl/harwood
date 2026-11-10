"""The cases of IRAF's response and illumination gabarito: the task, its images,
and the parameters when not the defaults."""

RESPONSE = {}
for axis in (2, 1):
    f = f"flat{axis}"
    for name, params in (
            ("default", {}),
            ("spline3-3", dict(function="spline3", order=3)),
            ("legendre-4", dict(function="legendre", order=4)),
            ("chebyshev-5", dict(function="chebyshev", order=5)),
            ("spline1-6", dict(function="spline1", order=6)),
            ("reject", dict(function="chebyshev", order=5, low_reject=2.0,
                            high_reject=2.0, niterate=3, grow=1.0)),
            ("naverage3", dict(function="legendre", order=4, naverage=3)),
            ("naverage-3", dict(function="legendre", order=4, naverage=-3)),
            ("sample", dict(function="chebyshev", order=3, sample="10:50 60:110")),
            ("threshold", dict(function="chebyshev", order=5, threshold=900.0)),
            ("sky", dict(function="chebyshev", order=5, normalization=f"sky{axis}"))):
        params = dict(params)
        norm = params.pop("normalization", f)
        RESPONSE[f"r{axis}-{name}"] = dict(calibration=f, normalization=norm, **params)
RESPONSE["r-section"] = dict(calibration="flat2[5:25,10:110]", normalization="flat2",
                             function="chebyshev", order=5)
RESPONSE["r-existing"] = dict(calibration="flat2[5:25,10:110]", normalization="flat2",
                              function="chebyshev", order=5, output="pre")
RESPONSE["r-nodisp"] = dict(calibration="nodisp", normalization="nodisp",
                            function="chebyshev", order=5, dispaxis=2)

ILLUMINATION = {}
for axis in (2, 1):
    s = f"sky{axis}"
    for name, params in (
            ("default", {}),
            ("course", dict(function="chebyshev", order=2, high_reject=1.0, grow=2.0,
                            niterate=2)),
            ("nbins1", dict(nbins=1)),
            ("nbins2", dict(nbins=2)),
            ("nbins3", dict(nbins=3)),
            ("nbins4", dict(nbins=4)),
            ("nbins6", dict(nbins=6)),
            ("nbins7", dict(nbins=7, function="legendre", order=3)),
            ("bins", dict(bins="1:30 40:70 100:120")),
            ("bins-unsorted", dict(bins="90:110,5:25 40:60 70:80")),
            ("bins-overlap", dict(bins="1:60 30:90 60:120 100:118")),
            ("nearest", dict(interpolator="nearest")),
            ("linear", dict(interpolator="linear")),
            ("poly5", dict(interpolator="poly5", nbins=6)),
            ("poly5-5", dict(interpolator="poly5")),
            ("poly5-3", dict(interpolator="poly5", nbins=3)),
            ("spline3", dict(interpolator="spline3")),
            ("spline3-3", dict(interpolator="spline3", nbins=3)),
            ("reject", dict(function="spline3", order=2, low_reject=2.5,
                            high_reject=2.5, niterate=4)),
            ("flat", dict(image=f"flat{axis}", function="legendre", order=3))):
        params = dict(params)
        image = params.pop("image", s)
        ILLUMINATION[f"i{axis}-{name}"] = dict(images=image, **params)
ILLUMINATION["i-section"] = dict(images="sky2[3:28,*]", function="chebyshev", order=2)
ILLUMINATION["i-nodisp"] = dict(images="nodisp", dispaxis=2)
