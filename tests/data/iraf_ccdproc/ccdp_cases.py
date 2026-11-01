"""The cases of ccdproc's gabarito: (images, parameters with IRAF's names, what to compare)."""

SECTIONS = dict(biassec="[53:60,1:40]", trimsec="[1:52,1:37]")
BASE = dict(overscan="yes", trim="yes", **SECTIONS)
FULL = dict(zerocor="yes", zero="zero.fits", darkcor="yes", dark="dark.fits", flatcor="yes",
            flat="flat.fits", **BASE)
COL = dict(overscan="yes", trim="yes", biassec="[1:60,36:40]", trimsec="[1:60,1:35]",
           readaxis="column")


def _c(image, **params):
    return (image, params)


CASES = {
    # The overscan, of each kind
    "ov": _c("obj.fits", overscan="yes", biassec="[53:60,1:40]"),
    "ov-part": _c("obj.fits", overscan="yes", biassec="[53:60,5:30]"),
    "ov-trim": _c("obj.fits", **BASE),
    "ov-short": _c("obj.fits", overscan="yes", trim="yes", biassec="[53:60,1:40]",
                   trimsec="[3:52,4:30]"),
    "ov-o3": _c("obj.fits", order=3, **BASE),
    "ov-cheb": _c("obj.fits", function="chebyshev", order=4, **BASE),
    "ov-spl3": _c("obj.fits", function="spline3", order=3, niterate=2, low_reject=2.5,
                  high_reject=2.5, **BASE),
    "ov-spl1": _c("obj.fits", function="spline1", order=3, **BASE),
    "ov-avg": _c("obj.fits", order=2, naverage=4, **BASE),
    "ov-med": _c("obj.fits", order=2, naverage=-5, niterate=0, **BASE),
    "ov-samp": _c("obj.fits", order=2, sample="5:15,20:30", **BASE),
    "ov-grow": _c("obj.fits", order=2, grow=1.0, low_reject=1.5, high_reject=1.5, niterate=3,
                  **BASE),
    "ov-mean": _c("obj.fits", function="mean", **BASE),
    "ov-median": _c("obj.fits", function="median", **BASE),
    "ov-minmax": _c("obj.fits", function="minmax", **BASE),
    "ov-col": _c("cobj.fits", **COL),
    "ov-col-o3": _c("cobj.fits", order=3, **COL),
    "trim": _c("obj.fits", trim="yes", trimsec="[5:50,3:35]"),
    # Each calibration
    "zero": _c("obj.fits", zerocor="yes", zero="zero.fits", **BASE),
    "dark": _c("obj.fits", darkcor="yes", dark="dark.fits", zerocor="yes", zero="zero.fits",
               **BASE),
    "flat": _c("obj.fits", flatcor="yes", flat="flat.fits", zerocor="yes", zero="zero.fits",
               **BASE),
    "full": _c("obj.fits", **FULL),
    "full-norm": _c("obj.fits", flatcor="yes", flat="flat.fits", minreplace=9000.0, **BASE),
    "comp": _c("comp.fits", **FULL),
    "other": _c("other.fits", **FULL),
    "notype": _c("notype.fits", **FULL),
    # Each type of target
    "t-zero": _c("zero.fits", ccdtype="zero", **FULL),
    "t-dark": _c("dark.fits", ccdtype="dark", **FULL),
    "t-flat": _c("flat.fits", ccdtype="flat", **FULL),
    "t-flat-min": _c("flat.fits", ccdtype="flat", minreplace=8300.0, **FULL),
    "t-illum": _c("illumraw.fits", ccdtype="illum", **FULL),
    "t-fringe": _c("fringeraw.fits", ccdtype="fringe", **FULL),
    # Illumination and fringe
    "illum": _c("obj.fits", illumcor="yes", illum="illum.fits", **FULL),
    "illum-mean": _c("obj.fits", illumcor="yes", illum="illummean.fits", **FULL),
    "illum-noflag": _c("obj.fits", illumcor="yes", illum="illumnoflag.fits", **FULL),
    "fringe": _c("obj.fits", fringecor="yes", fringe="fringe.fits", **FULL),
    "fringe-scl": _c("obj.fits", fringecor="yes", fringe="fringescl.fits", **FULL),
    "fringe-noflag": _c("obj.fits", fringecor="yes", fringe="fringenoflag.fits", **FULL),
    "both": _c("obj.fits", illumcor="yes", illum="illum.fits", fringecor="yes",
               fringe="fringescl.fits", **FULL),
    # readcor
    "read": _c("obj.fits", readcor="yes", zerocor="yes", zero="zero.fits", **BASE),
    "read-target": _c("zero.fits", ccdtype="zero", readcor="yes", **BASE),
    "read-col": _c("cobj.fits", readcor="yes", zerocor="yes", zero="czero.fits", **COL),
    "read-col-target": _c("czero.fits", ccdtype="zero", readcor="yes", **COL),
    # scancor
    "scan-short": _c("obj.fits", scancor="yes", scantype="shortscan", nscan=3, **FULL),
    "scan-1": _c("obj.fits", scancor="yes", scantype="shortscan", nscan=1, **FULL),
    "scan-key": _c("scan.fits", scancor="yes", scantype="shortscan", nscan=3, **FULL),
    "scan-long": _c("obj.fits", scancor="yes", scantype="longscan", **FULL),
    "scan-long-col": _c("cobj.fits", scancor="yes", scantype="longscan", zerocor="yes",
                        zero="czero.fits", flatcor="yes", flat="cflat.fits", **COL),
    "scan-short-col": _c("cobj.fits", scancor="yes", scantype="shortscan", nscan=3,
                         zerocor="yes", zero="czero.fits", flatcor="yes", flat="cflat.fits",
                         **COL),
    # fixpix
    "fix": _c("obj.fits", fixpix="yes", fixfile="bad.pl", **BASE),
    "fix-only": _c("obj.fits", fixpix="yes", fixfile="bad.pl"),
    "fix-fits": _c("obj.fits", fixpix="yes", fixfile="bad.fits", **FULL),
    "fix-median": _c("obj.fits", fixpix="yes", fixfile="bad.pl", function="median", **BASE),
    # Sections
    "sub": _c("sub.fits", overscan="yes", biassec="[53:60,*]"),
    "sub-trim": _c("sub.fits", overscan="yes", trim="yes", biassec="[53:60,*]",
                   trimsec="[5:50,3:35]"),
    "sub-cal": _c("sub2.fits", zerocor="yes", zero="zero2.fits", flatcor="yes",
                  flat="flat2.fits"),
    "sub-cal-illum": _c("sub2.fits", illumcor="yes", illum="illum2.fits", fringecor="yes",
                        fringe="fringe2.fits"),
    "hdr": _c("hdr.fits", overscan="yes", trim="yes", biassec="image", trimsec="image"),
}
