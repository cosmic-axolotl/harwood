"""The cases of the xcsao graph gabarito: RVSAO's xcsao with its graph (displot) and,
mostly, its cursor (curmode): the spectrum above, the correlation with the best
template below, the summary at the right (xcsplot, xcorplot, xciplot); dispmode 2
and 3 and the l and m keys go to emsao's graph (emplot). Each case: the spectra,
xcsao's parameters, the cursor reads ('x y wcs key' a line) and the answers to what
the keys ask (getline), a line each. The spectra are xc_inputs.py's."""

TWO = {"templates": "tg.fits,ta.fits"}


def keys(*reads: str) -> str:
    return "".join(f"{r}\n" for r in reads)


def at(x: float, key: str, y: float = 0.5, wcs: int = 1) -> str:
    return f"{x} {y} {wcs} {key}"


Q = at(5000, "q")

CASES = {
    # what is drawn
    "first": ("g2500.fits", TWO, keys(Q), ""),
    "one_template": ("g2500.fits", {"templates": "tg.fits"}, keys(Q), ""),
    "three": ("gal6000.fits", {"templates": "ta.fits,tg.fits,tem.fits"}, keys(Q), ""),
    "nocursor": ("g2500.fits", {**TWO, "curmode": "no"}, "", ""),
    "dispmode0": ("g2500.fits", {**TWO, "dispmode": 0}, keys(Q), ""),
    "dispmode_neg": ("g2500.fits", {**TWO, "dispmode": -1}, keys(Q), ""),
    "dispmode2": ("g2500.fits", {**TWO, "dispmode": 2}, keys(at(5000, "q")), ""),
    "dispmode3": ("g2500.fits", {**TWO, "dispmode": 3}, keys(at(5000, "q")), ""),
    "range": ("g2500.fits", {**TWO, "st_lambda": 4200.0, "end_lambda": 6000.0}, keys(Q),
              ""),
    "template_range": ("g2500.fits", {"templates": "tg.fits:4500-6200"}, keys(Q), ""),
    "dated": ("gsite.fits", {"templates": "tg.fits", "svel_corr": "barycentric"}, keys(Q),
              ""),
    "zeropad": ("g2500.fits", {"templates": "tg.fits", "zeropad": "yes"}, keys(Q), ""),
    "emchop": ("emabs.fits", {"templates": "tg.fits", "s_emchop": "yes"}, keys(Q), ""),
    "temchop": ("gal6000.fits", {"templates": "tem.fits", "t_emchop": "yes",
                                 "t_em_reject": 2.0}, keys(Q), ""),
    "pkfrac_negative": ("g2500.fits", {"templates": "tg.fits", "pkfrac": -0.4},
                        keys(Q), ""),
    "pkfrac_points": ("g2500.fits", {"templates": "tg.fits", "pkfrac": 9.0}, keys(Q), ""),
    "wavelength": ("g2500.fits", {"templates": "tg.fits", "correlate": "wavelength"},
                   keys(Q), ""),
    "pixel": ("g2500.fits", {"templates": "tg.fits", "correlate": "pixel"}, keys(Q), ""),
    "multispec": ("pair.fits", {"templates": "tg.fits", "specnum": "1-2"}, keys(Q, Q), ""),
    "two_files": ("g2500.fits,gneg.fits", {"templates": "tg.fits"}, keys(Q, Q), ""),
    # the keys
    "replot": ("g2500.fits", TWO, keys(at(5000, "r"), Q), ""),
    "template": ("g2500.fits", TWO, keys(at(5000, "t"), at(5000, "t"), at(5000, "t"), Q),
                 "2\n\n7\n"),
    "smooth": ("g2500.fits", TWO, keys(at(5000, "g"), at(5000, "r"), Q), "4\n"),
    "smooth_empty": ("g2500.fits", TWO, keys(at(5000, "g"), Q), "\n"),
    "mode": ("g2500.fits", TWO, keys(at(5000, "m"), at(5000, "m"), Q), "0\n-1\n"),
    "mode_same": ("g2500.fits", TWO, keys(at(5000, "m"), Q), "1\n"),
    "mode_lines": ("g2500.fits", TWO, keys(at(5000, "m"), at(5000, "x"), Q), "2\n"),
    "mode_lines_quit": ("g2500.fits", TWO, keys(at(5000, "m"), at(5000, "q")), "2\n"),
    "lines": ("g2500.fits", TWO, keys(at(5000, "l"), at(5000, "x"), Q), ""),
    "lines_quit": ("g2500.fits", TWO, keys(at(5000, "l"), at(5000, "q")), ""),
    "peak_params": ("g2500.fits", TWO, keys(at(5000, "c"), Q, Q), "3\n0.6\n10 30 200 300\n"),
    "peak_params_empty": ("g2500.fits", TWO, keys(at(5000, "c"), Q), "\n\n\n"),
    # what c and v mean to do, set from the start (RVSAO's own c and v are lost: xcfit
    # reads pkfrac, minvel and maxvel again, and the filter is its locals')
    "peak_params_direct": ("g2500.fits", {**TWO, "pkmode": 3, "pkfrac": 0.6,
                                          "low_bin": 10, "top_low": 30, "nrun": 200,
                                          "top_nrun": 300, "curmode": "no"}, "", ""),
    "velocity_direct": ("g2500.fits", {**TWO, "minvel": 0.0, "maxvel": 2000.0,
                                       "curmode": "no"}, "", ""),
    "refit": ("g2500.fits", TWO, keys(at(5000, "f"), Q), ""),
    "velocity_limits": ("g2500.fits", TWO, keys(at(5000, "v"), Q), "0\n2000\n"),
    "good": ("g2500.fits", {**TWO, "save_vel": "yes"}, keys(at(5000, "y"), Q), ""),
    "bad": ("g2500.fits", {**TWO, "save_vel": "yes"}, keys(at(5000, "n"), Q), ""),
    "questionable": ("g2500.fits", TWO, keys(at(5000, "j"), Q), ""),
    "exit": ("g2500.fits", {**TWO, "save_vel": "yes"}, keys(at(5000, "y"), at(5000, "x")),
             ""),
    "help": ("g2500.fits", TWO, keys(at(5000, "?"), at(5000, "h"), at(5000, "\\040"), Q), ""),
    "unknown": ("g2500.fits", TWO, keys(at(5000, "!"), Q), ""),
    "eof": ("g2500.fits", TWO, keys(at(5000, "r")), ""),
}
