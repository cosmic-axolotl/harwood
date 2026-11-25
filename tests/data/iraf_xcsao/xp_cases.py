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
    # emsao's graph from xcsao (emplot): the lines, and back to the correlation
    "lines_x": ("emlines.fits", TWO, keys(at(5000, "l"), at(5050, "z"), at(5100, "z"),
                                          at(5060, "w"), at(5000, "x"), Q), ""),
    "lines_f": ("emlines.fits", TWO, keys(at(5000, "l"), at(5000, "f"), Q), ""),
    "lines_set": ("emlines.fits", {**TWO, "save_vel": "yes"},
                  keys(at(5000, "l"), at(5000, "s"), at(1, "e"), at(5000, "y"),
                       at(5000, "]"), Q), ""),
    "lines_t": ("emlines.fits", TWO, keys(at(5000, "l"), at(5000, "t"), at(5000, "x"),
                                          Q), "2\n"),
    "lines_keys": ("emlines.fits", TWO, keys(at(5000, "l"), at(5000, "/"), at(5000, "h"),
                                             at(5000, "o"), at(1, "a"), at(5000, "v"),
                                             at(1, "e"), at(5000, "k"), at(5000, "p"),
                                             at(5000, "x"), Q), ""),
    "lines_emission": ("emlines.fits", TWO, keys(at(5000, "l"), at(5073.7, "e"),
                                                 at(5000, "x"), Q), "oiii\n"),
    "dispmode2_lines": ("emlines.fits", {**TWO, "dispmode": 2}, keys(at(5000, "x"), Q),
                        ""),
    "dispmode3_lines": ("emlines.fits", {**TWO, "dispmode": 3}, keys(at(5000, "q")), ""),
    "vel_plot_comb": ("emlines.fits", {**TWO, "dispmode": 2, "vel_plot": "combination"},
                      keys(at(5000, "q")), ""),
    "lines_plain": ("g2500.fits", TWO, keys(at(5000, "l"), at(5000, "]")), ""),
    # e: the spectrum, to edit by eye (plotspec), and correlated again
    "edit": ("g2500.fits", TWO, keys(at(5000, "e"), at(5000, "q"), Q), ""),
    "edit_keys": ("g2500.fits", TWO, keys(at(5000, "e"), at(5172.5, "w", 700),
                                          at(5000, "r"), at(5000, "h"), at(5000, "g"),
                                          at(5000, "h"), at(4800, "z"), at(5300, "z"),
                                          at(5000, "w"), at(5000, "u"), at(5000, "!"),
                                          at(5000, "?"), at(5000, "\\040"),
                                          at(5000, "s"), at(5000, "q"), Q), "3\n30\n"),
    "edit_point": ("g2500.fits", TWO, keys(at(5000, "e"), at(5172.7, ",", 2000),
                                           at(5000, "q"), Q), ""),
    "edit_delete": ("g2500.fits", TWO, keys(at(5000, "e"), at(5160, "d"), at(5190, "d"),
                                            at(5000, "q"), Q), ""),
    "edit_cancel": ("g2500.fits", TWO, keys(at(5000, "e"), at(4800, "z"), at(5000, "x"),
                                            at(5160, "d"), at(5000, "x"), at(5000, "Q"),
                                            Q), ""),
    "edit_ends": ("g2500.fits", TWO, keys(at(5000, "e"), at(3000, "w"), at(7000, "w"),
                                          at(3000, "z"), at(9000, "z"), at(5000, "q"),
                                          Q), ""),
    # p: each template's correlation, to choose its peak by eye
    "peak": ("g2500.fits", TWO, keys(at(5000, "p"), at(1500, "p", 0.2), at(0, "q"),
                                     at(0, "q"), Q), ""),
    "peak_auto": ("g2500.fits", TWO, keys(at(5000, "p"), at(0, "q"), at(0, "q"), Q), ""),
    "peak_width": ("g2500.fits", TWO, keys(at(5000, "p"), at(0, "s"), at(1500, "p", 0.2),
                                           at(0, "q"), at(2400, "p"), at(0, "q"), Q),
                   "4\n"),
    "peak_again": ("g2500.fits", TWO, keys(at(5000, "p"), at(2600, "p"), at(0, "q"),
                                           at(0, "q"), at(5000, "f"), at(0, "q"),
                                           at(0, "q"), Q), ""),
    "peak_keys": ("g2500.fits", {"templates": "tg.fits"},
                  keys(at(5000, "p"), at(0, "z"), at(5000, "z"), at(2500, "w"),
                       at(2500, "p"), at(0, "u"), at(0, "q"), Q), ""),
    # correlate=no: the results xcsao wrote, read back and shown
    "nocorr": ("gsaved.fits", {**TWO, "correlate": "no"}, keys(Q), ""),
    "nocorr_keys": ("gsaved.fits", {**TWO, "correlate": "no"},
                    keys(at(5000, "t"), at(5000, "r"), at(5000, "g"), at(5000, "m"), Q),
                    "2\n2\n0\n"),
    "nocorr_quality": ("gsaved.fits", {**TWO, "correlate": "no", "save_vel": "yes"},
                       keys(at(5000, "y"), Q), ""),
    "nocorr_lines": ("gsaved.fits", {**TWO, "correlate": "no"},
                     keys(at(5000, "l"), at(5000, "x"), Q), ""),
    "nocorr_report": ("gsaved.fits", {**TWO, "correlate": "no", "report_mode": 2,
                                      "curmode": "no"}, "", ""),
    "nocorr_none": ("g2500.fits", {**TWO, "correlate": "no"}, keys(Q), ""),
    "help": ("g2500.fits", TWO, keys(at(5000, "?"), at(5000, "h"), at(5000, "\\040"), Q), ""),
    "unknown": ("g2500.fits", TWO, keys(at(5000, "!"), Q), ""),
    "eof": ("g2500.fits", TWO, keys(at(5000, "r")), ""),
}

FIXED = {name for name, (_, _, cursor, _) in CASES.items()
         if any(line.split()[3:] == ["d"] for line in cursor.splitlines())}
