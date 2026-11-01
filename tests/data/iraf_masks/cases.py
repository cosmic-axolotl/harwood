"""What IRAF is asked; each output is kept here as IRAF wrote it."""

TEXT2MASK = {
    "regions": {"text": "regions.txt", "ncols": 20, "nlines": 12},
    "lines": {"text": "lines.txt", "ncols": 20, "nlines": 12},
    "values": {"text": "lines.txt", "ncols": 20, "nlines": 12, "linterp": 5, "cinterp": 6,
               "square": 7, "pixel": 5000},
    "big_value": {"text": "lines.txt", "ncols": 20, "nlines": 12, "pixel": 70000},
    "small": {"text": "lines.txt", "ncols": 3, "nlines": 2},
    "empty": {"text": "empty.txt", "ncols": 8, "nlines": 4},
    "wide": {"text": "lines.txt", "ncols": 9000, "nlines": 12},
}
BADPIX = {
    "plain": {"fixfile": "badpix.txt", "template": "tmpl.fits"},
    "values": {"fixfile": "badpix.txt", "template": "tmpl.fits", "goodvalue": 0,
               "badvalue": 1},
    "untrimmed_none": {"fixfile": "untrimmed.txt", "template": "tmpl.fits"},
    "untrimmed": {"fixfile": "untrimmed.txt", "template": "trim.fits"},
    "untrimmed2": {"fixfile": "untrimmed.txt", "template": "trim2.fits"},
    "empty": {"fixfile": "empty.txt", "template": "tmpl.fits"},
}
CCDMASK = {
    "flat": {"image": "flat.fits"},
    "flat2": {"image": "flats.fits"},
    "tight": {"image": "flat.fits", "lsigma": 3.0, "hsigma": 3.0},
    "loose": {"image": "flat.fits", "lsigma": 20.0, "hsigma": 20.0},
    "boxes": {"image": "flat.fits", "ncmed": 3, "nlmed": 11, "ncsig": 25, "nlsig": 10},
    "even": {"image": "flat.fits", "ncmed": 4, "nlmed": 6},
    "ngood1": {"image": "flat.fits", "ngood": 1},
    "ngood20": {"image": "flat.fits", "ngood": 20},
    "codes": {"image": "flat.fits", "linterp": 5, "cinterp": 6, "eqinterp": 7},
    "small": {"image": "small.fits"},
    "section": {"image": "flat.fits[11:100,21:90]"},
    "short": {"image": "short.fits"},
    "wide": {"image": "wide.fits"},
    "wider": {"image": "wider.fits"},
}
PLIO = {
    "imexpr": ("imexpr", {"expr": "a > 100", "output": "p_imexpr.pl", "a": "ramp.fits"}),
    "imexpr_value": ("imexpr", {"expr": "a / 10", "output": "p_imexpr_value.pl",
                                "a": "ramp.fits", "outtype": "int"}),
    "imexpr_dims": ("imexpr", {"expr": "I > 3 && J < 3", "output": "p_imexpr_dims.pl",
                               "dims": "8,5"}),
    "imcopy": ("imcopy", {"input": "ramp.fits", "output": "p_imcopy.pl"}),
}
READ_BACK = {
    "stat": ("t_lines.pl", "imstat"),
    "expr": ("t_lines.pl", "imexpr"),
    "minmax": ("t_lines.pl", "minmax"),
}

D = "24,18"
MSKEXPR = {}


def mx(name, expr, **params):
    MSKEXPR[name] = {"expr": expr, **{"dims": D, **params}}


for name, args in {
    "point": "7., 9.",
    "point_round": "7.4, 8.6",
    "circle": "12., 9., 5.",
    "circle_off": "2.5, 3.5, 6.",
    "circle_big": "12., 9., 500.",
    "ellipse": "12., 9., 8., 0.5, 30.",
    "ellipse_flat": "12., 9., 8., 0.2, 0.",
    "ellipse_up": "12., 9., 8., 0.5, 90.",
    "ellipse_neg": "12., 9., 8., 0.5, -45.",
    "box": "4., 3., 15., 11.",
    "box_rev": "15., 11., 4., 3.",
    "box_half": "4.5, 3.5, 15.5, 11.5",
    "rectangle": "12., 9., 6., 0.5, 0.",
    "rectangle_turn": "12., 9., 6., 0.5, 30.",
    "rectangle_45": "12., 9., 6., 1., 45.",
    "rectangle_90": "12., 9., 6., 0.5, 90.",
    "vector": "3., 3., 20., 14., 3.",
    "vector_flat": "3., 9., 20., 9., 4.",
    "vector_up": "9., 2., 9., 15., 4.",
    "vector_back": "20., 14., 3., 3., 2.5",
    "pie": "12., 9., 30., 120.",
    "pie_wide": "12., 9., 30., 300.",
    "pie_wrap": "12., 9., 300., 60.",
    "pie_right": "12., 9., 0., 90.",
    "pie_left": "12., 9., 180., 270.",
    "pie_up": "12., 9., 90., 180.",
    "pie_down": "12., 9., 270., 360.",
    "pie_none": "12., 9., 40., 40.",
    "pie_corner": "2., 2., 10., 80.",
    "cannulus": "12., 9., 3., 7.",
    "eannulus": "12., 9., 3., 8., 0.6, 30.",
    "rannulus": "12., 9., 2., 7., 0.7, 0.",
    "rannulus_turn": "12., 9., 2., 7., 0.7, 25.",
    "polygon": "3., 3., 20., 5., 12., 15.",
    "polygon_square": "5., 5., 15., 5., 15., 12., 5., 12.",
    "polygon_star": "12., 2., 15., 8., 22., 9., 16., 12., 18., 17., 12., 14., 6., 17., "
                    "8., 12., 2., 9., 9., 8.",
    "polygon_half": "5.5, 5.5, 15.5, 6.5, 12.5, 12.5",
    "polygon_u": "3., 3., 20., 3., 20., 14., 15., 14., 15., 7., 8., 7., 8., 14., 3., 14.",
    "polygon_w": "3., 14., 7., 3., 12., 10., 17., 3., 21., 14.",
    "polygon_flat": "4., 6., 14., 6., 18., 6., 18., 12., 4., 12.",
    "pannulus": "2., 8., 6., 16., 6., 16., 12., 8., 12.",
    "pannulus_tri": "1.5, 7., 5., 17., 6., 11., 13.",
}.items():
    base = name.split("_")[0]
    mx("r_" + name, f"{base} ({args}) ? 1 : 0")
for base, args in (("point", "7., 9."), ("circle", "12., 9., 5."),
                   ("ellipse", "12., 9., 8., 0.5, 30."), ("box", "4., 3., 15., 11."),
                   ("rectangle", "12., 9., 6., 0.5, 30."), ("vector", "3., 3., 20., 14., 3."),
                   ("pie", "12., 9., 30., 120."), ("cannulus", "12., 9., 3., 7."),
                   ("eannulus", "12., 9., 3., 8., 0.6, 30."),
                   ("rannulus", "12., 9., 2., 7., 0.7, 0."),
                   ("polygon", "3., 3., 20., 5., 12., 15."),
                   ("pannulus", "2., 8., 6., 16., 6., 16., 12., 8., 12.")):
    mx("ij_" + base, f"{base} (I, J, {args}) ? 1 : 0")
    mx("ji_" + base, f"{base} (J, I, {args}) ? 1 : 0", dims="18,24")
mx("cols", 'cols ("3-5,9,12-20x2") ? 1 : 0')
mx("cols_ij", 'cols (I, "3-5,9") ? 1 : 0')
mx("lines", 'lines ("3-5,9,12-16x2") ? 1 : 0')
mx("lines_ij", 'lines (J, "3-5,9") ? 1 : 0')
mx("lines_bad", 'lines ("a") ? 1 : 0')
mx("value", "circle (12., 9., 5.) ? 7 : 2")
mx("big_value", "circle (12., 9., 5.) ? 70000 : 0")
mx("depth", "circle (12., 9., 5.) ? 7 : 0", depth=2)
mx("depth_big", "circle (12., 9., 5.) ? 70000 : 0", depth=8)
mx("bool", "circle (12., 9., 5.)")
mx("not", "! cannulus (12., 9., 3., 7.) ? 1 : 0")
mx("and", "circle (10., 9., 5.) && circle (14., 9., 5.) ? 1 : 0")
mx("or", "circle (6., 9., 3.) || box (15., 5., 20., 12.) ? 3 : 0")
mx("sum", "(circle (10., 9., 5.) ? 1 : 0) + (circle (14., 9., 5.) ? 2 : 0)")
mx("coord", "I + 100 * J")
mx("scalar", "5")
mx("zero", "0")
mx("real", "I / 2.0")
mx("real_scalar", "1.5")
mx("negative", "I - 10")
mx("cube", "circle (3., 3., 2.) ? K : 0", dims="6,5,3")
mx("oned", "I > 3", dims="8")
mx("image", "i > 150 ? 1 : 0", refimages="ramp.fits")
mx("image_keyword", "i > i.EXPTIME * 10 ? 1 : 0", refimages="ramp.fits")
mx("image_circle", "(i > 100 || circle (5., 5., 3.)) ? 1 : 0", refimages="ramp.fits")
mx("image_short", "i > 1050 ? 1 : 0", refimages="short.fits")
mx("image_section", "i > 150 ? 1 : 0", refimages="ramp.fits[3:12,4:9]")
mx("image_mask", "(i > 150 || m > 0) ? 2 : 0", refimages="ramp.fits", refmasks="t_lines.pl")
mx("mask_only", "m", refmasks="t_lines.pl")
mx("mask_times", "m * 2", refmasks="t_lines.pl")
mx("mask_keyword", "m > 2 ? 1 : 0", refimages="ramp.fits", refmasks="t_small.pl")
mx("mask_fits", "m > 0", refmasks="short.fits")
mx("macro", "cmpie (12., 9., 7., 0., 30.) ? 1 : 0", exprdb="regions.db")
mx("macro_const", "circle (12., 9., RADIUS) ? 1 : 0", exprdb="regions.db")
mx("file", "@expr.dat")
mx("unknown", "nosuch (3.) ? 1 : 0")
mx("wrong_args", "circle (3., 4.) ? 1 : 0")
mx("polygon_few", "polygon (3., 4., 5., 6.) ? 1 : 0")
mx("no_image", "i > 3")
