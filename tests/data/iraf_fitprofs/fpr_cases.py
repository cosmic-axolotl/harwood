"""The cases of IRAF's fitprofs gabarito: its parameters, verbose to STDOUT and no log
file unless the case says otherwise; `o` already exists in the cases that start with
`pre` (fitprofs' own output of `a`, aperture 1, made first by the case "pre").

IRAF 2.18.1's fitprofs corrupts its memory when it writes an output verbosely and the
apertures have dispersions of their own: smw_swattrs turns the output's equispec into
multispec and frees the old structure, which the verify line's shdr still holds and
smw_close frees again ("Memory fwa has been corrupted"; the image is never written).
The cases that write an image are quiet, but for `fit_verbose`, whose image IRAF
never writes; and merge ends in a segmentation violation (`pre_merge`, refused)."""

CASES = {
    "gauss": dict(input="a", region="4080 4120", positions="p1", gfwhm=5.0),
    "gauss_one": dict(input="a", lines="2", region="4080 4120", positions="p1",
                      gfwhm=5.0),
    "blend": dict(input="a", region="4165 4210", positions="p2", gfwhm=5.0),
    "blend_single_positions": dict(input="a", region="4165 4210", positions="p2",
                                   gfwhm=5.0, fitpositions="single"),
    "blend_fixed_positions": dict(input="a", region="4165 4210", positions="p2",
                                  gfwhm=5.0, fitpositions="fixed"),
    "blend_single_gfwhm": dict(input="a", region="4165 4210", positions="p2",
                               gfwhm=5.0, fitgfwhm="single"),
    "blend_fixed_gfwhm": dict(input="a", region="4165 4210", positions="p2",
                              gfwhm=5.0, fitgfwhm="fixed"),
    "blend_nobackground": dict(input="a", region="4165 4210", positions="p2",
                               gfwhm=5.0, fitbackground=False),
    "lorentzian": dict(input="a", region="4240 4280", positions="p1", lfwhm=6.0,
                       profile="lorentzian", fitpositions="all"),
    "lorentzian_at": dict(input="a", region="4240 4280", positions="p3",
                          components="4"),
    "voigt": dict(input="a", region="4310 4355", positions="p4"),
    "voigt_fixed_lfwhm": dict(input="a", region="4310 4355", positions="p4",
                              fitlfwhm="fixed"),
    "voigt_default": dict(input="a", region="4310 4355", positions="p1", gfwhm=6.0,
                          lfwhm=5.0, profile="voigt"),
    "mixed": dict(input="a", region="4085 4275", positions="p3"),
    "mixed_components": dict(input="a", region="4085 4275", positions="p3",
                             components="2-3"),
    "absorption": dict(input="one", region="4130 4170", positions="p5"),
    "background_values": dict(input="a", region="4080 4120", positions="p1",
                              gfwhm=5.0, background="930 960"),
    "background_avg": dict(input="a", region="4080 4120", positions="p1", gfwhm=5.0,
                           background="avg(4070,4085) med(4115,4130)"),
    "background_scaled": dict(input="a", region="4080 4120", positions="p1",
                              gfwhm=5.0, background="avg(4070,4085,0.98) 960"),
    "background_one": dict(input="a", region="4080 4120", positions="p1", gfwhm=5.0,
                           background="950"),
    "errors": dict(input="a", region="4165 4210", positions="p2", gfwhm=5.0,
                   nerrsample=60, sigma0=6.0, invgain=0.05),
    "errors_voigt": dict(input="a", region="4310 4355", positions="p4",
                         nerrsample=30, sigma0=6.0, invgain=0.05),
    "errors_few": dict(input="a", region="4080 4120", positions="p1", gfwhm=5.0,
                       nerrsample=10, sigma0=6.0, invgain=0.05),
    "fit": dict(verbose=False, input="a", region="4080 4120", positions="p1", gfwhm=5.0,
                output="o"),
    "difference": dict(verbose=False, input="a", region="4165 4210", positions="p2", gfwhm=5.0,
                       output="o", option="difference"),
    "pre": dict(verbose=False, input="a", lines="1", region="4080 4120", positions="p1",
                gfwhm=5.0,
                output="o"),
    "fit_verbose": dict(input="a", region="4080 4120", positions="p1", gfwhm=5.0,
                        output="o"),
    "pre_exists": dict(input="a", lines="2", region="4080 4120", positions="p1",
                       gfwhm=5.0, output="o"),
    "pre_clobber": dict(input="a", lines="2", region="4080 4120", positions="p1",
                        gfwhm=5.0, output="o", clobber=True),
    "pre_merge": dict(input="a", lines="2", region="4080 4120", positions="p1",
                      gfwhm=5.0, output="o", clobber=True, merge=True),
    "complement": dict(input="a", lines="!1", region="4080 4120", positions="p1",
                       gfwhm=5.0),
    "bands": dict(verbose=False, input="c", region="4080 4120", positions="p1", gfwhm=5.0,
                  output="o", bands="1,3"),
    "two": dict(verbose=False, input="a,one", region="4130 4170", positions="p5",
                output="o1,o2"),
    "quiet_log": dict(input="a", region="4080 4120", positions="p1", gfwhm=5.0,
                      verbose=False, logfile="logfile"),
    "few_points": dict(input="a", region="4100 4102", positions="p1", gfwhm=5.0),
    "no_apertures": dict(input="a", lines="5", region="4080 4120", positions="p1"),
    "bad_background": dict(input="a", region="4080 4120", positions="p1",
                           background="foo(1,2)"),
    "no_positions": dict(input="a", region="4080 4120", positions=""),
    "empty_positions": dict(input="a", region="4080 4120", positions="empty"),
    "lists": dict(input="a,one", output="o1,o2,o3", region="4080 4120",
                  positions="p1"),
}
#: Cases whose image IRAF does not write: it fails after saying what it did.
CRASHES = {"fit_verbose", "pre_merge"}
#: Made before the case, in its folder: the existing output.
BEFORE = {name: "pre" for name in CASES if name.startswith("pre_")}
