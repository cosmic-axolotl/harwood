"""What IRAF is asked: (name, task parameters)."""

LISTPIXELS = {}
for image in ("real", "short", "double", "long", "oned", "big", "tiny"):
    LISTPIXELS[image] = {"images": f"{image}.fits"}
LISTPIXELS.update({
    "cube": {"images": "cube.fits[*,1:2,*]"},
    "cube_plane": {"images": "cube.fits[*,*,2]"},
    "cube_row": {"images": "cube.fits[*,2,2]"},
    "section": {"images": "real.fits[2:3,2:3]"},
    "flip": {"images": "real.fits[-*,1:1]"},
    "flipxy": {"images": "real.fits[3:2,3:2]"},
    "step": {"images": "real.fits[*:2,*:2]"},
    "line": {"images": "real.fits[*,2]"},
    "column": {"images": "real.fits[2,*]"},
    "pixel": {"images": "real.fits[2:2,3:3]"},
    "verbose": {"images": "real.fits[1:2,1:1]", "verbose": "yes", "wcs": "physical"},
    "list": {"images": "real.fits[1:2,1:1],short.fits[1:1,1:2]", "verbose": "yes"},
    "emptywcs": {"images": "real.fits[1:2,1:1]", "wcs": ""},
    "physical": {"images": "real.fits[2:3,2:3]", "wcs": "physical"},
    "physical_step": {"images": "real.fits[*:2,2:3]", "wcs": "physical"},
    "physical_flip": {"images": "real.fits[-*,2:3]", "wcs": "physical"},
    "world_none": {"images": "real.fits[2:3,2:3]", "wcs": "world"},
    "tan": {"images": "tan.fits[1:2,1:2]"},
    "tan_physical": {"images": "tan.fits[1:2,1:2]", "wcs": "physical"},
    "tan_world": {"images": "tan.fits", "wcs": "world"},
    "tan_section": {"images": "tan.fits[2:3,2:3]", "wcs": "world"},
    "tan_flip": {"images": "tan.fits[-*,*:2]", "wcs": "world"},
    "tan_line": {"images": "tan.fits[*,2]", "wcs": "world"},
    "tan_column": {"images": "tan.fits[3,*]", "wcs": "world"},
    "tilt": {"images": "tilt.fits", "wcs": "world"},
    "tilt_physical": {"images": "tilt.fits", "wcs": "physical"},
    "linear": {"images": "linear.fits", "wcs": "world"},
    "linear_line": {"images": "linear.fits[*,2]", "wcs": "world"},
    "linear_column": {"images": "linear.fits[2,*]", "wcs": "world"},
    "linear_pixel": {"images": "linear.fits[2:2,2:2]", "wcs": "world"},
    "linear_line_phys": {"images": "linear.fits[*,2]", "wcs": "physical"},
    "shear": {"images": "shear.fits", "wcs": "world"},
    "shear_section": {"images": "shear.fits[2:3,*:2]", "wcs": "world"},
    "shear_line": {"images": "shear.fits[*,2]", "wcs": "world"},
    "cdelt": {"images": "cdelt.fits", "wcs": "world"},
    "phys": {"images": "phys.fits[1:2,1:2]", "wcs": "physical"},
    "phys_world": {"images": "phys.fits[1:2,1:2]", "wcs": "world"},
    "spec": {"images": "spec.fits", "wcs": "world"},
    "spec_section": {"images": "spec.fits[2:3]", "wcs": "world"},
    "spec_physical": {"images": "spec.fits[2:3]", "wcs": "physical"},
    "tanwat": {"images": "tanwat.fits[1:2,1:2]", "wcs": "world"},
    "tanwat_logical": {"images": "tanwat.fits[1:2,1:2]"},
    "tanwat_given": {"images": "tanwat.fits[1:2,1:2]", "wcs": "world", "formats": "%10.3f"},
    "specwat": {"images": "specwat.fits[1:2]", "wcs": "world"},
    "specwat_physical": {"images": "specwat.fits[1:2]", "wcs": "physical"},
})
FORMATS = ["%5.1f %6.2f", "%5.1f", "%5.1f %5.1f %5.1f", "%d %d", "%5d %5d", "%g %e",
           "%12.4f %-12.1e", "%012.4f %8.2f", "%10g %0.3g", "%10.3e %f", "%h %h", "%H %h",
           "%m %m", "%M %m", "%.2h %.3h", "%12.2H %-12.1h", "%.0h %.0m", "%f xyz",
           "abc %f", "%s %s", "%.0f %.0e", "%.0g %.1g", "%3.0f %-4d"]
for k, text in enumerate(FORMATS):
    LISTPIXELS[f"format{k}"] = {"images": "tan.fits[1:2,1:2]", "wcs": "world", "formats": text}

MINMAX = {}
for image in ("f", "fs", "fd", "fl", "fb", "f1", "f3", "flat", "fine", "huge", "fu",
              "fscale", "limits", "lmin", "lmax", "lfine", "lzero", "lequal", "empty"):
    MINMAX[image] = [{"images": f"{image}.fits", "update": "no"}]
    MINMAX[image + "_force"] = [{"images": f"{image}.fits", "update": "no", "force": "yes"}]
    MINMAX[image + "_update"] = [{"images": f"{image}.fits"}, {"images": f"{image}.fits"},
                                 {"images": f"{image}.fits", "force": "yes"}]
MINMAX.update({
    "section": [{"images": "f.fits[2:4,2:4]", "update": "no"}],
    "section_update": [{"images": "f.fits[2:4,2:4]"}],
    "section_valid": [{"images": "f.fits"}, {"images": "f.fits[2:4,2:4]", "update": "no"}],
    "section_valid_force": [{"images": "f.fits"},
                            {"images": "f.fits[2:4,2:4]", "update": "no", "force": "yes"}],
    "flip": [{"images": "f.fits[-*,*]", "update": "no"}],
    "step": [{"images": "f.fits[*:2,*:2]", "update": "no"}],
    "pixel": [{"images": "f.fits[2:2,3:3]", "update": "no"}],
    "line": [{"images": "f.fits[*,4]", "update": "no"}],
    "column": [{"images": "f.fits[3,*]", "update": "no"}],
    "plane": [{"images": "f3.fits[*,*,2]", "update": "no"}],
    "quiet": [{"images": "f.fits", "update": "no", "verbose": "no"}],
    "list": [{"images": "f.fits,fs.fits", "update": "no"}],
    "glob": [{"images": "f?.fits", "update": "no", "verbose": "yes"}],
    "ext1": [{"images": "mef.fits[1]", "update": "no"}],
    "extname": [{"images": "mef.fits[ERR]", "update": "no"}],
    "ext_update": [{"images": "mef.fits[2]"}, {"images": "mef.fits[2]"}],
    "missing": [{"images": "nothere.fits,f.fits", "update": "no"}],
})
