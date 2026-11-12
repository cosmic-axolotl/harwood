"""The cases of the surface gabarito: the image (iraf_contour's ct_inputs.py makes
them) and surface's parameters."""

CASES = {
    "default": ("gal.fits", {}),
    "angles": ("gal.fits", {"angh": 40.0, "angv": 60.0}),
    "behind": ("gal.fits", {"angh": 150.0, "angv": 15.0}),
    "below": ("gal.fits", {"angh": -120.0, "angv": -20.0}),
    "steep": ("tiny.fits", {"angh": 10.0, "angv": 85.0, "xres": 0, "yres": 0}),
    "floor_ceiling": ("gal.fits", {"floor": 200.0, "ceiling": 900.0}),
    "swapped": ("gal.fits", {"floor": 900.0, "ceiling": 200.0}),
    "degenerate": ("gal.fits", {"floor": 500.0, "ceiling": 500.0}),
    "above_all": ("gal.fits", {"floor": 1e6}),
    "below_all": ("gal.fits", {"ceiling": -1e6}),
    "flat": ("flat.fits", {}),
    "label": ("gal.fits", {"label": "yes"}),
    "label_behind": ("gal.fits", {"label": "yes", "angh": 150.0, "angv": -15.0}),
    "fullres": ("tiny.fits", {"xres": 0, "yres": 0}),
    "subsample": ("gal.fits", {"subsample": "yes"}),
    "nopreserve": ("gal.fits", {"preserve": "no", "xres": 40, "yres": 20}),
    "title": ("gal.fits", {"title": "My surface"}),
    "section": ("gal.fits[60:140,50:110]", {}),
}
