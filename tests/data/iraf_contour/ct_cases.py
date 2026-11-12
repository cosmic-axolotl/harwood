"""The cases of the contour gabarito: the image and contour's parameters."""

CASES = {
    "default": ("gal.fits", {}),
    "noperimeter": ("gal.fits", {"perimeter": "no"}),
    "ncontours": ("gal.fits", {"ncontours": 5}),
    "interval": ("gal.fits", {"interval": 250.0}),
    "floor_ceiling": ("gal.fits", {"floor": 200.0, "ceiling": 1500.0}),
    "same_floor_ceiling": ("gal.fits", {"floor": 0.0, "ceiling": 0.0, "ncontours": 8}),
    "zero": ("gal.fits", {"zero": 150.0}),
    "nhi0": ("gal.fits", {"nhi": 0}),
    "nhi1": ("tiny.fits", {"nhi": 1, "xres": 0, "yres": 0}),
    "fullres": ("tiny.fits", {"xres": 0, "yres": 0}),
    "subsample": ("gal.fits", {"subsample": "yes"}),
    "nopreserve": ("gal.fits", {"preserve": "no", "xres": 50, "yres": 20}),
    "nolabel": ("gal.fits", {"label": "no"}),
    "negative": ("neg.fits", {}),
    "small": ("small.fits", {}),
    "large": ("large.fits", {}),
    "title": ("gal.fits", {"title": "My contours"}),
    "section": ("gal.fits[50:150,40:120]", {}),
    "flat": ("flat.fits", {}),
}
