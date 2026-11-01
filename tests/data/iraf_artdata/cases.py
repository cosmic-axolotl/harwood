"""What IRAF is asked."""

PRECESS = {
    "doc": {"input": "coords.txt", "startyear": 1950.0, "endyear": 1990.0},
    "std": {"input": "coords.txt", "startyear": 1950.0, "endyear": 1990.0, "stdepoch": 2000.0},
    "j2000": {"input": "coords.txt", "startyear": 2000.0, "endyear": 2026.8},
}
PAT = dict(ncols=9, nlines=7)
MKPATTERN = {}
for pattern in ("constant", "grid", "checker", "coordinates", "slope", "square"):
    for size in (1, 2, 3):
        MKPATTERN[f"{pattern}{size}"] = {
            "pattern": pattern,
            "size": size,
            "v1": 2.5,
            "v2": 7.25,
            **PAT,
        }
    MKPATTERN[f"{pattern}_cube"] = {
        "pattern": pattern,
        "size": 2,
        "v1": 2.5,
        "v2": 7.25,
        "ndim": 3,
        "n3": 2,
        **PAT,
    }
    MKPATTERN[f"{pattern}_line"] = {
        "pattern": pattern,
        "size": 2,
        "v1": 2.5,
        "v2": 7.25,
        "ndim": 1,
        **PAT,
    }
    MKPATTERN[f"{pattern}_tall"] = {
        "pattern": pattern,
        "size": 2,
        "v1": 1.1,
        "v2": 3.3,
        "ncols": 5,
        "nlines": 30,
    }
    MKPATTERN[f"{pattern}_big"] = {"pattern": pattern, "size": 20, "v1": 1.1, "v2": 3.3, **PAT}
    for kind in ("short", "ushort", "integer", "long", "double"):
        MKPATTERN[f"{pattern}_{kind}"] = {
            "pattern": pattern,
            "size": 2,
            "v1": 1.5,
            "v2": 9.5,
            "pixtype": kind,
            **PAT,
        }
    for option in ("replace", "add", "multiply"):
        MKPATTERN[f"{pattern}_{option}_in"] = {
            "input": "base.fits",
            "pattern": pattern,
            "size": 2,
            "v1": 2.5,
            "v2": 7.25,
            "option": option,
        }
        MKPATTERN[f"{pattern}_{option}_out"] = {
            "input": "base.fits",
            "output": "out.fits",
            "pattern": pattern,
            "size": 2,
            "v1": 2.5,
            "v2": 7.25,
            "option": option,
        }
        MKPATTERN[f"{pattern}_{option}_short"] = {
            "input": "bases.fits",
            "output": "out.fits",
            "pattern": pattern,
            "size": 2,
            "v1": 2.5,
            "v2": 7.25,
            "option": option,
        }
    MKPATTERN[f"{pattern}_section"] = {
        "input": "base.fits[3:6,2:4]",
        "pattern": pattern,
        "size": 2,
        "v1": 2.5,
        "v2": 7.25,
        "option": "add",
    }
    MKPATTERN[f"{pattern}_section_out"] = {
        "input": "base.fits[3:6,2:4]",
        "output": "out.fits",
        "pattern": pattern,
        "v1": 2.5,
        "v2": 7.25,
    }
MKPATTERN["title"] = {"title": "my title", **PAT}
MKPATTERN["cards"] = {"header": "cards.dat", **PAT}
MKPATTERN["from_image"] = {"header": "base.fits", **PAT}
MKPATTERN["double_in"] = {
    "input": "based.fits",
    "pattern": "slope",
    "v1": 2.5,
    "v2": 7.25,
    "option": "multiply",
}

SMALL = dict(ncols=20, nlines=15, header="")
MKNOISE = {
    "background": {"background": 100.5, **SMALL},
    "read": {"rdnoise": 10.0, **SMALL},
    "read_gain": {"rdnoise": 10.0, "gain": 2.5, **SMALL},
    "seed": {"rdnoise": 5.0, "seed": 12345, **SMALL},
    "poisson_low": {"background": 5.0, "poisson": "yes", **SMALL},
    "poisson_20": {"background": 25.0, "poisson": "yes", **SMALL},
    "poisson_high": {"background": 1000.0, "poisson": "yes", **SMALL},
    "poisson_gain": {"background": 1000.0, "poisson": "yes", "gain": 3.0, **SMALL},
    "both": {"background": 500.0, "poisson": "yes", "rdnoise": 8.0, "gain": 2.0, **SMALL},
    "on_image": {
        "input": "base.fits",
        "output": "out.fits",
        "rdnoise": 3.0,
        "background": 10.0,
    },
    "on_image_poisson": {"input": "base.fits", "output": "out.fits", "poisson": "yes"},
    "on_short": {"input": "bases.fits", "output": "out.fits", "rdnoise": 3.0},
    "in_place": {"input": "base.fits", "rdnoise": 3.0},
    "rays_file": {"cosrays": "rays.txt", "radius": 0.5, **SMALL},
    "rays_noise": {"cosrays": "rays.txt", "rdnoise": 5.0, "background": 10.0, **SMALL},
    "rays_random": {"ncosrays": 6, "energy": 5000.0, **SMALL},
    "rays_random_gain": {"ncosrays": 6, "energy": 5000.0, "gain": 2.0, **SMALL},
    "rays_random_write": {"ncosrays": 4, "cosrays": "new.txt", **SMALL},
    "rays_wide": {"cosrays": "rays.txt", "radius": 2.0, "ar": 0.5, "pa": 30.0, **SMALL},
    "std_header": {"ncols": 8, "nlines": 6, "header": "artdata$stdheader.dat", "title": "t"},
    "no_comments": {"ncols": 8, "nlines": 6, "comments": "no", "header": ""},
}
OBJECTS = dict(ncols=50, nlines=50, header="", background=0.0, comments="no")
MKOBJECTS = {
    "moffat": {"objects": "stars.txt", **OBJECTS},
    "moffat_wide": {"objects": "stars.txt", "radius": 2.0, "beta": 3.5, **OBJECTS},
    "gaussian": {"objects": "stars.txt", "star": "gaussian", **OBJECTS},
    "gaussian_wide": {"objects": "stars.txt", "star": "gaussian", "radius": 2.5, **OBJECTS},
    "elliptical": {"objects": "stars.txt", "radius": 2.0, "ar": 0.5, "pa": 30.0, **OBJECTS},
    "exptime": {"objects": "stars.txt", "exptime": 30.0, "magzero": 10.0, **OBJECTS},
    "distance": {"objects": "stars.txt", "distance": 1.5, **OBJECTS},
    "offset": {"objects": "stars.txt", "xoffset": 3.5, "yoffset": -2.25, **OBJECTS},
    "background": {
        "objects": "stars.txt",
        "ncols": 50,
        "nlines": 50,
        "header": "",
        "background": 100.0,
        "exptime": 2.0,
        "comments": "no",
    },
    "noise": {
        "objects": "stars.txt",
        "ncols": 50,
        "nlines": 50,
        "header": "",
        "background": 100.0,
        "rdnoise": 5.0,
        "poisson": "yes",
        "gain": 2.0,
        "comments": "no",
    },
    "on_image": {
        "input": "base.fits",
        "output": "out.fits",
        "objects": "stars.txt",
        "comments": "no",
    },
    "on_image_noise": {
        "input": "base.fits",
        "output": "out.fits",
        "objects": "stars.txt",
        "poisson": "yes",
        "rdnoise": 3.0,
        "comments": "no",
    },
    "galaxies": {"objects": "galaxies.txt", **OBJECTS},
    "star_file": {"objects": "stars.txt", "star": "prof.txt", "radius": 4.0, **OBJECTS},
    "star_integral": {"objects": "stars.txt", "star": "grow.txt", "radius": 3.0, **OBJECTS},
    "star_file_shaped": {"objects": "stars.txt", "star": "prof.txt", "radius": 4.0, "ar": 0.6,
                         "pa": 40.0, **OBJECTS},
    "star_image": {"objects": "stars.txt", "star": "psfimg.fits", "radius": 5.0, **OBJECTS},
    "star_image_shaped": {"objects": "stars.txt", "star": "psfimg.fits", "radius": 5.0,
                          "ar": 0.7, "pa": 30.0, **OBJECTS},
    "star_image_small": {"objects": "stars.txt", "star": "psfimg.fits", "radius": 1.5,
                         **OBJECTS},
    "object_image": {"objects": "imgals.txt", **OBJECTS},
    "object_image_gauss": {"objects": "imgals.txt", "star": "gaussian", "radius": 1.5,
                           **OBJECTS},
    "object_image_star_image": {"objects": "imgals.txt", "star": "psfimg.fits", "radius": 2.0,
                                **OBJECTS},
    "object_file": {"objects": "protos.txt", **OBJECTS},
    "object_file_gauss": {"objects": "protos.txt", "star": "gaussian", "radius": 1.5,
                          **OBJECTS},
    "galaxies_gauss": {
        "objects": "galaxies.txt",
        "star": "gaussian",
        "radius": 1.5,
        **OBJECTS,
    },
    "galaxies_noise": {
        "objects": "galaxies.txt",
        "ncols": 50,
        "nlines": 50,
        "header": "",
        "background": 50.0,
        "poisson": "yes",
        "comments": "no",
    },
    "std_header": {
        "objects": "stars.txt",
        "ncols": 50,
        "nlines": 50,
        "header": "artdata$stdheader.dat",
        "title": "x",
        "exptime": 3.0,
        "gain": 2.0,
        "rdnoise": 4.0,
    },
}

STARLIST = {
    "default": {"nstars": 30},
    "sfile": {"nstars": 30, "spatial": "file", "sfile": "density.txt"},
    "sfile_center": {"nstars": 30, "spatial": "file", "sfile": "density.txt", "xcenter": 100.0,
                     "ycenter": 80.0, "nssample": 50, "sorder": 6},
    "lfile": {"nstars": 30, "luminosity": "file", "lfile": "lum.txt", "minmag": 8.0,
              "maxmag": 18.0},
    "lfile_narrow": {"nstars": 30, "luminosity": "file", "lfile": "lum.txt", "minmag": 2.0,
                     "maxmag": 30.0, "nlsample": 40, "lorder": 5},
    "both_files": {"nstars": 30, "spatial": "file", "sfile": "density.txt",
                   "luminosity": "file", "lfile": "lum.txt", "minmag": 8.0, "maxmag": 18.0},
    "uniform": {"nstars": 25, "luminosity": "uniform", "minmag": 12.0, "maxmag": 18.0},
    "seeds": {
        "nstars": 25,
        "sseed": 77,
        "lseed": 99,
        "xmin": 10.0,
        "xmax": 100.0,
        "ymin": -5.0,
        "ymax": 50.0,
    },
    "power": {"nstars": 25, "power": 0.3, "minmag": 10.0, "maxmag": 20.0},
    "hubble": {"nstars": 40, "spatial": "hubble"},
    "hubble_center": {
        "nstars": 40,
        "spatial": "hubble",
        "xcenter": 100.0,
        "ycenter": 300.0,
        "core_radius": 10.0,
        "base": 0.0001,
    },
    "hubble_off": {
        "nstars": 20,
        "spatial": "hubble",
        "xcenter": 900.0,
        "ycenter": 200.0,
        "core_radius": 100.0,
    },
    "salpeter": {"nstars": 40, "luminosity": "salpeter", "minmag": 0.0, "maxmag": 12.0},
    "bands": {"nstars": 40, "luminosity": "bands", "minmag": 5.0, "maxmag": 20.0},
    "bands_order": {
        "nstars": 40,
        "luminosity": "bands",
        "minmag": 5.0,
        "maxmag": 20.0,
        "nlsample": 50,
        "lorder": 5,
    },
}
