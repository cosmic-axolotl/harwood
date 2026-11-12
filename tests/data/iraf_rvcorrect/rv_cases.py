"""The cases of the rvcorrect gabarito: the task's parameters, by how the
observations come (files, images, or the parameters themselves)."""

CL = {"year": 1985, "month": 12, "day": 19, "ut": 3.35, "ra": 1.628, "dec": 13.145,
      "epoch": 1950.0, "vobs": 780.0}

CASES = {
    "files": {"files": "obs.txt"},
    "files_input": {"files": "obs.txt", "input": "yes"},
    "files_noheader": {"files": "obs.txt,more.txt", "header": "no"},
    "files_ctio": {"files": "obs.txt", "observatory": "ctio", "input": "yes"},
    "files_lasilla": {"files": "more.txt", "observatory": "lasilla"},
    "files_vsun": {"files": "obs.txt", "vsun": 15.5, "ra_vsun": 17.8, "dec_vsun": 28.0,
                   "epoch_vsun": 1950.0},
    "files_missing": {"files": "nothere.txt,more.txt"},
    "cl": dict(CL),
    "cl_ctio": {**CL, "observatory": "ctio"},
    "cl_noepoch": {**CL, "epoch": "INDEF", "vobs": 0.0},
    "cl_south": {"year": 2026, "month": 11, "day": 12, "ut": 1.5, "ra": 5.5,
                 "dec": -69.75, "epoch": 2000.0, "observatory": "lna", "vobs": 260.0},
    "images": {"images": "im1.fits,im2.fits,im3.fits,im4.fits"},
    "images_input": {"images": "im1.fits,im3.fits", "input": "yes"},
    "images_update": {"images": "im1.fits,im2.fits,im3.fits,im4.fits", "imupdate": "yes"},
    "images_noepoch": {"images": "noepoch.fits,im1.fits"},
}
