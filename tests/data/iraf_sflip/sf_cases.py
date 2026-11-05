"""The cases of IRAF's sflip gabarito: its two switches, and in place."""

CASES = {
    "data": dict(coord_flip=False, data_flip=True),
    "coord": dict(coord_flip=True, data_flip=False),
    "both": dict(coord_flip=True, data_flip=True),
    "neither": dict(coord_flip=False, data_flip=False),
    "inplace": dict(coord_flip=True, data_flip=True, output="in"),
}
