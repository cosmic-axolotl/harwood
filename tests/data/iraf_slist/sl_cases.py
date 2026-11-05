"""The cases of IRAF's slist gabarito."""

CASES = {
    "short": dict(images="in,pix"),
    "long": dict(images="in,pix", long_header=True),
    "apertures": dict(images="in", apertures="1,3"),
    "missing": dict(images="nope,in"),
}
