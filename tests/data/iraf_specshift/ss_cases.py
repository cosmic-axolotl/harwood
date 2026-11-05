"""The cases of IRAF's specshift gabarito."""

#: Each case: a dopcor first or not, then specshift's parameters.
CASES = {
    "shift": (None, dict(shift=5.5, verbose="yes")),
    "aperture": (None, dict(shift=-2.0, apertures="2", verbose="yes")),
    "quiet": (None, dict(shift=1.25, verbose="no")),
    "redshifted": (0.01, dict(shift=3.0, verbose="yes")),
}
