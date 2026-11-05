"""The cases of IRAF's disptrans gabarito, on ../iraf_dopcor's spectrum (three
apertures, linear in Angstroms from 4010)."""

CASES = {
    "nm": dict(units="nm"),
    "microns": dict(units="microns"),
    "inverse": dict(units="inverse centimeters"),
    "inverse_lin": dict(units="inverse centimeters", linearize=True),
    "velocity": dict(units="km/s 4100 angstroms"),
    "hertz_lin": dict(units="hertz", linearize=True),
    "logang": dict(units="log angstroms"),
    "air2vac": dict(units="angstroms", air="air2vac"),
    "vac2air": dict(units="angstroms", air="vac2air", t=20.0, p=700.0, f=10.0),
    "loose": dict(units="inverse cm", error=0.5),
}
