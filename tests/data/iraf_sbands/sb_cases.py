"""The cases of IRAF's sbands gabarito, on ../iraf_dopcor's spectrum (4010 to 4370 A),
and the band and filter files they read. `pair` writes its continua without ids, so
IRAF reads its line word after word into one band and a continuum `none`; `lick`
writes them as they should be; `edge` starts before the spectrum."""

BANDS = """# id centre width filter, and continua
band1 4100 20 none
# a comment line
pair 4150 10 none 4130 10 none 4170 10 none
lick 4150 10 none blue 4130 10 none red 4170 10 none
filtered 4200 30 filt.dat
oneside 4250 20 none 4230 10 none
tiny 4290.3 0.4 none
edge 4005 30 none
"""
FILTER = """# wavelength response
4180 0.1
4190 0.6
4200 1.0
4210 0.7
4220 0.2
"""
CASES = {
    "flux": dict(),
    "nonorm": dict(normalize=False),
    "mag": dict(mag=True, magzero=20.0),
    "apertures": dict(apertures="1,3", verbose=False),
}
