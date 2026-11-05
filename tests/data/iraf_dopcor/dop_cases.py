"""The cases of IRAF's dopcor gabarito: each a chain of runs, the first on `in`, each
next on the output before it, o1, o2..."""

CASES = {
    "number": [dict(redshift="0.01", verbose=True)],
    "velocity": [dict(redshift="300", isvelocity=True, verbose=True)],
    "flux": [dict(redshift="0.02", flux=True, verbose=True)],
    "factor": [dict(redshift="0.02", flux=True, factor=1.0, dispersion=False,
                    verbose=True)],
    "keyword": [dict(redshift="Z", verbose=True)],
    "minuskey": [dict(redshift="-Z", verbose=True)],
    "minusvel": [dict(redshift="-VHELIO", isvelocity=True, verbose=True)],
    "minusnum": [dict(redshift="-0.01", verbose=True)],
    "plusnum": [dict(redshift="+0.01", verbose=True)],
    "apertures": [dict(redshift="0.01", flux=True, apertures="1,3", verbose=True)],
    "add": [dict(redshift="0.01"), dict(redshift="0.005", add=True, verbose=True)],
    "replace": [dict(redshift="0.01"), dict(redshift="0.005", verbose=True)],
    "twice": [dict(redshift="0.01"), dict(redshift="0.01", add=True)],
    "impossible": [dict(redshift="-1.5")],
    "missing": [dict(redshift="NOPE")],
    "fast": [dict(redshift="400000", isvelocity=True)],
}
