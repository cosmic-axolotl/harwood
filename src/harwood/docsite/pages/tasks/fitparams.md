Fits the transformation equations -- how an instrumental magnitude depends
on the standard magnitude, the colour and the airmass -- to the standard
stars of the night, and appends the coefficients to a parameters file that
[invertfit](task:invertfit) reads.

```console
$ harwood fitparams observations.obsfile catalogo.mkcatalog configuration.mkconfig fitfile.fitparams --logfile fitparams.log --log-fit --log-results
```

**The configuration is the model.** Nothing about a photometric system is
built in. The configuration file, in IRAF's language, says which column of
the catalog and of the observations holds which variable, and writes each
equation, for instance

```text
transformation
fit   v1 = 25.0, v2 = -0.03, v3 = 0.18
VFIT : mV = V + v1 + v2 * (B - V) + v3 * XV
```

`const` fixes a parameter, `delta` sets the step of its numerical
derivative, `set` names an expression to use in the others, and
`derivative`, `error` and `weight` equations do what IRAF's do. The same
file works in IRAF and in harwood, both ways. [mkconfig](task:mkconfig) makes one
from a standards catalog's format file and a template of equations, and
[chkconfig](task:chkconfig) says what is wrong with one, line by line.

**Matching.** Each observation is paired with its star in the catalog by
the id in the first column, upper-cased and reduced to letters, digits, `+`,
`-` and `_`. Observations with no catalog line -- the program stars, or a
standard misspelt -- take no part, and are listed.

**The fit.** Each equation is fitted alone by IRAF's nlfit
(Levenberg-Marquardt) from the initial values in the configuration.
`--weighting uniform` counts every star the same; `photometric` by the
inverse variance of the magnitudes in the equation, and `--addscatter` fits
an extra scatter term so that stars with underestimated errors do not
dominate; `equations` uses the configuration's weight equations.
`--nreject` passes of rejection at `--low-reject`/`--high-reject` times the
rms refit from the initial values each time; they need five stars.

**Identical to IRAF.** The parameters file and the log agree with IRAF's
to the last digit printed, in every weighting, with rejection, and with the
whole configuration language exercised. The interactive graph is not
ported.
