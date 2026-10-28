Turns instrumental magnitudes into standard ones, for the standards and the
program stars alike, with the coefficients [fitparams](task:fitparams)
found.

```console
$ harwood invertfit observations.obsfile configuration.mkconfig fitfile.fitparams observations.magcalibrada --catalogs catalogo.mkcatalog
```

**Turning the equations round.** The equations give the instrumental
magnitude from the standard one; a star of unknown magnitude needs them the
other way. With as many equations as unknowns, harwood solves them star by
star as IRAF does, by minimising the equations' rms from a starting point:
the catalog's mean values when `--catalogs` are given, zero otherwise.
With catalogs, each standard also gets a `resid` column beside each
magnitude, its difference from the catalog, so how well the night
calibrated can be read straight off the file; `--objects program` or
`standards` writes only those.

**Errors.** `--errors obserrors` (the default) moves each measured
magnitude by its error and adds the changes in the solution in quadrature;
`equations` does the same with the configuration's error equations;
`undefined` writes none. Neither includes the uncertainty of the
coefficients themselves.

**The output** is IRAF's text file: `--print` adds variables or set
equations before the results, and `--format` takes an IRAF printf format
for each line.

**Identical to IRAF** to the seventh decimal, in every error mode, with and
without catalogs, for INDEF magnitudes and unmatched stars.
