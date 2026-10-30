Works out the transformation equations for every star with the coefficients
[fitparams](task:fitparams) found, without solving anything. It is for
equations written with the standard magnitude on the left:

```text
VFIT : V = mV + v1 + v2 * (mB - mV) + v3 * XV
```

```console
$ harwood evalfit observations.obsfile config.cfg fitfile.fitparams observations.calib --catalogs catalogo.mkcatalog
```

**Which task.** Equations written the other way round, with the instrumental
magnitude on the left and a catalog variable on the right, are
[invertfit](task:invertfit)'s. Given to `evalfit`, they come to INDEF for any
star not in a catalog, and to the *instrumental* magnitude the fit predicts for
a standard, and the run says so.

**The columns** are, for each equation, the value, its error and, with
`--catalogs`, a `resid` for the standards: the left side minus the right. They
are named after the left side, so a colour equation gives `B-V`, `error(B-V)`,
`resid(B-V)`. A program star has INDEF in `resid`, and `--objects program` or
`standards` writes only those.

**Errors.** `--errors obserrors` (the default) moves each observed variable by
its error and adds the changes in quadrature. A variable reached only through a
`set` equation is not moved, as in IRAF, so write it out in the equation if its
error matters. `equations` evaluates the configuration's error equations, and
gives INDEF for an equation that has none; `undefined` writes none. None of them
includes the uncertainty of the coefficients themselves.

**The output** is IRAF's text file: `--print` adds variables or set equations
before the results, and `--format` takes an IRAF printf format for each line.

**Identical to IRAF** to the seventh decimal, in every error mode, with and
without catalogs, for INDEF magnitudes and unmatched stars, with two exceptions
where IRAF has no useful answer:

- An error that takes an expression out of its domain, such as the square root
  of a negative number, is infinite in IRAF and printed as `0.00E309`. Here it
  is INDEF.
- A configuration with no `catalog` section stops IRAF with *Out of memory*. Here
  it runs, and its stars have no id.
