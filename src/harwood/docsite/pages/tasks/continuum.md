Fits the continuum of spectra and divides by it -- or subtracts it, or writes it --
rejecting the absorption lines below it as it goes.

```console
$ harwood continuum a o --function legendre --order 4 --lines 1 --logfiles STDOUT

SFIT: harwood 0.2.0.dev0 costamands@costa.mands Nov  5 23:41

Line 1 of a.fits:

      coeff           value
        1           9.5385E5
        2          -6.9713E2
        3          1.6971E-1
        4         -1.3746E-5


END: harwood 0.2.0.dev0 costamands@costa.mands Nov  5 23:41

  -> o.fits
$ harwood continuum @night @night --order 6
$ harwood continuum star.ms star.ms --interactive
```

continuum is sfit with the defaults of a continuum: `--type ratio`, a spline3, and
ten passes rejecting what is 2 sigma *below* the fit and nothing above it, so the
lines fall out and the continuum stays. Everything sfit does, it does; see
[sfit](task:sfit) for the parameters, the lines and bands, SFIT, and how it is held to
IRAF's.
