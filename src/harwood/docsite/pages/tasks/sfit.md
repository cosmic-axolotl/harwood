Fits a curve to each spectrum -- icfit's, line by line and band by band -- and writes
the fit, the data, their difference or their ratio in its place.

```console
$ harwood sfit a f --order 6 --logfiles ""
ok sfit in 0.06s
  -> f.fits
$ harwood continuum a --listonly --function chebyshev --order 2 --lines 2 --logfiles STDOUT
...
Line 2 of a.fits:

      coeff           value
        1          -3.1109E3
        2          9.9446E-1
```

`--type` is what replaces each fitted spectrum in the output: `fit`, `data` (with
`--replace`, its rejected points replaced by the fit), `difference` (data minus
fit) or `ratio` (data over fit, and 0 where the fit is 0). The output is a copy of the
input with those spectra replaced; an output that exists has them replaced in it, and
the input itself as output fits it in place. `--listonly` writes nothing.

`--lines` are the image's lines -- its apertures, in order, `1,3:5` or `1-3` -- and
`--bands` its bands: 1 the spectrum, 2 the raw spectrum, 3 the sky, 4 the sigma, as
apall writes them. The header's SFIT and SFITB say which have been fitted; what SFIT
says of the input or of an existing output is not fitted again unless `--override`,
so a spectrum fitted by hand is not overwritten by a later run.

The fit is icfit's: `--function` (spline3, legendre, chebyshev, spline1) of `--order`
(terms, or a spline's pieces), over `--sample` ranges in x, `--naverage` points
averaged into each one fitted (with a minus, their median), and `--niterate` passes
rejecting what is `--low-reject` or `--high-reject` sigma away, with what lies within
`--grow` of it. x is the wavelength, or the pixel without `--wavescale`; `--logscale`
fits the log10 of both, and the log of what is zero or negative is 0. For a legendre or
a chebyshev, `--logfiles` (STDOUT, files, or empty for none) get the coefficients of
the fit as a power series, as IRAF prints them.

`--interactive` opens the icfit panel for each spectrum, with IRAF's keys; what is set
there -- the order, the function, the sample -- carries on to the next spectrum, as
IRAF's does. `--cursor` reads the keys from a file instead, each spectrum to its `q`.
IRAF asks before each spectrum whether to show it; harwood shows each.

IRAF 2.18's sfit and continuum are held to the bit: forty-five sessions -- every output
type, function and order, rejection with and without grow, the rejected points replaced
in each of icfit's ways (all the points, a sample, averaged, a sample averaged), samples, averages and
medians, pixels for x, logs of both axes and of what has no log, division by zero, the
lines and bands to fit, SFIT in the input and in an existing output, override, in
place, listonly, the log to a file, and what IRAF warns of or refuses -- write the same
log line for line, and leave every image in the folder with the same pixels, every one,
and the same SFIT and SFITB. A ratio or difference with `--replace` and a sample that
leaves out some points and is averaged as well makes IRAF's ic_clean read past the end
of the sample; that one is not held to it.
