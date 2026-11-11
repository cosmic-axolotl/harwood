A function fitted to every line or column of an image, as IRAF's `imfit.fit1d`
does it, and the fit, the data minus the fit, or the data over it written.
[background](task:background) is fit1d with the difference.

```console
$ harwood fit1d sky.fits skyfit.fits fit --order 5
$ harwood fit1d flat.fits flatnorm.fits ratio --axis 2 --function chebyshev --order 3
$ harwood fit1d frame.fits clean.fits difference --bpm bpm.fits --function legendre --order 3
```

The fit is ICFIT's (`--function`, `--order`, `--sample`, `--naverage`, rejection
with `--low-reject`, `--high-reject`, `--niterate`, `--grow`) along `--axis` 1 (each
line) or 2 (each column). It is set up on the first line and only re-solved on
the next ones, as IRAF calls it: a line after one that rejected no points goes
through the earlier fit's factorization -- with a mask, the earlier line's
weights -- and one after a rejection is fitted afresh. `--bpm` leaves masked
pixels out.

A new output is real, the size of the whole input, zeros outside a section (ones
for a ratio); an existing one is written in place, only where the section is.
`--interactive` asks for the lines to look at ('Fit line =', or `--lines`) and
fits each in the icfit window or from `--cursor`; what is set there fits the
image. On 23 synthetic cases -- every output, function, rejection and averaging,
both axes, masks, sections, an existing image, in place, integers, and by hand
from cursor files -- the output is IRAF 2.18's pixel for pixel.
