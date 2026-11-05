Operates on spectra: copies them, cuts them to a wavelength region, puts them
on an even grid, and does arithmetic on them -- alone, with a number, or with
another spectrum resampled onto theirs.

```console
$ harwood sarith in copy o --w1 4100 --w2 4200 --verbose
in[*,1](1)  -->  o[*,1](1)
in[*,2](2)  -->  o[*,2](2)
$ harwood sarith in - o --input2 other --verbose
in[*,1](1)  -  other[*,1](1)  -->  o[*,1](1)
$ harwood sarith star.fits / ratio.fits --input2 star2.fits --ignoreaps
$ harwood sarith in mag o --errval 99
```

The operations of one spectrum are `abs`, `copy`, `dex` (10^x), `exp`, `flam`
and `fnu` (F-nu to F-lambda and back, c/λ²), `inv`, `ln`, `log`, `lum`
(10^(-0.4 x)), `mag` (-2.5 log10) and `sqrt`; with `--input2`, a spectrum or a
number, `replace`, `+`, `-`, `*`, `/` and `^`, `--reverse` putting the second
first. A second spectrum is resampled onto the first's pixels -- the mean of the
interpolant (`--interp`, poly5) over each -- and paired by aperture number, or
by line with `--ignoreaps`; an aperture it does not have is copied. The error
band is only ever copied. What cannot be computed, a division by zero or the
log of a negative, is `--errval`.

`--w1` and `--w2` cut a region: resampled onto as many pixels as it covered,
from one wavelength exactly to the other (`--w1` above `--w2` turns it round),
or with `--no-rebin` at the nearest whole pixels. `--apertures`, `--beams` and
`--bands` (1 the spectrum, 2 raw, 3 the sky, 4 the error, IRAF's order of
apall's bands) choose, a `!` before taking the others; `--apmodulus` takes the
aperture numbers modulo it; `--renumber` numbers the output's from 1 and
`--offset` adds to them. Every input goes into one output spectrum, or with as
many outputs as inputs one each; `--format onedspec` writes a spectrum per
aperture, `o.0001`, `o.0002`. An output that exists is refused unless
`--clobber`, and `--merge` adds to it, an aperture it has replaced.

Everything is computed in real, as IRAF's: thirty-two sessions IRAF 2.18 ran
-- copies, regions resampled and not and backwards, every operation, constants
and spectra on other dispersions, unmatched apertures, bands, onedspec's files,
an output that exists, a merge, two inputs with and without renumbering -- say
the same line for line, and of 17354 pixels 96% are IRAF's to the bit, the rest
a unit of the last place of the operands. IRAF's ways kept: over exactly whole
pixels the resampling sums them and divides by one fewer, and the output's
wavelengths after a cut come from the first one and the step in real.
