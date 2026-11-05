Combines spectra: resamples them onto one dispersion, scales them, rejects what
does not belong, and averages, medians or sums what is left.

```console
$ harwood scombine a,b,c,d,e o --reject avsigclip --scale median

Nov  5 13:47: SCOMBINE
  combine = average, scale = median, zero = none, weight = none
  reject = avsigclip, mclip = yes, nkeep = 1
  lsigma = 3., hsigma = 3.
  blank = 0.
                Images   Median  Scale
                 a[  1]  818.69  1.094
                 b[  1]  1149.6  0.779
                 c[  1]  968.71  0.925
                 d[  1]   634.9  1.411
                 e[  1]  907.92  0.987

  Output image = o, ncombine = 5
  w1 = 4007., w2 = 4308.49, dw = 1.945098, nw = 156., dtype = 0
$ harwood scombine @night o --group all --combine median --noutput n
$ harwood scombine star1,star2 o --w1 4000 --w2 5000 --dw 1 --log
```

`--group apertures` (the default) combines each aperture number across the
inputs into one aperture of the output; `--group all` every spectrum into one;
`--group images` the apertures of each input into an output of its own, as many
outputs as inputs. `--apertures` chooses which take part.

The output's dispersion is the first spectrum's with `--first`; otherwise it runs
over all the inputs' wavelengths in the smallest of their steps, unless
`--w1`, `--w2`, `--dw` and `--nw` say otherwise (any three of them fix the fourth),
evenly in log wavelength with `--log`. If any input has no wavelengths, the output
is in pixels. Each spectrum is resampled onto it, the mean of its interpolant
(`--interp`, poly5) over each pixel, and the pixels it covers by less than 1% of
one of its own are left out of the combining.

`--scale`, `--zero` and `--weight` are `none`, the spectrum's `mode`, `median`
or `mean` (over the `--sample` regions, `4100:4200,4250:4300`, if given),
`exposure` (not for zero), `@file` with a value per spectrum, or `!KEYWORD` from
the header; they are made relative to their mean before combining, as the log
says. `--lthreshold` and `--hthreshold` leave values out before anything else.

`--reject` is `minmax` (`--nlow` and `--nhigh`), `sigclip` or `avsigclip` (a sigma
from the values at the pixel, or pooled over the spectrum, about the median with
`--mclip` or the average without), `ccdclip` and `crreject` (the CCD noise model,
`--rdnoise`, `--gain`, `--snoise`, each a number or a header keyword) and `pclip`
(the sigma from the median to the value `--pclip` places from it). `--nkeep`
keeps at least that many, or rejects at most that many when negative;
`--grow` rejects that many pixels either side of a rejected one, in the same
spectrum. Where nothing is left the output is `--blank`. `--noutput` writes the
number combined at each pixel. The header gets NCOMBINE and, when exposures were
used, EXPTIME.

IRAF's onedspec scombine is held to the bit: fifty-one sessions IRAF 2.18 ran --
every grouping, the grid from the inputs and given, linear and logarithmic, the
first's, every rejection about the median and the average, values put back,
grow, every scaling, zero and weight from statistics, exposures, keywords and a
file, a sample, thresholds, spectra that cover part of the output or none of it
-- write the same log line for line and the same pixels, every one. It is an
older icombine than imcombine's, and its ways are kept: values put back after a
sigma clip are looked for among all those rejected; pclip's putting back stops
at the median's place; avsigclip's putting back reuses the pooled sigma's
variable, and so changes the sigma for the pixels after; the median of the
statistics is the n/2-th value; and when three values are sorted in the order
IRAF calls bca, only half their spectra's ids move with them, so that a scaled
sigma clip may take a value's scale from another spectrum.
