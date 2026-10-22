Combines images pixel by pixel: at each position, the values of all the
frames are scaled to a common level if asked, the deviant ones are
rejected, and what survives is averaged, or its median taken. One engine
does it all, with IRAF's parameter names and meanings -- `combine`,
`reject`, `lsigma`, `nlow`, `rdnoise`, `scale` -- and IRAF's `zerocombine`,
`darkcombine` and `flatcombine` are presets of it, so there is one
implementation and one set of names to learn. It is checked against IRAF's
own output.

```example
title: Stack the aligned frames of the field
after: reduced
run:
  - harwood imalign red/obj001.fits,red/obj002.fits,red/obj003.fits red/obj001.fits bright.coo --output aligned
  - harwood imcombine aligned/obj00*.fits final.fits --combine average --reject sigclip
output: |
  -> final.fits
```

## Which rejection

| `--reject` | the threshold comes from | good for |
|---|---|---|
| `none` | -- | frames already clean |
| `minmax` | nothing: the `nlow` lowest and `nhigh` highest are dropped | a few bias frames |
| `sigclip` | the spread of the stack at that pixel | many frames |
| `avsigclip` | the spread pooled along the image line, scaled like photon noise | a few flats |
| `ccdclip`, `crreject` | the detector: read noise, gain and flat error (`--rdnoise`, `--gain`, `--snoise`) | a few exposures of the sky |
| `pclip` | a percentile of the stack (`--pclip`) | when nothing is known about the noise |

With few frames, `sigclip` has a weakness that `ccdclip` does not: two
cosmic rays on the same pixel widen the spread enough to hide each other,
whereas a threshold from the noise model does not depend on the stack.
`--mclip` (on, as in IRAF) measures deviations from the median rather than
the mean, so one strong outlier does not drag the centre towards itself and
take the whole stack with it. `--grow` also rejects the neighbours of a
rejected pixel, for the faint edge of a cosmic ray.

## Levels, and what is written

`--scale mode` (or `median`, `mean`, `exposure`) brings frames taken
through cloud or for different times to a common level before rejection,
since otherwise the whole of a fainter frame looks deviant; `--zero`
removes an additive offset instead, for a changing sky. `--statsec` says
where to measure them. `--lthreshold`/`--hthreshold` drop saturated pixels
before anything else. `--combine median` is noisier than the average by
about a quarter and almost unmoved by whatever rejection missed.

The combined image names every input in `IMCMB` cards. `--sigmas` writes
the standard deviation at each pixel, `--nrejmasks` how many values were
rejected there; `--ccdtype` and `--subsets` pick frames by type and combine
each filter apart, as the presets do.

## The variance, and what each pixel is worth

When the inputs carry a variance -- as ccdproc's do, once it knows the
gain -- the result carries one too, worked out from the values each pixel
actually kept: the weighted mean's variance for `average`, the sum of them
for `sum`. A median is noisier than the mean, by a factor that reaches π/2
only for many frames: with three it is 1.35, and harwood uses the right one
for each count, with the harmonic mean of the sigmas when the frames differ
(a median is held by its sharpest values). Rejected pixels leave the variance
with the data. A simulated stack scatters as the variance says, case by case
(`tests/test_combine_variance.py`).

Each input's DQ leaves its flagged pixels out of the combination --
saturated, cosmic rays, bad columns -- unless every frame flags that pixel,
when the result keeps what they all said; a pixel nothing went into is
marked *no data*. `--masktype none` uses every pixel, as IRAF does by
default; IRAF's own images have no DQ, so for them nothing changes.
`--format iraf` writes the result as IRAF reads it ([coming from
IRAF](page:iraf/migration)).
