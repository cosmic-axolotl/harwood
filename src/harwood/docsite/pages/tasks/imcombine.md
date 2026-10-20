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
