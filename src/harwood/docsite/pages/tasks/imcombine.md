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

`--weight ivar`, which IRAF does not have, weights each pixel by the inverse of its variance,
for the lowest noise any average of them can have, and the variance of the result is
1 / sum(1 / var). It needs every image to carry a VAR, as ccdproc's do, and the average.

`--scale`, `--zero` and `--weight` also take `@file`, a number for each image, or
`!KEYWORD`; a scale from those is a factor to multiply by and a zero an offset to
add, where a measured one is a level. `--gain`, `--rdnoise` and `--snoise` take a
keyword as well as a number, for images that differ.

The combined image names every input in `IMCMB` cards. `--sigmas` writes
the standard deviation at each pixel, `--nrejmasks` how many values were
rejected there, `--bpmasks` where nothing was combined, `--rejmasks` a plane for each
image with what it did not give, and `--expmasks` the exposure each pixel got, each by
the name asked for, a `.pl` or a FITS image. `--ccdtype` and `--subsets` pick frames by
type and combine each filter apart, as the presets do.

`--combine quadrature` combines images of errors, the square root of the sum of the
squares over n; `--combine nmodel` works out what the noise of the average would be
from `--rdnoise`, `--gain` and `--snoise`. IRAF gets each value's model from its place
in the list rather than its image, which only shows when they differ, and so does this.

## Frames that do not line up

```
harwood imcombine frames/*.fits stack.fits --offsets wcs --masktype goodvalue \
    --reject crreject --gain GAIN --rdnoise RDNOISE --expmasks exposure.pl
```

`--offsets` says where each image starts in an output that holds them all: `wcs` or
`world` from their world coordinates, `physical`, `grid nx dx ny dy`, or a file with `x y`
for each, rounded to whole pixels, with a `# absolute` line if they are not relative to
one another. There is no interpolation: for subpixel alignment `imalign` or `imshift` come
first. `--outlimits x1 x2 y1 y2` keeps part of the output, and images of different sizes
with no offsets go from the first pixel. Each output pixel combines the images that reach
it, and its WCS follows the first image. `--statsec` may start with `input`, `output` or
`overlap` to say where a section is measured, and `--project` takes the planes of a cube.

## Pixel masks

`--masktype` reads the mask each image's `BPM` keyword names, or another with `!KEYWORD`:
`goodvalue` keeps the pixels with `--maskvalue`, 0 by default, `badvalue` leaves them out,
`goodbits` and `badbits` go by their bits, and `<v` or `>v` as the value keeps what is below
or above it. `novalue` calls the pixels with `--maskvalue` no data at all, and the others
bad, but used if a pixel has nothing better. A mask is matched pixel by pixel, whatever
the LTV, and is not used through `pmmatch`. Images with no mask are all good.

All of IRAF's is compared with IRAF 2.18, bit by bit, on 103 cases. One difference, on purpose:
`--project`'s output keeps no card of the axis it took away, where IRAF keeps them behind a
`WAXMAP`, which only IRAF reads.

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
