# The first night

The [sample night](page:start/sample), from raw frames to magnitudes, in the
order a CCD night is reduced: the bias, the flat, the object frames, then
the stars. Each step is one or two commands; each example has the same steps
as Python and as clicks, in its tabs. Write the night and go into it first:

```console
$ harwood docs --sample night
$ cd night
```

## 1. Look at what there is

`ccdlist` reads the headers and says what each frame is: the type from
`IMAGETYP`, the filter from `FILTERS`, the size, the exposure.

```example
title: What the night holds
run: harwood ccdlist @bias.lis @flat.lis @obj.lis
output: |
  bias001.fits
  obj003.fits
```

## 2. The bias

A bias frame is a zero-second exposure: it holds the electronic offset the
amplifier adds to every pixel, and nothing else. Each one is first trimmed
of its overscan strip, as every frame will be, and then the three are
combined into one, `Zero.fits`, with less noise than any of them.

```example
title: Process and combine the bias frames
run:
  - harwood ccdproc @bias.lis --overscan --trim --output bias
  - harwood zerocombine bias/bias00*.fits Zero.fits
output: |
  -> Zero.fits
```

`zerocombine` is [imcombine](task:imcombine) with IRAF's settings for
bias frames: it drops the highest pixel of each stack (`minmax`, `nhigh=1`)
before averaging, so a cosmic ray in one frame does not reach the master.
`harwood lpar zerocombine` lists those values and says that they came from
the preset.

## 3. The flat

A dome flat is a uniformly lit exposure: it shows how each pixel responds
compared with the rest -- the vignetting in the corners, the pixel-to-pixel
pattern. The flats get the same overscan and trim, and the bias is taken off
them, before they are combined.

```example
title: Process and combine the flats
after: zero
run:
  - harwood ccdproc @flat.lis --overscan --trim --zerocor --zero Zero.fits --output flat
  - harwood flatcombine flat/flat00*.fits Flat.fits
output: |
  V: 3 images -> FlatV.fits
```

```note
`Flat.fits` was asked for and `FlatV.fits` was written. `flatcombine`
combines each filter separately and puts the filter in the name, as IRAF
does, so that a night with B and V flats gets `FlatB.fits` and `FlatV.fits`
from one command.
```

## 4. The object frames

Now the frames of the sky get everything: overscan, trim, bias, flat. Before
running it, `--plan` says what it will do with the real values -- which
overscan strip, which bias, which flat -- and does nothing. Add `--explain`
and each step also says why it is there.

```example
title: See the steps first
after: flat
run: harwood ccdproc @obj.lis --overscan --trim --zerocor --flatcor --zero Zero.fits --flat FlatV.fits --plan
output: |
  overscan: legendre order 1, 1 rejection passes over [105:120,1:100]
  trim: [1:100,1:100]
```

```example
title: Process the object frames
after: flat
run: harwood ccdproc @obj.lis --overscan --trim --zerocor --flatcor --zero Zero.fits --flat FlatV.fits --output red
output: |
  -> red/obj001.fits
```

Each processed frame carries a mark for each step in its header (`OVERSCAN`,
`TRIM`, `ZEROCOR`, `FLATCOR`), and harwood will not apply a step twice: give
ccdproc a frame from `red/` with the same options and it reports four steps
skipped, as IRAF does.
Leave out `--output` and the frames go to a directory of their own under
`output/`; see [where results go](page:concepts/outputs).

## 5. Check the result

```example
title: Statistics of the processed frames
after: reduced
run: harwood imstatistics red/obj00*.fits
output: |
  red/obj001.fits │ 10000 │ 434.777 │ 949.948 │ 239.892 │ 17631.3
```

Ten thousand pixels: the 120 × 100 frame trimmed to its 100 × 100 of sky. To
look at a frame, `harwood display red/obj001.fits` puts it in the window,
and `harwood imexamine red/obj001.fits` lets you point at a star and press
`r` for its profile and FWHM, as in IRAF.

## 6. Find the stars

[daofind](task:daofind) looks for peaks of the width you give it. With the
FWHM set and the noise left for it to measure, it finds the stars of the
field and nothing else:

```example
title: Find the stars on obj001
after: reduced
run: harwood daofind red/obj001.fits obj001.coo --fwhmpsf 3
output: |
  sources: 24
```

The field holds thirty stars. The ones it does not separate all lie within
4.5 pixels -- one and a half FWHM -- of a neighbour, close enough that two
stars make a single peak; that is a property of the data, not of the
search.

```warning
Giving `--sigma` lower than the real noise lowers the threshold with it, and
noise peaks start to come out as stars: with `--sigma 8` on this frame,
whose sky noise is about 14.5, daofind reports fifty "stars", twenty-six of
them noise. Leave it out, as here, and daofind measures it (`sigma:` in
what it prints).
```

## 7. Measure them

[phot](task:phot) puts apertures on each star, takes the sky from an annulus
around it, and writes magnitudes. Here two apertures, of 4 and 6 pixels, a
sky annulus from 10 to 15 pixels, the gain of 2 electrons per ADU, and an
arbitrary zero point of 25:

```example
title: Aperture photometry of obj001
after: found
run: harwood phot red/obj001.fits obj001.coo obj001.mag --apertures 4,6 --annulus 10 --dannulus 5 --epadu 2 --zmag 25
output: |
  stars: 24
  measured: 24
```

`obj001.mag` is a table you can read with any tool that reads ECSV, or open
in the window's [Files](page:app/files) panel: one row per star, with the
centre, the sky, and the flux, magnitude and error in each aperture.

## 8. Align and combine

The telescope drifted between the three exposures. [imalign](task:imalign)
centres the three bright stars of `bright.coo` on every frame, measures the
shift of each frame against obj001, and shifts them all onto it:

```example
title: Align the frames, then combine them
after: reduced
run:
  - harwood imalign red/obj001.fits,red/obj002.fits,red/obj003.fits red/obj001.fits bright.coo --output aligned
  - harwood imcombine aligned/obj00*.fits final.fits
output: |
  red/obj002.fits     -2.297 (0.006)      1.421 (0.006)
  red/obj003.fits      1.100 (0.006)     -3.192 (0.006)
  -> final.fits
```

The sample night drifted by (2.3, −1.4) and (−1.1, 3.2) pixels; imalign
measures the shifts that undo it, to a few thousandths of a pixel. It also
trims the frames to the part all three cover, so `final.fits` is a little
smaller than the frames it came from.

## 9. What was done

Every run is in the log, with every value and the command that repeats it:

```example
title: The night's log
after: reduced
run: harwood log
output: |
  ccdproc
```

`harwood log --replay 3` prints the command that repeats run 3 exactly, and
`harwood log ccdproc --full` lists every parameter of every ccdproc run and
where its value came from. The same reduction can now be written as a
[Python script](page:concepts/python), or done again from the window's
[Tasks panel](page:app/clicks).
