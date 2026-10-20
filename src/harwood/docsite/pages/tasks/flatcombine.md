Combines flat fields into one master flat per filter. It is
[imcombine](task:imcombine) with IRAF's settings for flats, from ccdred's
`flatcombine.cl`: each frame scaled to a common level by its mode
(`scale=mode`), then averaged with `avsigclip` rejection, of the frames of
type `flat`, **each subset separately** (`subsets=yes`).

`avsigclip` is the right rejection for a few flats: the spread is not
measured from the handful of values at each pixel but pooled along the
whole image line and scaled like photon noise, a far steadier sigma than
three values can give.

```example
title: The master flat
after: zero
run:
  - harwood ccdproc @flat.lis --overscan --trim --zerocor --zero Zero.fits --output flat
  - harwood flatcombine flat/flat00*.fits Flat.fits
output: |
  V: 3 images -> FlatV.fits
```

`Flat.fits` was asked for and `FlatV.fits` was written: with subsets on, the
subset's short name goes into the name, so B and V flats taken in one night
become `FlatB.fits` and `FlatV.fits` from one command. The short names come
from the `subsets` file in the reduction directory, which you can edit to
say `V` rather than whatever the filter wheel reported. `--no-subsets`
combines everything into the one name.

Where every value at a pixel is rejected, 1 is written (`blank=1`), so
dividing by the flat leaves that pixel as it was rather than dividing by
zero.
