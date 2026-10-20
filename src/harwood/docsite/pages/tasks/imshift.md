Shifts images by a whole or a fractional number of pixels, as IRAF's
`imshift`: positive moves the image towards +x and +y.

```example
title: Move one frame onto another
after: reduced
run: harwood imshift red/obj002.fits --xshift -2.297 --yshift 1.421 --output shifted.fits
output: |
  -> shifted.fits
```

A whole-pixel shift moves values without touching them, whatever the
interpolant. A fractional one resamples: `--interp-type` is IRAF's --
`nearest`, `linear`, `poly3`, `poly5`, `spline3` -- and harwood
warns that interpolation correlates the noise, so that an error estimated
from the shifted frame afterwards comes out too small. Past the edges the
image is extended by `--boundary-type` (`nearest`, `constant` with
`--constant`, `reflect`, `wrap`). `--shifts-file` gives one `x y` line per
input image instead of one shift for all. The world coordinates move with
the pixels.
