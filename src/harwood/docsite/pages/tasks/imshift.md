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
`nearest`, `linear`, `poly3`, `poly5`, `spline3`, `sinc`, `lsinc` and
`drizzle` -- and harwood warns that interpolation correlates the noise, so that an error estimated
from the shifted frame afterwards comes out too small. Past the edges the
image is extended by `--boundary-type` (`nearest`, `constant` with
`--constant`, `reflect`, `wrap`). `--shifts-file` gives one `x y` line per
input image instead of one shift for all. The world coordinates move with
the pixels.

`sinc` is a tapered sinc 31 pixels wide (`sinc21` for 21); in imshift
`lsinc` is the same, tabulated once for the shift. `drizzle` averages each
output pixel over the input pixels it covers -- flux-preserving, no
ringing -- and `drizzle[0.5]` shrinks the footprint to half a pixel. Every
interpolant but sinc writes what IRAF 2.18's imshift writes, pixel for
pixel.

sinc is the exception, because IRAF's is broken: `t_imshift.x` hands the
fractional shift to the sinc set-up as a double where it reads a real, so
the kernel is built for the low four bytes of the shift read as a float --
4.2 × 10⁻⁸ for a shift of 0.3, −1.1 × 10⁸ for 0.45 -- and a frame of values
near 130 comes out with values of 10⁵. harwood builds the kernel for the
actual shift. The tests put IRAF's mistake back and get IRAF's output
exactly, which is what shows the rest is IRAF's code.
