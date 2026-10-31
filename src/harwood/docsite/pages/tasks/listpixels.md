IRAF's `listpixels`: one line per pixel, its coordinates and then its value. The
lines go to standard output and nothing else does, so they can go into a file or
another program. On anything larger than a section it is a long list.

```example
title: A small piece of a frame, in the pixels of the whole frame
after: reduced
run: harwood listpixels "red/obj001.fits[10:11,20:20]" --wcs physical
output: |
  10.
```

`--wcs` chooses the coordinates: `logical`, the default, counts the pixels of what
was read, so a section starts at 1; `physical` counts those of the original image
(`LTV`, `LTM`, and the section itself); `world` is what the header's WCS says, and
without one it is the physical. A `TAN` image is computed with IRAF's own formula
and the other celestial projections with astropy's. The world of a `multispec`
image is not read, and the command says so.

An axis cut down to one pixel is no longer an axis: `[*,4]` prints one coordinate.

`--formats` gives a format for each axis, as IRAF's printf reads them:

| | |
|---|---|
| `%d` | the integer part |
| `%f`, `%e`, `%g` | with a width and a precision: `%8.2f`. Alone they print all the digits of a double |
| `%h`, `%m` | sexagesimal, `h:m:s` and `m:s`, degrees as they are |
| `%H`, `%M` | the same after dividing by 15, degrees to hours |

An axis with no format gets `%0.15g`, or with `--wcs world` the `format` the WCS
has for it in its `WAT` cards. A format that cannot be read stops the command before
anything is printed; IRAF prints what it can.

`--verbose` prints IRAF's line with the name and the system before each image.

Line by line it is what IRAF 2.18 prints, in every pixel type, section and system
above. The value of a NaN or an infinity is `INDEF`, and the command says how many
there were: IRAF reads them as 0, and a 0 in the list looks like a measurement.
