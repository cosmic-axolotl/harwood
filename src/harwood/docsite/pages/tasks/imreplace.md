Sets the pixels whose values fall in a window to one value, as IRAF's
`imutil.imreplace` does: from `--lower` to `--upper`, both included, a side
left out having no limit and neither side meaning every pixel. With
`--radius`, every pixel within that many pixels of one in the window gets the
value too -- the edge of a bad region, which is bad without being far off.

```console
$ harwood imreplace teste.fits 0 --lower 0 --output teste.fits
$ harwood imreplace cuar_lower.fits 0 --lower -201 --output cuar_lower.fits
$ harwood imreplace 'frame.fits[1:40,*]' --upper 0 --output frame.fits -- -1
```

A negative value goes after `--`, as any negative positional does. A section
changes only its own pixels; the rest of the file is written as it was.

What the window is depends on the pixel type, as in IRAF, and harwood keeps
IRAF's answer to the pixel. In an integer image the limits are made integers:
upper truncated, lower truncated when it is whole and raised by one when it
is not -- so for a negative limit that is not whole the window moves by one
(`--lower -100.5 --upper -20.5` is -99 to -20 in a short image), and the
value is truncated toward zero (2.7 is 2, and a value too large for the type
wraps, as IRAF's conversion does). A real image compares in single
precision, a double one in double, against limits that are the CL's reals.
Every pixel read comes through IRAF's IEEE input, which reads a NaN, an
infinity, -0 and a number below the type's normal range as 0 -- so they
compare as 0, and are written back as 0.

The radius has IRAF's defects too. Its buffer marks a pixel by setting it to
INDEF, so in a short image a pixel that is -32767 is replaced wherever it is;
without `--lower` the window starts at -32767, so -32768 never is. A line is
written only once the line `int(radius)` further on has been read: a 1-D
image, or one with no more lines than the radius, is left as it was, and in a
cube the last lines of the last band are never written. harwood does the
same and says so.

```example
title: Blank the stars of a frame, and their wings
after: reduced
run:
  - harwood imreplace red/obj001.fits 0 --lower 2000 --radius 3 --output stars_
output: |
  red/obj001.fits │ real │ 1051
  -> stars_obj001.fits
```

218 pixels of obj001 are above 2000 counts; with their neighbours within 3
pixels, 1051 are set to 0, which leaves the sky for a statistic that the
stars would otherwise pull up.

Every case compared -- short, ushort, int, real, double, 1-D and a cube; each
kind of window; values that truncate and wrap; radii from 0.5 to past the
image's height; sections; NaN, infinities, -0 and subnormals -- gives IRAF
2.18's file, pixel for pixel, BITPIX and BZERO included. IRAF cannot write
an 8-bit or a scaled image back in place and stops; harwood writes those as
the type IRAF reads them as, short and real, and says so. IRAF edits in
place; harwood writes to the run's directory unless `--output` names the
input. The pixel type is kept, double included.
