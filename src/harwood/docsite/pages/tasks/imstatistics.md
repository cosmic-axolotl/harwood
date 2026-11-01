Statistics of the pixels, one row per image, with the fields you ask for:
`image`, `npix`, `mean`, `midpt`, `mode`, `stddev`, `skew`, `kurtosis`,
`min`, `max`. The arithmetic is IRAF's, field by field, and is checked
against IRAF's own output: the median (`midpt`) and the mode come from a
histogram with bins a tenth of a standard deviation wide (`binwidth`), and
the mode is the peak of that histogram refined by a parabola.

Which one to read:

- the **median** resists a bright star where the **mean** does not;
- the **mode** is the most common value -- the sky, on a frame mostly of
  sky, and what a flat is normalised by;
- the **standard deviation** is the noise there is to work against.

```example
title: The sky of a processed frame, with the stars clipped out
after: reduced
run: harwood imstatistics red/obj001.fits --fields image,mean,midpt,mode,stddev --nclip 3
output: |
  red/obj001.fits │ 293.602 │ 287.818 │ 289.116 │ 32.6894
```

Without clipping, the thirty stars pull the mean up to 434.8 and the
standard deviation to 950; three passes of clipping at three sigma leave
the sky. `--lower` and `--upper` leave pixels out altogether (saturated
ones, say), and a section measures part of a frame:

```example
title: A patch of sky
after: reduced
run: harwood imstatistics 'red/obj001.fits[60:90,10:40]' --fields image,npix,midpt,stddev
output: |
  961
```
