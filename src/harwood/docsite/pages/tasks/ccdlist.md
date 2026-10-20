Lists frames as the reduction sees them: the image type and the subset read
through the instrument translation, the size, the exposure, the title -- and,
with `--long`, which corrections each frame has had.

This is the check to run before [ccdproc](task:ccdproc): a frame that comes
back `unknown` is one ccdproc will not know what to do with either, and the
translation (see [setinstrument](task:setinstrument)) is what to fix.

```example
title: The raw night
run: harwood ccdlist @bias.lis @flat.lis @obj.lis
output: |
  bias001.fits
```

```example
title: What has been done to the flats
after: flat
run: harwood ccdlist 'flat/*.fits' --long
output: |
  trim, zerocor
```

`--ccdtype flat` lists only the flats.
