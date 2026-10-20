IRAF's `imhistogram`, listed: the centre and count of each bin, between
`z1` and `z2` (the image's own minimum and maximum unless given), in
`nbins` bins. Integer images get IRAF's treatment -- limits rounded to
integers and, with autoscale, a bin width that is a whole number of values
-- decided by the image's pixel type, not by its values.

```example
title: Where the pixels of a frame lie
after: reduced
run: harwood imhistogram red/obj001.fits --nbins 5
output: |
  nbins: 5
```

Most pixels of a frame of sky fall in the first bin; a long tail of
brighter bins is the stars. Narrow the range (`--z1`, `--z2`) to look at the
sky itself.
