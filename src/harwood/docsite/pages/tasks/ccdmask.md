IRAF's `ccdmask`: a mask of the pixels that deviate from a local median, from a flat
field, or better the ratio of two flats at different levels, where whatever flat
fields properly has gone and only what does not is left.

```example
title: The bad pixels of a detector
after: reduced
run: harwood ccdmask red/obj001.fits bad.pl --lsigma 10 --hsigma 10
output: |
  -> bad.pl
```

A pixel is bad if it is more than `--lsigma` or `--hsigma` from the median of a box
of `--ncmed` by `--nlmed`, with a sigma from the percentiles of boxes of `--ncsig` by
`--nlsig`. Then sums along the column are tried, from two pixels to the whole length,
which is what finds a charge trap too weak to see pixel by pixel. A good stretch of a
column shorter than `--ngood` goes too.

Each bad pixel's value says the narrower way across its run, for
[fixpix](task:fixpix): `--linterp` (2) along lines, `--cinterp` (3) along columns,
`--eqinterp` (2) when they are equal. The mask is a `.pl` file.

On a frame that fits one block of columns (500000 pixels, about) these are the bytes
IRAF 2.18 writes. Over it, IRAF's second pass leaves the pixel after a run as its buffer
had it, nearly always 0; here it is always 0.
