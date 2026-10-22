Arithmetic pixel by pixel: `+`, `-`, `*`, `/`, between two images, an image
and a number, or two lists matched one to one. A single image or number as
the second operand applies to every image of the first. Images must be the
same shape -- two frames of different sizes are not two frames of the same
thing -- and the result is written as 32-bit floating point whatever came
in, because a reduced frame holds fractions of a count and negative
residuals. Where a division would be by zero, `--divzero` is written.

```example
title: One exposure minus the next
after: reduced
run: harwood imarith red/obj001.fits - red/obj002.fits diff.fits
output: |
  -> diff.fits
```

```example
title: Counts per second, for a whole list
run: harwood imarith @obj.lis / 60 %obj%rate%
output: |
  -> rate001.fits
```

The result is named like any output: one name per input, a directory, an
`@list`, or IRAF's `%old%new%` template -- `%obj%rate%` turns `obj001.fits`
into `rate001.fits`. The operation goes into the result's HISTORY.

Images that carry a variance give the result one: sums and differences add
the variances, products and quotients propagate them to first order, and a
number counts as exact. The DQ is whatever either operand said, and a pixel
divided by zero is marked *no data* as well as set to `--divzero`.
