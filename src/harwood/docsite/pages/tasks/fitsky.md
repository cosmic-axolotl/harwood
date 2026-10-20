Measures the sky around each position, as [phot](task:phot) does before it
sums an aperture: the pixels of an annulus from `--annulus` to
`--annulus + --dannulus`, rejected and estimated by IRAF's algorithms --
the centroid of the histogram (IRAF's default), the mode, median, mean, a
Gaussian fit, and the rest. The positions are used as given, not
recentred.

```example
title: The sky around every star of obj001
after: found
run: harwood fitsky red/obj001.fits obj001.coo obj001.sky --annulus 10 --dannulus 5
output: |
  stars: 24
```

For each position the table has the level (`msky`), the scatter (`stdev`),
the skew (`sskew`), how many pixels were used and how many rejected
(`nsky`, `nsrej`), and IRAF's code for anything that went wrong (`sier`).
The sky is measured locally rather than once for the frame because it is
not flat -- there are gradients, scattered light, and the wings of
neighbours -- and the annulus starts outside the aperture so a star does not
measure its own background.
