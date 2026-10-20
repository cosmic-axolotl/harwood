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

## Marking the sky by hand

`--salgorithm histplot` draws the histogram of each annulus and lets you
mark its peak; `radplot` draws the annulus' pixels against their distance
from the star and lets you mark the level -- IRAF's way of measuring a few
stars in a complicated region carefully, or of checking what another
algorithm did. In the window, click (or press any key but `q`) where the
sky is, as often as you like, and `q` takes the last mark; `q` at once
keeps the first guess, the median for histplot and the mean for radplot.
Without a screen the marks come from `--gcommands`, IRAF's graphics cursor
file: `x y 1 key` lines, each star reading on until a `q`. With neither,
every star is flagged `208`, as IRAF flags it in batch.

The values these give have not been compared with IRAF's: IRAF draws them
only on a graphics terminal, which could not be opened where harwood was
checked. The histogram, the first guess and the moments are the same
routines the other algorithms use, which are.

`--salgorithm file` reads the sky from a file, which is
[phot](task:phot)'s `--skyfile`; fitsky has none, and flags every star
`209`, as IRAF does.
