Recentres a list of stars, as IRAF's `center`: from each starting position
it finds the centre of the light in a box `--cbox` wide, by IRAF's centring
algorithms -- the marginal centroid (the default), a Gaussian fit
(`gauss`), or the optimal filter (`ofilter`) -- iterating until the centre
settles.

```example
title: Better centres for the stars daofind found
after: found
run: harwood center red/obj001.fits obj001.coo obj001.ctr --cbox 5
output: |
  flagged: 0
```

The table keeps where each star started (`xinit`, `yinit`), where it ended
(`x`, `y`), the shift and its error, and IRAF's code for anything flagged
(`cier`): too faint (`--minsnratio`), moved further than `--maxshift`, too
near the edge. Its `x` and `y` are the new centres, so it can be handed
straight to [phot](task:phot) with `--calgorithm none`.

[phot](task:phot) centres every star itself before measuring it, with the
same parameters, so `center` on its own is for when the centres are what
you want -- to measure a shift, or check a list from another frame.
