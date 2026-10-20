Measures the shift between frames of the same field from the stars
themselves, as IRAF's `imcentroid`: a few stars of a coordinate list, given
on the reference frame, are centred in every image, and each image's shift
is the mean over its stars of *reference minus image* -- the shift that
moves it onto the reference. Guiding errors, flexure and refraction all move
a field by amounts nobody wrote down; this measures them.

```example
title: How far the telescope drifted
after: reduced
run: harwood imcentroid red/obj002.fits,red/obj003.fits bright.coo --reference red/obj001.fits
output: |
  red/obj002.fits     -2.297 (0.006)      1.421 (0.006)    3
```

The table is IRAF's: per image the shift and its error, how many stars were
used, and the internal error -- the scatter of the stars about the mean
shift, which is large when a star was a wrong one or the field rotated. It
also prints the section every frame has in common (`#Trim_Section`), which
is what [imalign](task:imalign) trims to.

Without `--shifts`, the first star of the list is found in each image with
a coarse box (`--bigbox`) to get the shift roughly, and then every star is
centred in a fine one (`--boxsize`). For frames that moved more than the
big box, give the rough shifts in a file, one `x y` per image.
