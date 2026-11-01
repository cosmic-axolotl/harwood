The sky of an image as a surface, where [fitsky](task:fitsky) measures an annulus around each
star. It is what SExtractor does (Bertin & Arnouts 1996): IRAF has no task for it.

```console
$ harwood skymodel obj001.fits --output obj001.sky.fits --subtracted obj001.sub.fits --rms obj001.rms.fits
```

The image is cut into `--box` pixels a side. In each box the values are clipped at 3 sigma about
the median until they stop changing, and the level is 2.5 median - 1.5 mean, or the median where
mean and median disagree by more than 0.3 sigma, which says that the box is crowded. A box a mask
takes more than half of (`--exclude`) goes by its neighbours, and the pixels flagged in the DQ are
left out. The grid goes through a median filter of `--filter-size` boxes and a bicubic spline.

Take a box larger than a star and smaller than what the sky does. The result goes on to
[psfphot](task:psfphot) and [zogy](task:zogy), which build their own the same way.

Not IRAF's, so it has no IRAF to compare with: it is held box for box to photutils' `Background2D`
with its `SExtractorBackground`, and to a varying sky known beforehand under 300 stars, where it
is within a quarter of the noise of one pixel.
