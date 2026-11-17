Makes an artificial CCD observation, as ccdred's demonstration does (IRAF's `artobs`, of
the ccdtest package): noise, sky, the flat field's slope, the zero level and the dark
current, a bias over the whole frame, bad pixels, and the header ccdproc reads.

```example
title: A bias and an object, as the demonstration makes them
run:
  - harwood artobs artzero 0 zero
  - harwood artobs artobj 60 object --filter V --badpix 'ccdtest$badpix.dat'
output: |
  -> artobj.fits
```

The frame is `--ncols` by `--nlines` short integers. Over `--datasec` go the sky
(`--skyrate` times the exposure), the flat field (a slope, `--flatslope`, multiplied
in) and the zero level and dark (`--zeroval` plus `--darkrate` times the exposure, with
their slopes); over the whole frame, the bias (`--biasval`) and its noise (`--sigma`).
Bad pixels -- `c1 c2 l1 l2` a line, `ccdtest$badpix.dat` being IRAF's -- are set to
`--badval`. The exposure, the type, the sections and the filter go into the header
through the instrument's translation, as [ccdhedit](task:ccdhedit) writes them.

`harwood setinstrument demo` sets these parameters as IRAF's demonstration does, in the
project file. `--imdata` copies an image's data section in as the scene, if the CL's
`access()` finds it: as in IRAF, a FITS image named without its `.fits` is not found,
and nothing is copied. An image that is there is left alone unless `--overwrite`.

The frames are IRAF 2.18's, bit for bit, the noise included: the generator is
[mkimage](task:mkimage)'s, its seed carried from call to call.
