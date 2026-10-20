IRAF's `daofind`, ported: the image is convolved with a Gaussian the width
of the stars (`--fwhmpsf`), and the maxima of the result above
`--threshold` times the noise are stars. A filter matched to the stars
suppresses everything that is not star-shaped -- a cosmic ray, which is
sharper, and a galaxy, which is broader, both respond less than a star of
the same brightness -- and the shape statistics finish the job: sharpness
(`--sharplo`, `--sharphi`) and roundness (`--roundlo`, `--roundhi`).

The output is a coordinate list, 1-based and (x, y) as IRAF writes them,
with IRAF's columns -- position, magnitude estimate, sharpness, roundness --
ready for [phot](task:phot). Left unnamed it goes into the run's own
directory as `obj001.coo`, the frame's name with IRAF's extension.

```example
title: The stars of one frame
after: reduced
run: harwood daofind red/obj001.fits obj001.coo --fwhmpsf 3
output: |
  sources: 24
```

## The two parameters that matter

- **`--fwhmpsf`**, the width of the stars, is the whole filter. Measure it
  rather than guess: point at a star in the window with
  [imexamine](task:imexamine) and press `r`.
- **`--sigma`**, the noise of the sky. IRAF takes it from datapars and,
  left INDEF, detects at a threshold of zero; harwood measures it from the
  image instead -- a sigma-clipped standard deviation, so the stars do not
  inflate it -- and prints what it measured. Give it only if you know it:
  set too low, it lowers the threshold with it, and noise starts to come out
  as stars.

```example
title: A sigma below the real noise finds noise
after: reduced
run: harwood daofind red/obj001.fits noisy.coo --fwhmpsf 3 --sigma 8
output: |
  sources: 50
```

On this frame, whose sky noise daofind measures as 14.5, a sigma of 8
gives fifty "stars", of which twenty-six are noise.

Stars that touch the edge are found -- IRAF extends the image by its
nearest pixel, so the filter is defined to the border -- and then dropped,
since phot would lose their outer apertures; how many were dropped is said.
Stars closer than about one and a half FWHM make a single peak and are
found as one.
