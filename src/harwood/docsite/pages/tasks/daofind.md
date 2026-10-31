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

**By the keys.** `--interactive` shows the image in the window and hands it
daofind's keys; `--icommands` takes the same keys from a file, `x y wcs key` a
line, without a screen. `f` finds the stars and circles them, and writes nothing;
the space bar finds them and writes the list. `d` is the radial profile of the
star at the cursor, in the Examine panel. `:threshold 6`, `:fwhmpsf 3.1`, `:sigma 5`,
`:nsigma`, `:ratio`, `:theta`, `:sharplo`, `:sharphi`, `:roundlo`, `:roundhi`,
`:epadu`, `:readnoise` and `:scale` set a parameter, or show it with no value, and
`:show` shows them all. `q` ends.

Each list written goes to a file of its own, the output name, then `name.2`,
`name.3`; the space bar finds nothing new until a parameter has changed.

Finding again with the same kernel gives a different sharpness, because IRAF's
code marks the kernel's centre as skipped after the first find and counts one
pixel too few from then on. Only sharpness changes, by a few hundredths for a
bright star and more for a faint one (0.49 to 0.34 in one test), but `--sharplo`
then cuts on it. harwood does what IRAF does, and warns when a list written has it:
after `f`, or after changing only `:threshold`, `:sharplo`, `:sharphi`, `:roundlo`
or `:roundhi`. Changing anything else rebuilds the kernel and starts clean.

Held to IRAF on ten cursor files: every line printed and every list written,
but for `ground`, which agrees to 0.001 as it does without the keys, and the third
file's name, which IRAF takes from the image (`field.fits.coo.1`). `i`, `v` and `w`
are not ported: `r` over a star gives the FWHM, `:fwhmpsf` sets it, and `run.yaml`
keeps what was used. Nor are the colon commands for what harwood's daofind has no
use for, such as `:datamin`.
