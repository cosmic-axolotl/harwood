Shifts a set of frames so that their pixels line up, as IRAF's `imalign`
script does: [imcentroid](task:imcentroid) measures each frame's shift
from the stars of a coordinate list, [imshift](task:imshift) applies it,
and the frames are trimmed to the part all of them cover.

```example
title: Put the three exposures on the same pixels
after: reduced
run: harwood imalign red/obj001.fits,red/obj002.fits,red/obj003.fits red/obj001.fits bright.coo --output aligned
output: |
  red/obj003.fits      1.100 (0.006)     -3.192 (0.006)
  -> aligned/obj001.fits
```

The sample night drifted by (2.3, −1.4) and (−1.1, 3.2) pixels between
exposures; imalign measures the shifts that undo it to a few thousandths of
a pixel, applies them, and trims the three frames to `[3:97,3:96]`, the
part all of them still cover. The shifts it applies are the ones
imcentroid prints, to a thousandth of a pixel, as in IRAF, which reads them
back from that table.

- `--no-shiftimages` only measures; `--no-trimimages` keeps the full frames,
  with the edges extended by `--boundary-type`.
- A fractional shift interpolates (`--interp-type`, linear by default), and
  harwood says so: interpolation mixes neighbouring pixels and correlates
  the noise, so the sky of an aligned frame looks quieter than it is. For
  photometry, measure the frames as they are and shift the coordinates
  instead.
- Each frame records its shift in HISTORY, and its world coordinates move
  with the pixels.
