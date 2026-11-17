Makes a frame read out from a part of the chip, from a full one -- IRAF's `subsection`,
of ccdtest: what a detector read out in part gives, with `CCDSEC` saying where the
frame sits on the chip, so that the calibrations of the whole chip still apply.

```example
title: A frame of the middle of the chip
run:
  - harwood artobs artobj 60 object
  - harwood subsection sub001 artobj
output: |
  -> sub001.fits
```

The new frame is `--ncols` by `--nlines` of the full one; `--ccdsec` of the full frame
is copied into its `--datasec`, and its `--biassec` replaced by a fresh bias, at
artobs's level and noise unless `--biasval` and `--sigma` say otherwise. The sections go
into the header through the instrument's translation; an empty `--trimsec` deletes the
keyword. The frames are IRAF 2.18's, bit for bit.
