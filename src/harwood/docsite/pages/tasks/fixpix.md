Replaces the pixels a mask marks by interpolating across them, as IRAF's
`proto.fixpix` does: across the narrowest side of each bad region, linearly
between the good pixels either side, with the edge taking its one good
neighbour. The arithmetic is a port of IRAF's own (xtfixpix), checked pixel
for pixel against IRAF 2.18 on a mask of every shape it decides differently
on.

The mask is an image of the same size, non-zero where a pixel is bad: one
for every image, or one each. A mask value given as `--linterp` forces
interpolation along lines and one given as `--cinterp` along columns, for
regions whose shape misleads.

```example
title: Repair the pixels the cosmic rays hit
after: reduced
run:
  - harwood cosmicrays red/obj003.fits --output cr_ --gain 2 --rdnoise 10
  - harwood fixpix red/obj003.fits cr_obj003.mask.fits --output fixed_
output: |
  -> fixed_obj003.fits
```

The mask cosmicrays wrote marks 18 pixels; fixpix interpolates across each
small region along its narrow side -- here 16 along columns and 2 along
lines. For a detector's permanent bad columns, keep one mask for the night
and let [ccdproc](task:ccdproc) apply it (`--fixpix --bpm mask.fits`) before
anything else.
