Replaces the pixels a mask marks by interpolating across them, as IRAF's
`proto.fixpix` does: across the narrowest side of each bad region, linearly
between the good pixels either side, with the edge taking its one good
neighbour. The arithmetic is a port of IRAF's own (xtfixpix), checked pixel
for pixel against IRAF 2.18 on a mask of every shape it decides differently
on.

The mask is non-zero where a pixel is bad: one for every image, or one
each. It is named the way IRAF names masks: a FITS image, IRAF's own `.pl`
(read without IRAF), `BPM` for the file the image's BPM keyword names (or
`!KEY` for any keyword), a section such as `[120:122,*]`, or a text file of
`x y` and `x1 x2 y1 y2` lines -- where, as in IRAF, a region taller than it
is wide is marked 2 and one wider than tall 3. `^` in front inverts it. A
mask of the whole detector is matched to a trimmed frame through LTV, so it
follows the pixels.

A mask value given as `--linterp` forces interpolation along lines and one
given as `--cinterp` along columns, for regions whose shape misleads -- with
a region file, `--linterp 2` interpolates every column-shaped region along
its lines.

`--verbose` names each image and its mask, and with `--pixels` lists every
pixel replaced, as IRAF prints it: x, y, the value it had, the value it got,
and the good pixels it came from.

```console
$ harwood fixpix frame.fits mask.pl --verbose --pixels
FIXPIX: image frame.fits with mask mask.pl
   5    3    140.4   137.65    4    3    6    3
   9    4    168.5 170.3333    9    3    9    6
```

(two lines of the test frame the tests compare with IRAF: pixel 5,3 was
fixed along its line from 4,3 and 6,3, pixel 9,4 along its column from 9,3
and 9,6.)

Each fixed frame gets IRAF's `FIXPIX` keyword (`Oct 20  3:21 Bad pixel file
is badcols.pl`, `FIXPIX02` the next time) and a HISTORY line with the counts.

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
