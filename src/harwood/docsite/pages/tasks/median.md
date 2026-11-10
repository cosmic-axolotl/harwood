Replaces every pixel by the median of a window around it, as IRAF's
`imfilter.median` does: the middle value, which one cosmic ray or one hot
pixel cannot move the way it moves a mean. The window is `xwindow` by
`ywindow`; a side of even size is made odd by adding one, as IRAF makes it,
so a 4x4 is a 5x5.

```console
$ harwood median cuar_btpf.fits cuar_median.fits 7 1
$ harwood median frame.fits clean.fits 3 3 --zhireject 30000
$ harwood median 'obj*.fits' '%obj%med%' 5 5 --boundary reflect
```

The first is the spectroscopy course's: a 7 x 1 window along the lines of an
arc lamp follows the lamp's lines and leaves out its noise, and the frame
minus its median is the residual a bad-pixel mask is cut from.

Values below `--zloreject` and above `--zhireject` stay out of each median.
What is left may be an even number: the median is then the lower of the two
in the middle. With nothing left it is `--zhireject` if more of the window
was above than below, else `--zloreject`.

Near the edge the window reaches past the image, and what it finds there is
the boundary extension: `nearest` (the edge pixel, the default), `constant`
(`--constant`, in the image's own pixel type: 37.5 in a short image is 37),
`reflect` (the image mirrored about its edge pixel, which is not repeated --
so a window can reach no further than the image is wide) or `wrap`. For an
image section the window finds the image's own pixels around the section,
and the extension only beyond the image's edge.

```example
title: Smooth a frame with a 5 x 5 median
after: reduced
run:
  - harwood median red/obj001.fits smooth.fits 5 5
output: |
  5x5 Box median filter red/obj001.fits to smooth.fits
  -> smooth.fits
```

The stars lose their peaks -- the brightest pixel of obj001 falls from 17631
counts to 7116 -- and the sky keeps its level: what a median is for, a
background with the sharp things taken out of it.

The output is in the input's pixel type, as IRAF writes it: a median in a
short image is one of its own values, or a truncated boundary constant or
limit.

The port is literal, IRAF's sorted linked list walked back and forth over
the image, because the list does not always hold the window: in an image no
wider than the window, the row it drops on changing line is the one it last
put in, and the medians are not the window's; and a subsampled section
(`[2:20:3,*]`) near the image's edge is read by IMIO from where its regions
begin, off the section's own grid. Both are IRAF's, and harwood gives them
too: every case compared -- each window shape, the four boundaries, the
limits, sections, 1-D images, single columns, short, real and double --
gives IRAF 2.18's file pixel for pixel.
