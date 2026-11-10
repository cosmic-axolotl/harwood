Smooths images with the mean over a box of `xwindow` by `ywindow` pixels, as
IRAF's `imfilter.boxcar` does.

```console
$ harwood boxcar im010.fits bim010.fits 3 3
$ harwood boxcar @inlist @outlist 3 3
$ harwood boxcar 'obj*.fits' '%obj%b%' 5 1 --boundary reflect
```

The second is the course's, an @list of inputs and one of outputs, one each.

The mean is kept as IRAF keeps it, running sums in single precision -- a
column sum that adds each new line and drops the one leaving, and along the
line a sum that adds the pixel coming and drops the one going -- so the last
digits are IRAF's, not those of a sum done afresh. Unlike
[median](task:median), the box is not made odd: an even side reaches a pixel
further one way than the other (xwindow/2 to the left and xwindow/2 - 1 to
the right; ywindow/2 below, ywindow/2 - 1 above), and shifts the image by
half a pixel, as IRAF's does.

Past the image's edge the box finds the boundary extension -- `nearest`,
`constant` (in the image's pixel type), `reflect` (no further than the image
is wide) or `wrap` -- and off an image section, the image's own pixels, read
as [median](task:median) reads them. The output is in the input's pixel
type: a mean in a short image is truncated.

```example
title: Smooth a frame with a 3 x 3 box
after: reduced
run:
  - harwood boxcar red/obj001.fits box.fits 3 3
output: |
  -> box.fits
```

The mean of the frame stays what it was, 434.777, and the brightest pixel of
its stars falls from 17631 counts to 12237: a box spreads a star's light over
its neighbours where a median removes it.

Every case compared -- odd and even boxes, boxes larger than the image, the
four boundaries, sections, 1-D images, single columns, short, real and double
-- gives IRAF 2.18's file pixel for pixel; a cube is refused, as IRAF refuses
it.
