The fast median of IRAF's `imfilter.fmedian`: each pixel replaced by the
median of the window around it, as [median](task:median) does, but kept as a
running median over a histogram of the pixels made integers first -- which is
quick for any window, and costs resolution.

```console
$ harwood fmedian frame.fits clean.fits 5 5
$ harwood fmedian frame.fits clean.fits 9 9 --hmin 0 --hmax 4095 --zmin 0 --zmax 40000
```

The pixels from `--zmin` to `--zmax` (left out, the image's own minimum and
maximum -- or DATAMIN and DATAMAX when the header has them, believed as IRAF
believes them) are mapped linearly onto the integers `--hmin` to `--hmax`,
by default -32768 to 32767, and truncated. The medians are mapped back
(`--no-unmap` keeps the integers), so the output is quantized to
(zmax - zmin)/(hmax - hmin) of the data: a frame spanning 0 to 65535 counts
comes back in steps of 1, a narrower `--hmin`/`--hmax` gives coarser steps.
When `--zmin` and `--zmax` already are `--hmin` and `--hmax` nothing is
mapped, and an integer image keeps its own values.

The window is made odd, the rejection limits `--zloreject` and `--zhireject`
are mapped and truncated like the pixels, and the median of an even number
left is the lower of the two in the middle -- except the first pixel's,
which is IRAF's amed's: of one or two values, the first. The boundary
extension and image sections are read as [median](task:median) reads them.

```example
title: The fast median of a frame
after: reduced
run:
  - harwood fmedian red/obj001.fits fmed.fits 5 5
output: |
  5x5 Box median filter red/obj001.fits to fmed.fits
  Pixels from 239.8923 to 17631.3 mapped to integers from -32768 to 32767
  -> fmed.fits
```

The same 5 x 5 window as median's, and almost the same image: the brightest
pixel is 7116.04 where median gives 7115.91, the 0.27 counts of one step of
17391 counts in 65535.

Every case compared -- each window shape, the histogram's range and the data's,
unmapped output, the rejection limits, the four boundaries, sections, 1-D
images, short, real and double, an image whose header limits are not its
pixels' -- gives IRAF 2.18's file pixel for pixel, and what verbose prints.
A 1-D image reflected for its range reads, as IRAF does, the lines it does not
have from past the end of its pixels: the zeros that pad a FITS file.
