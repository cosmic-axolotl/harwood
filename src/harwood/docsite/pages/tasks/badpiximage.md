IRAF's `badpiximage`: an image of the bad pixels listed in a file, with the size and
the header of a template, to blink against the data or use in arithmetic. The file is
the one for `text2mask` and `ccdproc`: column line, or first and last column and line.

Its sense is not a mask's: a good pixel is `--goodvalue` (1) and a bad one `--badvalue`
(0), in a short image. A name ending in `.pl` makes a mask file instead of a FITS image.

A comment with the word `untrimmed` says the coordinates are of the frame before its
trim, and they are moved by the section in the header's `TRIM`. This is what IRAF 2.18
means to do and does not: it stops with "Error in image section specification", having
read the section from the wrong place. It is otherwise what IRAF writes.
