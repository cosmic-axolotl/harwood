IRAF's `mimstatistics`: `imstatistics` through a mask, with IRAF's lines. In a mask
given with `--imasks`, 0 is a good pixel; a `^` before the list turns that over. There
can be no mask, one for all the images, or one for each. `--lower`, `--upper` and
`--nclip` work on the pixels the mask leaves.

`--omasks` writes which pixels did not count, as 1, with the limits of the last
clipping, as a `.pl` mask (`.pl` is added) or, for a name ending in `.fits`, a FITS image.

Line by line it is what IRAF 2.18 prints. A NaN is left out, where IRAF counts it as 0.
