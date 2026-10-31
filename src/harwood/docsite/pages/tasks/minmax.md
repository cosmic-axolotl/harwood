IRAF's `minmax`: the lowest and the highest pixel of each image, and where they
are. The line it prints is IRAF's, position before value:

```example
title: The extremes of a frame
after: reduced
run: harwood minmax red/obj001.fits --force --no-update
output: |
  red/obj001.fits [
```

The pixels are read only when they have to be: if the header already has
`DATAMIN` or `DATAMAX`, those are what is printed, with no position, and that is
how a whole night goes by quickly. `--force` looks at the pixels anyway. The values
found are written to the header unless `--no-update`, to seven figures as IRAF
writes them, and the pixels on disk stay as they were: scaled integers stay scaled
integers. A file that cannot be written is still measured.

Harwood's own tasks do not carry those limits over to a new image, as IRAF's do
not: `imarith` of a frame that had them gives a frame without them.

A section is measured and leaves the header alone, and its positions are in the
section: `[3,4]` of `obj001.fits[10:50,10:50]` is pixel (12, 13) of the image. An
axis cut down to one pixel is no longer an axis, so `[*,4]` gives `[3]`.

| | |
|---|---|
| `--force` | look at the pixels even if the header has limits |
| `--no-update` | leave the header alone |
| `--no-verbose` | print nothing; the values are still in the run's result |

Line by line it is what IRAF 2.18 prints, for every pixel type and the first of
two equal extremes, but in four places, on purpose:

- A section is always measured. IRAF, with limits in the header, prints the
  whole image's for a section, which may hold neither.
- `minmax` on a section never changes the file. IRAF's, with `force` and `update`,
  takes `DATAMIN` and `DATAMAX` out of it.
- A NaN or an infinity has no value, and the command says how many it left out.
  IRAF reads them as 0, and tells you a frame of a thousand counts has a minimum of 0.
- Limits already in the header are not rewritten; an image with no pixels gets no
  limits (IRAF writes `DATAMIN = INDEF`, which is not FITS).

Equal limits, or one missing and the other 0, mean nothing to IRAF, so here too the
pixels are read.
