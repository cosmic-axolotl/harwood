Copies images, pixels and header together. A section copies part of a
frame, which is how a subraster becomes an image in its own right:

```example
title: Cut out a corner
after: reduced
run: harwood imcopy 'red/obj001.fits[20:60,20:60]' cut.fits
output: |
  41x41
```

Sections are IRAF's, 1-based, (x, y), inclusive: `[20:60,20:60]` is 41
pixels on a side. `[*,1:50]` is every column of the first fifty lines,
`[-*,*]` flips the image left to right. A section also moves the world
coordinates, so the cut-out still knows where it is on the sky. The copy
records where it came from in its HISTORY.

The copy keeps the image's pixel type, as IRAF's imcopy does: a raw frame of
16-bit integers stays 16-bit, unsigned ones stay unsigned, a double stays
double; bytes become short integers and integers scaled by BSCALE or BZERO
become real, as IRAF's FITS kernel reads them. `--pixtype` writes another --
short, ushort, int, long, real or double -- integers truncated toward zero:

```example
title: A raw frame copied as real
run: harwood imcopy obj001.fits objreal.fits --pixtype real
output: |
  objreal.fits
```

The types are held to IRAF 2.18's copies, bit for bit.

Quote a section in the shell: brackets mean something to it.
