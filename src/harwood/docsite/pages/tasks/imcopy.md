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
coordinates, so the cut-out still knows where it is on the sky. The copy is
written as 32-bit floating point and records where it came from in its
HISTORY.

Quote a section in the shell: brackets mean something to it.
