IRAF's `text2mask`: a pixel mask from a text file. A line of two numbers is a pixel,
column and line; a line of four is a rectangle, first and last column, first and last
line. Anything else is passed over, a comment included, and regions are cut at the
edges of the mask.

```
12 40
100 100 1 512
```

`harwood text2mask bad.txt bad.pl 512 512` makes of that a mask with pixel 12, 40 and
column 100.

The value says the shape, so that [fixpix](task:fixpix) interpolates across the narrow
way: `--linterp` (1) for a rectangle narrower along lines, `--cinterp` (2) narrower along
columns, `--square` (3) and `--pixel` (4). They are shorts, as in IRAF, so 70000 is 4464.

The mask is a `.pl` file, which IRAF reads, and `.pl` is added to its name.

Regions of one line are the bytes IRAF 2.18 writes. A rectangle of several lines is
whole here: IRAF's own keeps its first line and a little of the second, because its
short pixels go wrong on the way into a mask of ints.
