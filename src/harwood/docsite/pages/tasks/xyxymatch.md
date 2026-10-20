Pairs up the stars of two coordinate lists, as IRAF's `xyxymatch`: the
input list is mapped onto the reference by a linear transformation, and
each input star is paired with the reference star within `--tolerance`
pixels of it.

The transformation is found one of two ways:

- **`triangles`** (the default) needs nothing to start from: it matches the
  pattern of triangles the brightest stars make (Groth's method), which
  survives a shift, a rotation and a change of scale.
- **`tolerance`** needs the transformation to be right already -- given as
  `--xin`, `--xref`, `--xmag`, `--xrotation`…, or by tie points
  (`--refpoints`) -- and pairs nearest neighbours.

```example
title: Pair the stars of two exposures
after: reduced
run:
  - harwood daofind red/obj001.fits obj001.coo --fwhmpsf 3
  - harwood daofind red/obj002.fits obj002.coo --fwhmpsf 3
  - harwood xyxymatch obj002.coo obj001.coo pairs.txt --tolerance 2
output: |
  pairs: 17
```

The 17 pairs give the telescope's drift between the two exposures, about
(−2.25, 1.42) pixels, with no change of scale or rotation. The output file
is IRAF's, line for line: the transformation, then each pair -- reference x
and y, input x and y, and the line each came from in its list.

Lists are read as IRAF reads them, x and y in columns 1 and 2 unless
`--xcolumn` and `--ycolumn` say otherwise -- except that a catalogue
harwood wrote, which names its columns, is read by its `x` and `y`, since
daofind's puts its id first.
