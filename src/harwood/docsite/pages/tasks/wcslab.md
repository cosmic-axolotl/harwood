IRAF's `images.tv.wcslab`: a labelled grid of world coordinates -- RA and Dec, or
any linear system -- over an image on the display, or as a graph, alone or over
the last one drawn.

```console
$ harwood wcslab wpix.fits 1                    # over frame 1, in the window
$ harwood contour wpix.fits --no-perimeter
$ harwood wcslab wpix.fits --device stdgraph --append      # over the map
$ harwood wcslab wpix.fits --device grid.svg --no-major-grid --title M51
```

The ranges come from twenty points on the image's edges and are rounded to a step
from IRAF's tables -- 5 seconds of time, 1 minute of arc on dev$wpix --, unless
`--axis1-beg`, `--axis1-end` and `--axis1-int` (RA in hours, `13:28:00` is fine)
and the three of axis 2 are all given. The lines of constant RA and Dec are drawn
in segments across the image (`--no-major-grid` leaves ticks at the edges instead,
`--minor-grid` draws the minor lines dotted); where each crosses an edge it is
labelled, in hms and dms, the one nearest the corner of the two labelled sides in
full and the rest by what changes. With the pole in the image the graph is polar
(the pole marked with a +, the RAs around the edges, the declinations along one
RA, `--axis2-dir`), and near-polar when the pole is close.

Labels follow their lines' angle (`--no-rotate` keeps them upright; within 10
degrees of horizontal or vertical they are upright anyway), go outside the axes
(`--no-labout` puts them inside) on the sides IRAF picks from the grid's rotation
or the ones given (`--axis1-side top,bottom`), and the titles go beyond them
(`--title`, `--axis1-title`, `--title-side`).

`--device imd` (the default) draws over the image in the display's frame,
displaying it first when the frame does not hold it; `stdgraph` draws a graph (the
window, or an SVG without a screen); `--append` adds the grid to the last graph
drawn -- `contour --no-perimeter`, as in IRAF's exercises. `--usewcs` takes the WCS
from `--ctype1` ... `--cd2-2` instead of the image's, and with no image at all
over `--log-x1` ... `--log-y2`.

What is IRAF's: on 47 cases -- RA/Dec rotated, transposed, across 0h, at and near
both poles, a ten-degree and a two-milliarcsecond field, ARC and ZEA, linear and
no WCS, wcspars' own WCS and none -- every text wcslab writes, the labels, their
units and the titles, is in IRAF 2.18's place (to 2 of GKI's 32767 units) with its
angle, size and justification, and every vertex IRAF draws, the box, the grid, the
ticks and the pole's plus, is within 2 units of harwood's lines and the other way
round. IRAF's own quirk is kept: a rotated label's units ("h m s", the degree's o)
are written on the number's own line, so "13 28 00 " and "  h  m  s" read
13h28m00s -- mw_v2tranr's rotated branch reads the output where it means the input,
and wl_rotate loses its y. Upright labels have the units half a character above,
as wcslab means them. The same slip, in double, collapses IRAF's grid for a linear
WCS with a rotation (every line of constant axis 1 drawn as a point, nothing
labelled); harwood draws that grid as the WCS says. `remember` (writing the values back to wlpars) is the
result's `axis1` and `axis2` instead.
