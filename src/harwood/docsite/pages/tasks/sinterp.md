Interpolates a table of x and y points: at the x's a file asks for, along a
range, or into a spectrum whose wavelengths are the x's.

```console
$ harwood sinterp table.dat --input points.dat --interp-mode linear
       2500.       0.89696
       3000.            1.
      3125.5        1.0259
$ harwood sinterp table.dat --curve-gen --x1 3100 --dx 250 --npts 24 --interp-mode curves
$ harwood sinterp table.dat --make-image --image model --x1 3000 --x2 9000 --npts 200
```

The table is lines of x and y; the others are skipped. `--interp-mode` is
`linear`, a straight line through the two points around x; `curves`, IRAF's
intrp, the two parabolas through the points around it blended by how far
apart they are; or `legendre`, `chebyshev`, `spline3`, `spline1`, a fit of
`--order` through every point. Beyond the table, linear and curves continue
the end pieces and a fit is its own function.

Without `--curve-gen` or `--make-image` each line of the input file (`STDIN`
by default) gives an x, and its value is said; with either, the range is
`--x1` and two of `--x2`, `--dx` and `--npts`, and the points are said or made
into a spectrum. Everything is in real, as IRAF's: nine sessions IRAF 2.18 ran
-- every interpolant at x's inside, on and beyond the table, ranges given
three ways, a spectrum -- print the same to the character, and the spectrum's
wavelengths are IRAF's world coordinates, its start and step written to the
header with a real's seven figures, as IRAF writes them.
