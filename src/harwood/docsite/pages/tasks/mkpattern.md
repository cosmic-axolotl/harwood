IRAF's `mkpattern`: an image, or part of one, set to a simple pattern, or the
pattern added or multiplied in. The image is created if it is not there.

```example
title: A test image of checks
run: harwood mkpattern check.fits --pattern checker --size 8 --v1 100 --v2 200 --ncols 64 --nlines 64
output: |
  -> check.fits
```

| | |
|---|---|
| `constant` | `--v1` |
| `grid` | lines of `--v2` every `--size` pixels on a ground of `--v1`; a size of at least 2 |
| `checker` | squares of `--size` of `--v1` and `--v2` |
| `coordinates` | the pixels numbered, columns fastest |
| `slope` | a plane from `--v1` towards `--v2`, in steps of `--size` |
| `square` | a checker whose squares grow with the square of the coordinate |

`--option` is `replace`, `add` or `multiply` for an image that exists, and a section
changes only that part: `mkpattern "x.fits[10:20,*]" --pattern constant --v1 0`. With
`--output` the input stays as it is. A new image has `--pixtype`, `--ndim`, `--ncols`,
`--nlines`, `--n3`, `--n4`, `--title` and the keywords of `--header`, an image or a file of
FITS cards.

These are the pixels of IRAF 2.18. A cube repeats the pattern of its first plane, but for
its last line in each, which IRAF computes as if it were the first; a one-dimensional
image is all first line too. Complex pixels and more than four dimensions are not here.
