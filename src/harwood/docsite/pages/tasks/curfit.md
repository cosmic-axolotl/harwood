IRAF's `curfit`: a function fit to a list of points or to the projection of an image, with the
coefficients, their errors and the fit at each point. The digits are IRAF's, in `real` and in
`double`.

```example
title: A cubic through the rows of an image
after: reduced
run: harwood curfit red/obj001.fits --axis 2 --order 4 --calctype real --power
output: |
  # function = legendre
```

A file with text is a list: `x y`, and a weight in a third column, or only `y`, when `x` counts
the points. A list is put in order of `x` first. Anything else is an image, or its section,
averaged over the axes other than `--axis`. `--weighting` is `uniform`, `statistical` (`1/y`),
or for a list `user` (the third column) or `instrumental` (the third column is a sigma, and the
weight `1/sigma**2`). `--listdata` prints only the points and `--verbose` adds them to the rest.
`--power` turns a legendre or a chebyshev into coefficients of powers of `x`.

`--sample`, `--naverage`, `--low-reject`, `--high-reject`, `--niterate` and `--grow` are what in
IRAF only the interactive session sets, with `:sample` and the rest, and they give what the
session gives after `f` and `q`: 36 sessions were compared. As in IRAF, the rejection does
nothing until `--niterate` is more than 0.

`--interactive` opens that session in the icfit panel, with IRAF's keys, before the output; or
`--cursor` takes them from a file, `x y 1 key` a line or just the key, and 7 of those were
compared, with `d`, `u`, the colon commands and a `q` with no `f`. IRAF has `interactive=yes` by
default and harwood `no`: a task that opens a window has to be asked to.
