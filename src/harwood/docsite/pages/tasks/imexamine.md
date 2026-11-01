Point at the image and press a key; the answer goes to the Examine panel
and the session log, and to `--logfile` when one is named.

| key | | key | |
|---|---|---|---|
| `r` | radial profile and FWHM | `j` | Gaussian fit along the line |
| `a` | aperture photometry and FWHM | `k` | Gaussian fit along the column |
| `m` | statistics of a box | `h` | histogram of a box |
| `l` | plot the line | `e` | contour plot of a box |
| `c` | plot the column | `s` | surface plot of a box |
| `x` | coordinates and value | `z` | the 11 × 11 pixels around it |
| `b` | box of two positions, `c1 c2 l1 l2` | `y` | the origin `x` then measures from |
| `v` | vector plot between two positions | `u` | vector plot centred on a position |
| `,` | quick photometry | `.` | quick radial profile and fit |
| `t` | a section around it as a new image | `o` | overplot the next graph |
| `f` | redraw the last graph | `w` | start or stop the logfile |
| `d` | load an image, in the window | `g`, `i` | the cursor: both are live here |
| `n`, `p` | next and previous frame | `?` | list the keys |
| `q` | end the task | `:` | a colon command: `:radius 7` sets, `:radius` shows |

`b`, `u` and `v` take two positions: press the key at each, or give two lines in a
cursor file. `:naverage` and `:boundary` go with the vector, the width of its strip and what
is outside the image. `t` writes `--ncoutput` by `--nloutput` pixels as `root.001.fits`, with
the root from `--output` or else the image's name, keeping its header but not its WCS. `,` and
`.` are one fit with no iterating of the aperture, so they are quicker than `a` and `r`, and
a colon command that changes what a graph shows redraws it, as IRAF's autoredraw does.

Every star is recentred before it is measured, so the cursor only has to
land near it. The parameters are IRAF's, with IRAF's defaults and meanings:
`radius` is the aperture, `rplot` how far the profile is drawn, and `r` and
`a` widen the aperture to three times the direct FWHM, `iterations` times.
Every key's arithmetic is a port of IRAF's own, and the tests hold what `r`, `a`, `m`,
`j`, `k`, `z`, `b`, `x`, `y`, `,` and `.` print to what IRAF 2.18 printed for the same pixels,
the images `t` writes to the last bit, and the curves of `l`, `c`, `u`, `v` and `.` to its
metacode. IRAF's `t` also complains of a printf of its own when a log is open; that is not
reproduced.

A section that passes the bottom or the top of the image writes no image, in IRAF and here: IRAF's
task dies of it at the bottom, or of one wholly outside, and here `t` warns and the session goes
on. A section that passes a side is written up to the side. IRAF keeps the width it was asked for
and fills the rest with what is behind its buffer in memory, which changes from one read to the
next and is no part of the image.

## Without a screen

Put the positions in a cursor file -- one `x y [frame] [key]` per line --
and pass it as `--imagecur`: the same code answers, so the numbers are the
ones the window would have shown.

```example
title: The profiles of the two brightest stars
after: reduced
run: harwood imexamine red/obj001.fits --imagecur stars.cur
output: |
  MFWHM=2.95  direct=7.46
  MFWHM=3.01  direct=2.93
```

## Reading the line

The line reads as IRAF's: radius, magnitude, flux, sky, peak, ellipticity,
position angle, Moffat beta, and IRAF's three FWHMs -- of a fit to the
enclosed flux, of a fit to the pixels (`MFWHM` for a Moffat, `GFWHM` for a
Gaussian), and the direct one, measured without a model. The example shows
why there are three: the second star's are all about 3, the FWHM the sample
night was made with; the first star has a fainter neighbour seven pixels
away, inside its profile, and while the fit still says 2.95 the direct
measure, which takes the light as it comes, says 7.46. Two measures that
disagree are the sign to look at the star.

Colon commands are IRAF's: a name shared by several of IRAF's psets
(`width`, `rplot`, `naverage`, `ncolumns`, `floor`…) changes the one of the
last graph drawn, and `:unlearn` resets it. In a cursor file they go after
the frame, `612 540 1 :radius 7`, and act on the lines that follow.
